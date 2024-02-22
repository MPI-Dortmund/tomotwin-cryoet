"""
Copyright (c) 2022 MPI-Dortmund
SPDX-License-Identifier: MPL-2.0

This file is subject to the terms of the Mozilla Public License, Version 2.0 (MPL-2.0).
The full text of the MPL-2.0 can be found at http://mozilla.org/MPL/2.0/.

For files that are Incompatible With Secondary Licenses, as defined under the MPL-2.0,
additional notices are required. Refer to the MPL-2.0 license for more details on your
obligations and rights under this license and for instructions on how secondary licenses
may affect the distribution and modification of this software.
"""
import glob
import hashlib
import os
import random
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp

import tomotwin
from tomotwin.modules.common.io.mrc_format import MrcFormat
from tomotwin.modules.common.utils import check_for_updates
from tomotwin.modules.inference.argparse_embed_ui import EmbedArgParseUI, EmbedMode, EmbedConfiguration, DistrMode
from tomotwin.modules.inference.boxer import Boxer, SlidingWindowBoxer
from tomotwin.modules.inference.embedor import TorchEmbedorDistributed, Embedor, TorchEmbedor
from tomotwin.modules.inference.volumedata import FileNameVolumeDataset


def sliding_window_embedding(
    tomo: np.array, boxer: Boxer, embedor: Embedor
) -> np.array:
    '''
    Embeds the tomogram using a sliding window approach

    :param tomo: Tomogram
    :param boxer: Box provider
    :param embedor: Embedor to embed the tomogram
    :return: Embeddings from sliding window
    '''
    boxes = boxer.box(tomogram=tomo)
    embeddings = embedor.embed(volume_data=boxes)
    if embeddings is None:
        return None
    positions = []
    for i in range(embeddings.shape[0]):
        positions.append(boxes.get_localization(i))
    positions = np.array(positions)
    embeddings = np.hstack([positions, embeddings])

    return embeddings


def volume_embedding(volume_pths: List[str], embedor: Embedor):
    '''
    Embeds a complete volume (tomogram))
    :param volume_pths: List of tomograms
    :param embedor: Embeddor
    :return:
    '''
    volume_dataset = FileNameVolumeDataset(
        volumes=volume_pths, filereader=MrcFormat.read
    )
    embeddings = embedor.embed(volume_dataset)

    return embeddings


def get_window_size(model_path: str) -> int:
    '''
    Extract the window size from the model, otherwise it returns the default
    :param model_path: Path to model file
    :return: Window size
    '''

    checkpoint = torch.load(model_path)
    if "window_size" in checkpoint["tomotwin_config"]:
        return int(checkpoint["tomotwin_config"]["window_size"][0])
    print("Can't find window size in model. Use window size of 37.")
    return 37


def get_file_md5(path: str) -> str:
    '''
    Calculate md5 checkfsum for file
    :param path: Path for file
    :return: MD5  checksum.
    '''
    try:
        with open(path, "rb") as f:
            data = f.read()
        md5hash = hashlib.md5(data).hexdigest()
        return md5hash
    except TypeError:
        return None

def embed_subvolumes(paths: List[str], embedor: Embedor, conf: EmbedConfiguration) -> pd.DataFrame:
    '''
    Embeds a set of subvolumes
    '''
    embeddings = volume_embedding(paths, embedor=embedor)

    if embeddings is None:
        return
    column_names = []
    for i in range(embeddings.shape[1]):
        column_names.append(str(i))
    df = pd.DataFrame(data=embeddings, columns=column_names)
    df.insert(0, "filepath", paths)
    df.index.name = "index"
    df.attrs["modelpth"] = conf.model_path
    df.attrs["modelmd5"] = get_file_md5(conf.model_path)
    f = os.path.join(conf.output_path, "embeddings.temb")
    df.to_pickle(f)
    print(f"Done. Wrote results to {f}")
    return df


def embed_tomogram(
        tomo: np.array,
        embedor: Embedor,
        conf: EmbedConfiguration,
        window_size: int,
        mask: np.array = None) -> pd.DataFrame:
    """
    Embeds a tomogram
    :return: DataFrame of embeddings
    """

    if mask is not None:
        assert tomo.shape == mask.shape, f"Tomogram shape ({tomo.shape}) and mask shape ({mask.shape}) need to be equal."

    if conf.zrange:
        hb = int((window_size - 1) // 2)
        minz = max(0, conf.zrange[0] - hb)
        maxz = min(conf.zrange[1] + hb, tomo.shape[0])
        conf.zrange = (
            minz,
            maxz,
        )  # here we need to take make sure that the box size is subtracted etc.

    boxer = SlidingWindowBoxer(
        box_size=window_size, stride=conf.stride, zrange=conf.zrange, mask=mask
    )
    embeddings = sliding_window_embedding(tomo=tomo, boxer=boxer, embedor=embedor)
    if embeddings is None:
        return

    # Write results to disk

    filename = (
            os.path.splitext(os.path.basename(conf.volumes_path))[0]
            + "_embeddings.temb"
    )
    print("Embeddings have shape:", embeddings.shape)
    column_names = ["Z", "Y", "X"]
    for i in range(embeddings.shape[1] - 3):
        column_names.append(str(i))
    df = pd.DataFrame(data=embeddings, columns=column_names)
    df.index.name = "index"
    df.attrs["tt_version_embed"] = tomotwin.__version__
    df.attrs["filepath"] = conf.volumes_path
    df.attrs["modelpth"] = conf.model_path
    df.attrs["modelmd5"] = get_file_md5(conf.model_path)
    df.attrs["window_size"] = window_size
    df.attrs["stride"] = conf.stride
    df.attrs["tomogram_input_shape"] = tomo.shape
    df.attrs["tomotwin_config"] = embedor.tomotwin_config
    if conf.zrange:
        df.attrs["zrange"] = conf.zrange

    for col in df:
        if col == "filepath":
            continue
        df[col] = df[col].astype(np.float16)
    df.to_pickle(os.path.join(conf.output_path, filename))

    print(f"Wrote embeddings to disk to {os.path.join(conf.output_path, filename)}")
    print("Done.")

def make_embeddor(conf: EmbedConfiguration, rank: int, world_size: int) -> Embedor:
    '''
    Create the embeddor
    :param conf: Embed configuratio from an UI
    :return: Instance of embeddor
    '''
    if rank is None:
        embedor = TorchEmbedor(
            weightspth=conf.model_path,
            batchsize=conf.batchsize,
            workers=4,
        )
    else:
        embedor = TorchEmbedorDistributed(
            weightspth=conf.model_path,
            batchsize=conf.batchsize,
            rank=rank,
            world_size=world_size,
            workers=4,
        )
    return embedor

def run(rank, conf: EmbedConfiguration, world_size) -> None:
    '''
    Runs the embed procedure
    :param conf: Configuration from a UI
    '''
    os.makedirs(conf.output_path, exist_ok=True)
    embedor = make_embeddor(conf, rank=rank, world_size=world_size)

    window_size = get_window_size(conf.model_path)
    if conf.mode == EmbedMode.TOMO:
        tomo = -1 * MrcFormat.read(conf.volumes_path)  # -1 to invert the contrast
        mask = None
        if conf.maskpth is not None:
            mask = MrcFormat.read(conf.maskpth)!=0
        embed_tomogram(tomo, embedor, conf, window_size, mask)
    elif conf.mode == EmbedMode.VOLUMES:
        paths = []
        for p in conf.volumes_path:
            if os.path.isfile(p) and p.endswith(".mrc"):
                paths.append(p)
            if os.path.isdir(p):
                foundfiles = glob.glob(os.path.join(p, "*.mrc"))
                paths.extend(foundfiles)
        embed_subvolumes(paths, embedor, conf)


def run_distr(config, world_size: int):
    """
    Starts a distributed run using DistributedDataParallel
    """
    mp.set_sharing_strategy('file_system')
    print(f"Found {world_size} GPU(s). Start DDP + Compiling.")
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29' + str(random.randint(1, 500)).zfill(3)
    mp.spawn(
        run,
        args=([config, world_size]),
        nprocs=world_size
    )


def start(config):
    '''
    Start the embedding procedure
    '''

    if config.distr_mode == DistrMode.DDP:
        world_size = torch.cuda.device_count()
        run_distr(config, world_size)
    else:
        run(None, config, None)

def _main_():
    ########################
    # Get configuration from user interface
    ########################
    ui = EmbedArgParseUI()
    ui.run()
    check_for_updates()
    config = ui.get_embed_configuration()
    start(config)



if __name__ == "__main__":
    _main_()
