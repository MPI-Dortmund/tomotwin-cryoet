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

import json
import os
from typing import List, Dict

import mrcfile
import numpy as np
import pandas as pd
import tqdm
from scipy.ndimage import zoom

import tomotwin
from tomotwin.modules.common.preprocess import label_filename
from tomotwin.modules.common.utils import check_for_updates
from tomotwin.modules.inference.argparse_locate_ui import LocateArgParseUI
from tomotwin.modules.inference.findmaxima_locator import FindMaximaLocator
from tomotwin.modules.inference.locate_ui import (
    LocateConfiguration,
)
from tomotwin.modules.inference.locator import Locator


def read_map(path: str) -> pd.DataFrame:
    """
    Read the results after mapping
    :param path: Path to map
    """
    if path.endswith(".txt"):
        df_map = pd.read_csv(path)
    elif path.endswith(".pkl") or path.endswith(".tmap"):
        df_map = pd.read_pickle(path)

    else:
        print("Format not implemented")
        return None
    if "filename" in df_map.columns:
        df_map.drop(columns=["filename"], inplace=True)
    return df_map


def run_non_maximum_suppression(class_frames: List[pd.DataFrame], boxsize: int, size_dict: Dict[str,int]=None) -> List[pd.DataFrame]:
    '''
    Runs nun maximum supression for each target class
    :param class_frames: Results for each target
    :param boxsize: Boxsize to use
    :param size_dict: Alternative to boxsize, a dictionary can be provided that maps the traget name to a boxsize.
    :return: list of nms applied dataframes
    '''
    for class_id, class_frame in enumerate(class_frames):
        if len(class_frame) == 0:
            print("No particles for class", class_frame.attrs["name"])
            continue
        class_name = class_frame.attrs["name"]
        before_nms = len(class_frame)
        if size_dict is not None:
            try:
                size = size_dict[class_name]
            except KeyError:
                print(
                    f"Can't find boxsize for {class_name}. Try to use extract PDB id and use this"
                )
                pdb = label_filename(class_name)
                try:
                    size = size_dict[pdb.lower()]
                except KeyError as kr:
                    raise KeyError(
                        f"Can't find size for {class_name} in boxsize dictionary"
                    ) from kr
        else:
            size = boxsize

        class_frame = Locator.nms(class_frame, size)
        class_frames[class_id] = class_frame
        print(
            f"Particles of class {class_name}: {len(class_frame)} (before NMS: {before_nms}) "
        )
    return class_frames


def scale_and_pad_heatmap(vol: np.array, stride: int, tomo_input_shape: tuple) -> np.array:
    '''
    Scales the calculated volume so that it fits the tomo_input_shape
    '''
    def get_pad_tuble(total_pad):
        if total_pad % 2 == 0:
            return (total_pad // 2, total_pad // 2)
        return (total_pad // 2 + 1, total_pad // 2)
    print('tomo_input_shape ', tomo_input_shape)
    print('vol before zoom ', vol.shape)
    vol = zoom(vol, stride)
    print('vol after zoom ', vol.shape)
    vol = vol.swapaxes(0, 2)
    print('vol after swap ', vol.shape)
    get_pad_tuble(np.abs(tomo_input_shape[0] - vol.shape[0]))
    print("pad tuble", get_pad_tuble(np.abs(tomo_input_shape[0] - vol.shape[0])), get_pad_tuble(np.abs(tomo_input_shape[1] - vol.shape[1])), get_pad_tuble(np.abs(tomo_input_shape[2] - vol.shape[2])) )
    vol = np.pad(
        vol, (
            get_pad_tuble(np.abs(tomo_input_shape[0] - vol.shape[0])),
            get_pad_tuble(np.abs(tomo_input_shape[1] - vol.shape[1])),
            get_pad_tuble(np.abs(tomo_input_shape[2] - vol.shape[2]))),
        "constant",
        constant_values=np.min(vol) - 0.01 * np.abs(np.min(vol)))
    print('vol after pad ', vol.shape)
    print('constant ', np.min(vol) - 0.01 * np.abs(np.min(vol)))
    return vol


def write_heatmaps(reference_names: List[str], out_path: str, heatmaps: List[np.array], stride: int,
                   tomo_input_shape: tuple) -> None:
    '''
    Write heatmaps to disk
    :param reference_names: Name of the references
    :param out_path: Folder where the heatmaps will be written to.
    :param heatmaps: List of heatmaps
    :return: None
    '''
    assert len(reference_names) == len(heatmaps), "Unequal number of references and heatmaps"

    for ref_i, ref_name in tqdm.tqdm(
            enumerate(reference_names), desc="Write heatmaps"
    ):
        with mrcfile.new(
                os.path.join(out_path, ref_name + ".mrc"), overwrite=True
        ) as mrc:
            print("Write heatmap", os.path.join(out_path, ref_name + ".mrc"))
            vol = heatmaps[ref_i]
            vol = vol.astype(np.float32)
            vol = scale_and_pad_heatmap(vol, stride, tomo_input_shape)
            mrc.set_data(vol)


def run(conf: LocateConfiguration) -> None:
    '''
    Runs the locate procedure
    :param conf: Configuration file from a UI
    '''
    out_path = conf.output_path
    os.makedirs(out_path, exist_ok=True)
    map_result = read_map(conf.map_path)

    if "stride" in map_result.attrs:
        stride = map_result.attrs["stride"]
    else:
        raise ValueError(
            "Stride unknown. It seems that you are using an invalid model"
        )
    if len(stride) == 1:
        stride = stride * 3

    if "window_size" in map_result.attrs:
        window_size = map_result.attrs["window_size"]
    else:
        raise ValueError("Window size unknown. Stop.")

    size_dict = None
    if conf.boxsize is None:
        conf.boxsize = window_size
    try:
        conf.boxsize = int(conf.boxsize)
    except ValueError:
        print("Read boxsize from JSON")
        with open(conf.boxsize, "r", encoding="utf8") as conf_sizes_file:
            size_dict = json.load(conf_sizes_file)

    locator = FindMaximaLocator(
        tolerance=conf.tolerance,
        stride=stride,
        window_size=window_size,
        global_min=conf.global_min,
        processes=conf.processes
    )
    locator.output = out_path
    map_attrs = map_result.attrs

    class_frames_and_vols = locator.locate(map_result)
    del map_result



    class_vols = [t.attrs['heatmap'] for t in class_frames_and_vols if 'heatmap' in t.attrs]
    class_frames_and_vols = run_non_maximum_suppression(class_frames_and_vols, conf.boxsize, size_dict=size_dict)

    located_particles = pd.concat(class_frames_and_vols)

    located_particles.attrs["tt_version_locate"] = tomotwin.__version__

    # Add meta information from previous step
    for meta_key in map_attrs:
        located_particles.attrs[meta_key] = map_attrs[meta_key]

    located_particles.to_pickle(os.path.join(out_path, "located.tloc"))

    # Write picking headmaps
    if conf.write_heatmaps:
        write_heatmaps(map_attrs["references"], out_path, class_vols, stride, map_attrs["tomogram_input_shape"])

def _main_():
    ui = LocateArgParseUI()
    ui.run()

    check_for_updates()

    conf = ui.get_locate_configuration()
    run(conf=conf)


if __name__ == "__main__":
    _main_()
