"""
MIT License

Copyright (c) 2021 MPI-Dortmund

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from tomotwin.modules.inference.argparse_embed_ui import EmbedArgParseUI, EmbedMode
from tomotwin.modules.inference.embedor import TorchEmbedor, Embedor
from tomotwin.modules.inference.boxer import Boxer, SlidingWindowBoxer
from tomotwin.modules.inference.volumedata import FileNameVolumeDataset
import tomotwin.modules.common.io as io
from typing import List, Dict
import numpy as np
import os
import glob
import pandas as pd



def sliding_window_embedding(
    tomo: np.array, boxer: Boxer, embedor: Embedor
) -> np.array:
    boxes = boxer.box(tomogram=tomo)
    embeddings = embedor.embed(volume_data=boxes)
    positions = []
    for i in range(embeddings.shape[0]):
        positions.append(boxes.get_localization(i))
    positions = np.array(positions)
    embeddings = np.hstack([positions, embeddings])

    return embeddings


def volume_embedding(volume_pths: List[str], embedor: Embedor):
    volume_dataset = FileNameVolumeDataset(volumes=volume_pths, filereader=io.read_mrc)
    embeddings = embedor.embed(volume_dataset)

    return embeddings

def _main_():
    ########################
    # Get configuration from user interface
    ########################

    ui = EmbedArgParseUI()

    ui.run()
    conf = ui.get_embed_configuration()
    os.makedirs(conf.output_path, exist_ok=True)

    embeddings = None
    embedor = TorchEmbedor(
        weightspth=conf.model_path,
        batchsize=conf.batchsize,
        workers=12,#multiprocessing.cpu_count(),
    )


    if conf.mode == EmbedMode.TOMO:
        tomo = -1*io.read_mrc(conf.volumes_path)
        boxer = SlidingWindowBoxer(
            box_size=conf.window_size,
            stride=conf.stride,
        )
        embeddings = sliding_window_embedding(tomo=tomo, boxer=boxer, embedor=embedor)

        # Write results to disk

        filename = (
            os.path.splitext(os.path.basename(conf.volumes_path))[0] + "_embeddings.pkl"
        )
        print("Embeddings have shape:", embeddings.shape)
        column_names = ["Z", "Y", "X"]
        for i in range(embeddings.shape[1]-3):
            column_names.append(str(i))
        df = pd.DataFrame(data=embeddings, columns=column_names)
        df.insert(0, "filepath", [conf.volumes_path] * len(embeddings))
        df.index.name = "index"
        df.attrs["window_size"] = conf.window_size
        df.attrs["stride"] = conf.stride
        df.attrs["tomotwin_config"] = embedor.tomotwin_config

        df.to_pickle(os.path.join(conf.output_path, filename))

        print(f"Wrote embeddings to disk to {os.path.join(conf.output_path,filename)}")
        print("Done.")

    elif conf.mode == EmbedMode.VOLUMES:

        paths = []
        for p in conf.volumes_path:
            if os.path.isfile(p) and p.endswith(".mrc"):
                paths.append(p)
            if os.path.isdir(p):
                foundfiles = glob.glob(os.path.join(p, "*.mrc"))
                paths.extend(foundfiles)
        embeddings = volume_embedding(paths, embedor=embedor)
        column_names = []
        for i in range(embeddings.shape[1]):
            column_names.append(str(i))
        df = pd.DataFrame(data=embeddings, columns=column_names)
        df.insert(0, "filepath", paths)
        df.index.name = "index"
        df.to_pickle(os.path.join(conf.output_path, "embeddings.pkl"))
        print("Done")


if __name__ == "__main__":
    _main_()
