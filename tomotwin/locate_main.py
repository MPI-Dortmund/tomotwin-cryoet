
from typing import List
import os
import sys

import numpy as np
import pandas as pd

from tomotwin.modules.inference.locate_ui import LocateUI, LocateMode
from tomotwin.modules.inference.argparse_locate_ui import LocateArgParseUI
from tomotwin.modules.inference.findmaxima_locator import FindMaximaLocator
from tomotwin.modules.inference.locator import Locator
import tqdm
from pyStarDB import sp_pystardb as star


def readprobs(path):
    if path.endswith(".txt"):
        return pd.read_csv(path)
    elif path.endswith(".pkl"):
        return pd.read_pickle(path)
    else:
        print("Format not implemented")

def run(ui: LocateUI):
    ui.run()
    conf = ui.get_locate_configuration()
    out_path = conf.output_path
    os.makedirs(out_path, exist_ok=True)
    probabilities = readprobs(conf.probability_path)

    if conf.mode == LocateMode.FINDMAX:
        if "stride" in probabilities.attrs:
            stride = probabilities.attrs["stride"]
        else:
            raise ValueError("Stride unknown. It seems that you are using an invalid model")
        if len(stride) == 1:
            stride = stride * 3

        if "window_size" in probabilities.attrs:
            window_size = probabilities.attrs["window_size"]
        else:
            raise ValueError("Window size unknown. Stop.")

        locator = FindMaximaLocator(tolerance=conf.tolerance, stride=stride, window_size=window_size, global_min=0.5)
        locator.output = out_path

    class_frames = locator.locate(classify_output=probabilities)
    size_dict=None

    if conf.boxsize is None:
        conf.boxsize = window_size
    elif not conf.boxsize.isnumeric():
        import json

        with open(conf.boxsize, "r") as conf_sizes_file:
            size_dict = json.load(conf_sizes_file)
    else:
        conf.boxsize = int(conf.boxsize)

    for class_id, class_frame in enumerate(class_frames):
        if len(class_frame)==0:
            print("No particles for class", class_frame.attrs['name'])
            continue
        class_name = np.unique(class_frame["predicted_class_name"])
        if len(class_name)>1:
            print("More than one class name predicted? Quit")
            sys.exit(1)
        class_name=class_name[0]
        before_nms = len(class_frame)
        if size_dict is not None:
            size = size_dict[class_name]
        else:
            size = conf.boxsize

        class_frame = Locator.nms(class_frame, size)
        class_frames[class_id] = class_frame
        print(f"Particles of class {class_name}: {len(class_frame)} (before NMS: {before_nms}) ")

    located_particles = pd.concat(class_frames)

    # Add meta information from previous step
    for meta_key in probabilities.attrs:
        located_particles.attrs[meta_key] = probabilities.attrs[meta_key]

    located_particles.to_pickle(os.path.join(out_path, f"located.tloc"))

def _main_():
    ui = LocateArgParseUI()
    run(ui=ui)



if __name__ == "__main__":
    _main_()