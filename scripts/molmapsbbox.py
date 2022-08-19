from skimage.measure import label, regionprops
import numpy as np
from typing import List
import mrcfile
from tomotwin.modules.common.preprocess import label_filename
import argparse
import os
import json

def binarize(vol):
    t = np.max(vol) - (np.max(vol) - np.min(vol)) * 0.9
    return vol > t

def boxsize(vol):
    lbl = label(vol, background=0)
    regs = regionprops(lbl)
    size = -1
    for reg in regs:
        cand_size = np.max([np.abs(reg.bbox[i] - reg.bbox[i+3]) for i in range(int(len(reg.bbox)/2))])
        if cand_size>size:
            size = cand_size
    return size

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Estimate the box size of molmaps produced by EMAN2')
    parser.add_argument('-i','--input', type=str, nargs='+', required=True,
                        help='Path to all molmaps to analyse.')
    parser.add_argument('-o', '--out', type=str,
                        help='Output path')
    return parser

def _main_():
    parser = get_parser()
    args = parser.parse_args()
    pth : List[str] = args.input
    out_pth: str = args.out
    max_size = 37
    size_list = {}
    for p in pth:
        a = mrcfile.read(p)
        b = binarize(a)
        size = int(np.round(min(boxsize(b)*1.2, max_size)))

        pdb_lbl = label_filename(p)

        print(f"Used size of {size} for protein {pdb_lbl}")


        size_list[pdb_lbl] = size
    size_list["VESICLE"] = None
    size_list["FIDUCIAL"] = 18

    os.makedirs(out_pth,exist_ok=True)

    with open(os.path.join(out_pth,"boxsizes.json"), "w") as file:
        json.dump(size_list, file, indent=4, sort_keys=True)

if __name__ == '__main__':
    _main_()