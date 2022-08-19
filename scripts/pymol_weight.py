from pymol import cmd
from pymol import util
import os
import json
import argparse
from glob import glob
import os
import numpy as np
import re


def get_sorted_pdbs_and_sizes(pdb_paths):
    sizes = []
    pdb_ids = []
    for pdb in pdb_paths:
        pdb_id = os.path.splitext(os.path.basename(pdb))[0]
        cmd.load(pdb, pdb_id)
        sizes.append(int(util.compute_mass(pdb_id) / 1000))
        pdb_ids.append(pdb_id)
    sizes = np.array(sizes)
    pdb_ids = np.array(pdb_ids)
    sortsizes = np.argsort(sizes)[::-1]

    return (pdb_ids[sortsizes].tolist(), sizes[sortsizes].tolist())



def _main_():
    parser = argparse.ArgumentParser(
        description="Estimate the weight of pdbs using PyMOL")
    parser.add_argument('--pdb', nargs='+', type=str)
    parser.add_argument('--out', type=str)
    args = parser.parse_args()

    pdbs = args.pdb

    pdbids,size = get_sorted_pdbs_and_sizes(pdbs)

    result_dict = {}
    for index, id in enumerate(pdbids):
        result_dict[id] = size[index]

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "pdb_weight.json"), "w") as file:
        json.dump(result_dict, file, indent=4)
    print(json.dumps(result_dict, indent=4))

if __name__ == '__main__':
    _main_()