from glob import glob
import tempfile
import os
import subprocess
import shlex
import mrcfile
import sys
from typing import List
from dataclasses import dataclass
from multiprocessing import Pool
import numpy as np
from tqdm import tqdm
import argparse



# clip:
# e2proc3d.py molmaps/3pxi.mrc clipped/3pxi_clipped.mrc --clip 36
# e2proc3d.py clipped/1kb9_clipped.mrc 1kb9_moved.mrc --alignref clipped/3pxi_clipped.mrc --align rotate_translate_3d_tree
EMAN2_PTHS={}

@dataclass
class InitMap:

    initmap: np.ndarray
    column_pdbs: List[str]

    def _get_index(self, pdb: str):
        for p_i, p in enumerate(self.column_pdbs):
            if p.upper() == pdb.upper():
                return p_i
        return None

    def get_value(self, pdba: str, pdbb: str):
        index_a = self._get_index(pdba)
        index_b = self._get_index(pdbb)
        if index_a is None or index_b is None:
            return None

        return self.initmap[index_a,index_b]

@dataclass
class Pair:
    fixed: str
    move: str

def clip(inpth, outpth, size=36):
    if not os.path.exists(outpth):
        clip_cmd = f"{EMAN2_PTHS['python']} {EMAN2_PTHS['e2proc3d']} {inpth} {outpth} --clip {size}"
        args = shlex.split(clip_cmd)
        subprocess.run(args, check=True)

def ccc(align_vol,fixed_vol):
    align_vol_mask = align_vol>10**-6
    fixed_vol_mask = fixed_vol>10**-6
    align_vol[align_vol_mask] = (align_vol[align_vol_mask] - np.mean(align_vol[align_vol_mask]))
    fixed_vol[fixed_vol_mask] = (fixed_vol[fixed_vol_mask] - np.mean(fixed_vol[fixed_vol_mask]))

    a = np.sum(align_vol*fixed_vol)
    b = np.sum(align_vol*align_vol) * np.sum(fixed_vol*fixed_vol)
    return (a/np.sqrt(b))

def compare(pair: Pair, clipped_pth: str, initmap: InitMap) -> float:
    if initmap:
        pdba = os.path.splitext(os.path.basename(pair.move))[0]
        pdbb = os.path.splitext(os.path.basename(pair.fixed))[0]
        init_cross = initmap.get_value(pdba=pdba, pdbb=pdbb)
        if init_cross:
            if init_cross > 0.6:
                print(pdba, pdbb, init_cross)
            return init_cross

    move_clipped = os.path.join(clipped_pth, os.path.basename(pair.move))
    clip(pair.move, move_clipped)

    fixed_clipped = os.path.join(clipped_pth, os.path.basename(pair.fixed))
    clip(pair.fixed, fixed_clipped)


    with tempfile.TemporaryDirectory() as alignpth:
        alignfolder = os.path.join(alignpth, "aligned")
        os.makedirs(alignfolder, exist_ok=True)
        aligned_vol_pth = os.path.join(alignfolder, os.path.basename(pair.move))
        align_cmd = f"{EMAN2_PTHS['python']} {EMAN2_PTHS['e2proc3d']} {move_clipped} {aligned_vol_pth} --alignref {fixed_clipped} --align rotate_translate_3d_tree"
        args = shlex.split(align_cmd)
        subprocess.run(args, check=True)

        with mrcfile.open(aligned_vol_pth) as mrc:
            align_vol = np.copy(mrc.data)
        with mrcfile.open(fixed_clipped) as mrc:
            fixed_vol = np.copy(mrc.data)
        align_vol = align_vol.reshape(-1)
        fixed_vol = fixed_vol.reshape(-1)

        cross = ccc(align_vol,fixed_vol)
        if cross > 0.6:
            print(os.path.basename(aligned_vol_pth), os.path.basename(fixed_clipped), cross)

        return cross


def sim_matrix(molmaps, initmap: InitMap):
    sym = np.zeros(shape=(len(molmaps), len(molmaps)))
    global TOTAL
    with tempfile.TemporaryDirectory() as f:
        clipped_pth = os.path.join(f, "clipped")
        os.makedirs(clipped_pth, exist_ok=True)
        args = []
        args_indicis = []
        for i_fixed, pth_fixed in enumerate(molmaps):
            for i_move, pth_move in enumerate(molmaps[(i_fixed+1):], start=i_fixed+1):
                pair = Pair(fixed=pth_fixed,move=pth_move)
                args.append((pair,clipped_pth, initmap))
                args_indicis.append((i_fixed, i_move))

        with Pool() as pool:
            res = pool.starmap(compare, tqdm(args,total=len(args)), chunksize=1)

        for elem_index, index_tuble in enumerate(args_indicis):
            sym[index_tuble[0], index_tuble[1]] = res[elem_index]
            sym[index_tuble[1], index_tuble[0]] = res[elem_index]

        return sym

def gen_init_map(pth_matix: str, pth_elements: str = None) -> InitMap:

    if pth_elements == None:
        import pandas as pd
        df = pd.read_pickle(pth_matix)

        map = InitMap(
            initmap=df.to_numpy(),
            column_pdbs=df.attrs['elements']
        )

    else:
        initmap = np.load(pth_matix)

        with open(pth_elements) as f:
            lines = f.readlines()

        elements = []
        for l in lines:
            base = os.path.basename(l[:-1])
            pdbname = os.path.splitext(base)[0]
            elements.append(pdbname)
        map = InitMap(initmap=initmap, column_pdbs=elements)
    return map


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Convert clusters to coords')
    parser.add_argument('-i','--input', type=str, nargs='+', required=True,
                        help='Path to all molmaps to analyse.')
    parser.add_argument("--eman2", type=str, required=True,
                        help="eman2 environment path")
    parser.add_argument('--matrix', type=str,
                        help='Path to previous matrix')


    parser.add_argument('-o', '--output', type=str, required=True,
                        help='Output path for new similarity matrix')
    return parser

def _main_():
    global EMAN2_PTHS

    parser = get_parser()
    args = parser.parse_args()
    # Path to all molecular maps you want to compare
    pth_molmaps = args.input # "/mnt/data/twagner/Projects/TomoTwin/data/all_molmaps/"

    EMAN2_PTHS["env"]=args.eman2
    from pathlib import Path

    EMAN2_PTHS["python"] = str(list(Path(EMAN2_PTHS["env"]).rglob('bin/python'))[0])
    EMAN2_PTHS["e2proc3d"] = str(list(Path(EMAN2_PTHS["env"]).rglob('e2proc3d.py'))[0])

    print(EMAN2_PTHS)

    ### In case you want to start from a previous run you need to change this:
    #pth_init_matrix = args.initmatrix #"/home/twagner/Projects/TomoTwin/src/tomotwin/resources/sym_matrix_init.npy"
    #th_init_matrix_elements = args.initelements #"/home/twagner/Projects/TomoTwin/src/tomotwin/resources/sym_matrix_elements_init.txt"
    initmode = gen_init_map(args.matrix)
    molmaps = []
    for pth in pth_molmaps:
        glb = glob(pth+"**/*.mrc", recursive=True)
        molmaps.extend(glb)

    matrix = sim_matrix(molmaps, initmode)
    os.makedirs(args.output, exist_ok=True)
    import pandas as pd

    dat = pd.DataFrame(matrix)

    elements = []

    for item in molmaps:
        elements.append(os.path.splitext(os.path.basename(item))[0])

    dat.attrs['elements']=elements
    dat.to_pickle(os.path.join(args.output, "matrix.pkl"))

    np.save(os.path.join(args.output, "sym_matrix.npy"), matrix)

    with open(os.path.join(args.output, "sym_matrix_elements.txt"), 'w') as f:
        for item in elements:
            f.write("%s\n" % item)

if __name__ == '__main__':
    _main_()