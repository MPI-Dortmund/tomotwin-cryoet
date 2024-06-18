import argparse
import os

import pandas as pd
from tomotwin.modules.common.io.mrc_format import MrcFormat


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Estimate the box size of molmaps produced by EMAN2')
    parser.add_argument('-e', '--embedding', type=str, required=True,
                        help='Path to embedding file.')
    parser.add_argument('-c', '--coords', type=str, required=True,
                        help='Path to coords file.')
    parser.add_argument('-t', '--tomo', type=str, required=True,
                        help='Path to tomogram file.')
    parser.add_argument('-s', '--stride', type=str, default=2, help='Stride')
    parser.add_argument('-o', '--out', type=str, required=True,
                        help='Output path')
    return parser


def _main_():
    parser = get_parser()
    args = parser.parse_args()

    os.makedirs(args.out)
    coords = pd.read_csv(args.coords,
                         delim_whitespace=True,
                         header=None,
                         index_col=False,
                         dtype=float,
                         )
    coords.columns = ['X', 'Y', 'Z']

    embeddings = pd.read_pickle(args.embedding)
    embeddings = pd.concat([coords[['X', 'Y', 'Z']], embeddings], axis=1)
    embeddings.attrs["stride"] = (args.stride, args.stride, args.stride)
    embeddings.attrs["tomogram_input_shape"] = MrcFormat.read(args.tomo).shape
    fname = os.path.splitext(os.path.basename(args.embedding))[0]
    embeddings.to_pickle(os.path.join(args.out, fname + "_mod.temb"))


if __name__ == '__main__':
    _main_()
