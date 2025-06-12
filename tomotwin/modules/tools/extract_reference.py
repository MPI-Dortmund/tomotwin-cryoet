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
import argparse
import os
import warnings
from argparse import ArgumentParser
from typing import List

import mrcfile
import numpy as np
import pandas as pd
import tqdm

from tomotwin.modules.tools.tomotwintool import TomoTwinTool


class ExtractReference(TomoTwinTool):
    '''
    Extracts a subvolume (reference) from a volume and save it to disk.
    '''

    def get_command_name(self) -> str:
        '''
        :return: Command name
        '''
        return "extractref"

    def create_parser(self, parentparser : ArgumentParser) -> ArgumentParser:
        '''
        :param parentparser: ArgumentPaser where the subparser for this tool needs to be added.
        :return: Argument parser that was added to the parentparser
        '''

        parser = parentparser.add_parser(
            self.get_command_name(),
            help="Extracts reference subvolumes.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        parser.add_argument('--tomo', type=str, required=True,
                            help='Tomogram to extract from')
        parser.add_argument('--coords', type=str, required=True,
                            help='Coordinates of particle to extract (X, Y, Z)')
        parser.add_argument('--size', type=int, default=37,
                            help='Box size. For TomoTwin this should be typically a size of 37.')
        parser.add_argument('--out', type=str, required=True,
                            help='output path')
        parser.add_argument('--filename', type=str, default="reference",
                            help='filename of the reference')

        return parser

    @staticmethod
    def extract_and_save(volume: np.array, positions: pd.DataFrame, box_size: int, out_pth: str, basename: str, apix=None) -> List[str]:
        '''

        :param volume: Volume from which the the references should be extracted
        :param positions: Dataframe with coordinates
        :param box_size: Extraction boxsize in pixel
        :param out_pth: Folder where the subvolumes are written
        :param basename: Basename for files, index and filename are added automatically.
        :return: List with paths of written subvolumes.
        '''

        files_written = []
        print(positions.shape)
        for index, row in tqdm.tqdm(positions.iterrows()):
            print("iter")
            x = row['X']
            y = row['Y']
            z = row['Z']
            odd_factor = box_size % 2
            # Define corners of box
            nx1 = (x - (box_size - odd_factor) // 2)
            nx2 = (x + (box_size - odd_factor) // 2 + odd_factor)
            ny1 = (y - (box_size - odd_factor) // 2)
            ny2 = (y + (box_size - odd_factor) // 2 + odd_factor)
            nz1 = (z - (box_size - odd_factor) // 2)
            nz2 = (z + (box_size - odd_factor) // 2 + odd_factor)

            print(nx1, nx2, ny1, ny2, nz1, nz2)
            subvol = volume[int(nz1): int(nz2), int(ny1): int(ny2), int(nx1): int(nx2)]
            print(subvol.shape, box_size)
            if subvol.shape != (box_size, box_size, box_size):
                continue
            subvol = -1 * subvol  # invert
            subvol = subvol.astype(np.float32)
            fname = os.path.join(out_pth,f"{basename}_{index}.mrc")
            print(f"Writing {fname}")
            with mrcfile.new(fname) as newmrc:
                newmrc.set_data(subvol)
                if apix:
                    newmrc.voxel_size = apix
            files_written.append(fname)
        return files_written

    def run(self, args):
        path_tomo = args.tomo
        path_ref = args.coords
        path_output = args.out
        boxsize = args.size
        filebasename = os.path.splitext(args.filename)[0]




        #mute warnings when opening tomograms
        warnings.simplefilter('ignore')

        #Args to give cmd line interface
        os.makedirs(path_output,exist_ok=True)
        # Extract X Y Z coords from correct csv file
        #coords = pd.read_csv(path_ref, sep='    ', header=None)
        try:
            coords = pd.read_csv(path_ref,
                                 delim_whitespace=True,
                                 header=None,
                                 index_col=False,
                                 dtype=float,
                                 )
        except:
            print("Error while reading. Try to skip first row")
            coords = pd.read_csv(path_ref,
                                 delim_whitespace=True,
                                 header=None,
                                 index_col=False,
                                 dtype=float,
                                 skiprows=1
                                 )
            

        coords.columns = ['X', 'Y', 'Z']
        mrc = mrcfile.mmap(path_tomo, permissive=True, mode='r')

        ExtractReference.extract_and_save(mrc.data, coords, boxsize, path_output, filebasename, apix=mrc.voxel_size)
        print(f'wrote subvolume reference to {path_output}')
