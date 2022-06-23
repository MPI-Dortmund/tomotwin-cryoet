import mrcfile
import warnings
import numpy as np
import pandas as pd
import tqdm
import glob
import os
from pathlib import Path
from argparse import ArgumentParser

import argparse
from tomotwin.modules.tools.tomotwintool import TomoTwinTool

class ExtractReference(TomoTwinTool):
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
    def extract_and_save(volume: np.array, positions: pd.DataFrame, box_size: int, out_pth: str, basename: str):
        for index, row in tqdm.tqdm(positions.iterrows()):
            x = row['X']
            y = row['Y']
            z = row['Z']

            # Define corners of box
            nx1 = (x - (box_size - 1) // 2)
            nx2 = (x + (box_size - 1) // 2 + 1)
            ny1 = (y - (box_size - 1) // 2)
            ny2 = (y + (box_size - 1) // 2 + 1)
            nz1 = (z - (box_size - 1) // 2)
            nz2 = (z + (box_size - 1) // 2 + 1)


            coords = volume[int(nz1): int(nz2), int(ny1): int(ny2), int(nx1): int(nx2)]
            if coords.shape != (box_size, box_size, box_size):
                continue
            coords = -1 * coords  # invert

            with mrcfile.new(os.path.join(out_pth,basename+".mrc")) as newmrc:
                newmrc.set_data(coords)

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
        df = pd.read_csv(path_ref, sep='    ', header=None)
        df.columns = ['X', 'Y', 'Z']
        mrc = mrcfile.mmap(path_tomo, permissive=True, mode='r')

        ExtractReference.extract_and_save(mrc.data, df, boxsize, path_output, filebasename)
        print(f'wrote subvolume reference to {path_output}')


