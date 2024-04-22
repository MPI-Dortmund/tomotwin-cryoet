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
from argparse import ArgumentParser

import pandas as pd

from tomotwin.modules.tools.tomotwintool import TomoTwinTool


class ScaleCoordinates(TomoTwinTool):
    def get_command_name(self) -> str:
        '''
        :return: Command name
        '''
        return "scale_coordinates"

    def create_parser(self, parentparser : ArgumentParser) -> ArgumentParser:
        '''
        :param parentparser: ArgumentPaser where the subparser for this tool needs to be added.
        :return: Argument parser that was added to the parentparser
        '''

        parser = parentparser.add_parser(
            self.get_command_name(),
            help="Scales coordinates.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument('--coords', type=str, required=True,
                            help='Coords file to scale')
        parser.add_argument('--tomotwin_pixel_size', type=float, required=True,
                            help='Pixel size of tomograms picked with TomoTwin')
        parser.add_argument('--out', type=str, required=False, default=None,
                            help='output filename')
        parser.add_argument('--extraction_pixel_size', type=float, required=True,
                            help='Pixel size of tomograms to use for extraction')
        return parser

    @staticmethod
    def scale_coords(coords_pth: str, px1: float, px2: float) -> pd.DataFrame:
        coords_df = pd.read_csv(
            coords_pth,
            delim_whitespace=True,
            header=None,
            index_col=False,
            dtype=float,
        )

        coords_df.columns = ['X', 'Y', 'Z']
        scaling_factor = px1 / px2
        print('Scaling coordinates by ' + str(scaling_factor) + 'x')
        coords_df = coords_df * scaling_factor
        return coords_df

    def run(self, args):
        coords_pth = args.coords
        px1 = args.tomotwin_pixel_size
        px2 = args.extraction_pixel_size

        out_name = args.out
        if os.path.splitext(out_name)[1] == '':
            os.makedirs(args.out, exist_ok=True)
        if os.path.isdir(out_name):
            out_name = os.path.join(out_name, os.path.splitext(os.path.basename(coords_pth))[0])


        scaled_coords = ScaleCoordinates.scale_coords(coords_pth=coords_pth, px1=px1, px2=px2)

        scaled_coords.to_csv(out_name, index=False, header=False, sep=' ')



