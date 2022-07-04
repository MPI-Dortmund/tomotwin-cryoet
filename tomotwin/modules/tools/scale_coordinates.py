import pandas as pd
from argparse import ArgumentParser
import argparse
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
        parser.add_argument('--out', type=str, required=True,
                            help='output filename')
        parser.add_argument('--extraction_pixel_size', type=float, required=True,
                            help='Pixel size of tomograms to use for extraction')
        return parser

    @staticmethod
    def scale_coords(coords_pth: str, px1: float, px2: float) -> pd.DataFrame:

        coords_df = pd.read_csv(coords_pth, sep=' ', header=None)
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

        scaled_coords = ScaleCoordinates.scale_coords(coords_pth=coords_pth, px1=px1, px2=px2)

        scaled_coords.to_csv(out_name, index=False, header=False, sep=' ')



