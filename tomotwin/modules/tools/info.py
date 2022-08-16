import argparse
from argparse import ArgumentParser
from tomotwin.modules.tools.tomotwintool import TomoTwinTool
import pandas as pd
import numpy as np
class Info(TomoTwinTool):

    def get_command_name(self) -> str:
        '''
        :return: Command name
        '''
        return "info"

    def create_parser(self, parentparser : ArgumentParser) -> ArgumentParser:
        '''
        :param parentparser: ArgumentPaser where the subparser for this tool needs to be added.
        :return: Argument parser that was added to the parentparser
        '''

        parser = parentparser.add_parser(
            self.get_command_name(),
            help="Prints info about pickled tomotwin files",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        parser.add_argument('-i', '--input', type=str, required=True,
                            help='Tomogram to extract from')


        return parser

    def run(self, args):
        path_file = args.input

        dat = pd.read_pickle(path_file)

        print("###########")
        print("DATA:")
        print("###########")
        print(dat)
        print("")

        print("###########")
        print("STATS:")
        print("###########")

        for cl in range(len(dat.attrs['references'])):
            print(f"Picked particles for class {cl} ({dat.attrs['references'][cl]}): {np.sum(dat['predicted_class']==cl)}")
        print("")
        print("###########")
        print("ATTRIBUTES:")
        print("###########")
        import json
        print(json.dumps(dat.attrs, sort_keys=False, indent=3))