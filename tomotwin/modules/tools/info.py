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

    def print_pickle(self, dat):
        print("###########")
        print("DATA:")
        print("###########")
        print(dat)
        print("")

        print("###########")
        print("STATS:")
        print("###########")
        if 'predicted_class' in dat:
            for cl in range(len(dat.attrs['references'])):
                print(
                    f"Picked particles for class {cl} ({dat.attrs['references'][cl]}): {np.sum(dat['predicted_class'] == cl)}")
        else:
            print("-")

        print("")
        print("###########")
        print("ATTRIBUTES:")
        print("###########")
        import json
        print(json.dumps(dat.attrs, sort_keys=False, indent=3))

    def run(self, args):
        path_file = args.input
        try_torch = False
        try:
            dat = pd.read_pickle(path_file)
            self.print_pickle(dat)
        except:
            print("Is not a pickled tomotwin file. Try torch model.")
            try_torch = True

        if try_torch:
            import torch
            checkpoint = torch.load(path_file)
            print("#######################")
            print("Torch Model Info:")
            print("#######################")
            print("Keys:", checkpoint.keys())
            print("Best loss:", checkpoint['best_loss'])
            print("Best F1:", checkpoint['best_f1'])
            print("Epoch:", checkpoint['epoch'])
            if 'tt_version_train' in checkpoint:
                print("TomoTwin Version:", checkpoint['tt_version_train'])
            print("")
            print("#######################")
            print("Saved TomoTwin config:")
            print("#######################")
            self.tomotwin_config = checkpoint["tomotwin_config"]
            import json
            print(json.dumps(self.tomotwin_config, sort_keys=False, indent=3))