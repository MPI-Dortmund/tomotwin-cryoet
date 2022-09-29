from tomotwin.modules.tools.tomotwintool import TomoTwinTool
import pandas as pd
import numpy as np
import argparse
import os

class MedianTool(TomoTwinTool):

    def get_command_name(self) -> str:
        return 'median_embedding'

    def create_parser(self, parentparser : argparse.ArgumentParser) -> argparse.ArgumentParser:
        '''
        :param parentparser: ArgumentPaser where the subparser for this tool needs to be added.
        :return: Argument parser that was added to the parentparser
        '''

        parser = parentparser.add_parser(
            self.get_command_name(),
            help="Calculates the median embedding of a embedding file. That seems to be useful to detect background region.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        parser.add_argument('-i', '--input', type=str, required=True,
                            help='Embeddings file')

        parser.add_argument('-o', '--output', type=str, required=True,
                            help='Output folder')

    @staticmethod
    def calculate_median_embedding(tomo_embeddings: pd.DataFrame) -> pd.DataFrame:
        tomo_embeddings_nopos = tomo_embeddings.drop(['Z', 'Y', 'X'], axis=1)
        med = tomo_embeddings_nopos.median(axis=0)

        med['filepath'] = "median"
        column_number = len(med) - 1

        med = pd.DataFrame(np.atleast_2d(med))
        med = med.rename(columns={column_number: "filepath"})

        return med

    def run(self, args):
        os.makedirs(args.output,exist_ok=True)
        tomo_references = pd.read_pickle(args.input)
        print("Calculate median embedding")
        median = self.calculate_median_embedding(tomo_references)
        print("Write to disk")
        filename = os.path.splitext(os.path.basename(args.input))[0]

        median.to_pickle(os.path.join(args.output,f"{filename}_med.temb"))
        print("Done")


