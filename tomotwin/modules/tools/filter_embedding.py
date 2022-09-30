from tomotwin.modules.tools.tomotwintool import TomoTwinTool
import pandas as pd
import numpy as np
import argparse
import os
from tomotwin.modules.tools.median_embedding import MedianTool
class FilterTool(TomoTwinTool):


    def get_command_name(self) -> str:
        return 'filter_embedding'


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

        parser.add_argument('-m', '--map', type=str, required=True,
                            help='Map file for median embedding')

        parser.add_argument('-t', '--threshold', type=str, required=True,
                            help='All embeddings higher than this similarty threshold are discarded')

        parser.add_argument('-o', '--output', type=str, required=True,
                            help='Output folder')

    from typing import List
    @staticmethod
    def filter_embeddings(embeddings: pd.DataFrame, map_results: pd.DataFrame, threshold: float) -> List[pd.DataFrame]:
        """
        Removes all embeddings with a similarity to the median embedding higher than the threshold. Runs filtering for
        each reference in map_results seperately
        """
        filtered_embeddings = []
        map_results.df.drop(['X', 'Y', 'Z'], axis=1)
        for reference in map_results:
            mask = map_results[reference] > threshold
            mask = mask.to_numpy()
            filtered_embeddings.append(embeddings.iloc[mask])

        return filtered_embeddings

    def run(self, args):
        tomo_embeddings = pd.read_pickle(args.input)
        tomo_map = pd.read_pickle(args.map)
        os.makedirs(args.output, exist_ok=True)

        filtered = self.filter_embeddings(embeddings=tomo_embeddings,
                                          map_results=tomo_map,
                                          threshold=args.threshold)
        embedding_filename = os.path.splitext(os.path.basename(args.input))[0]
        for emb_index, emb in enumerate(filtered):
            ref_name = os.path.splitext(os.path.basename(tomo_map.attrs['references'][emb_index]))[0]
            out=os.path.join(args.output,f"{embedding_filename}_filtered_{ref_name}.temb")
            emb.to_pickle(out)
            print(f"Wrote {out}")
