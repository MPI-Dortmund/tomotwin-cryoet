import argparse
import os

import numpy as np
import pandas as pd

from tomotwin.modules.tools.tomotwintool import TomoTwinTool


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
            help="Filters the embedding file for relevant embeddings. Handy for median denoising",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        parser.add_argument('-i', '--input', type=str, required=True,
                            help='Embeddings file')

        parser.add_argument('-m', '--map', type=str, required=True,
                            help='tmap file')

        parser.add_argument('-t', '--threshold', type=float, required=True,
                            help='All embeddings higher (use --lower otherwise) than this similarty threshold are discarded')

        parser.add_argument('-o', '--output', type=str, required=True,
                            help='Output folder')

        parser.add_argument('--lower',
                            action='store_true',
                            help="Discard embeddings lower than given threshold")

        parser.add_argument('--concat',
                            action='store_true',
                            help="Concatenate all filtered embeddings into one embedding file")

    from typing import List
    @staticmethod
    def filter_embeddings(embeddings: pd.DataFrame, map_results: pd.DataFrame, threshold: float, discard_lower=False,
                          concat=False) -> List[pd.DataFrame]:
        """
        Removes all embeddings with a similarity to the median embedding higher than the threshold. Runs filtering for
        each reference in map_results seperately
        """

        def apply_thresh(tmap: pd.DataFrame, thresh: float, discard_lower: bool):
            if discard_lower:
                return tmap >= thresh
            return tmap < thresh


        filtered_embeddings = []
        map_results_no_coords = map_results.drop(['X', 'Y', 'Z'], axis=1)
        masks = []
        for ref_index, reference in enumerate(map_results_no_coords):
            print(f"Filter reference {map_results_no_coords.attrs['references'][ref_index]} (Map column: {reference})")

            mask = apply_thresh(map_results_no_coords[reference], threshold, discard_lower)
            mask = mask.to_numpy()

            if not concat:
                filtered_embeddings.append(embeddings.iloc[mask])
            else:
                masks.append(mask)

        if concat:
            m = masks[0]
            for i in range(1, len(masks)):
                m = np.logical_or(m, masks[i])
            filtered_embeddings.append(embeddings.iloc[m])

        return filtered_embeddings

    def run(self, args):
        tomo_embeddings = pd.read_pickle(args.input)
        tomo_map = pd.read_pickle(args.map)
        os.makedirs(args.output, exist_ok=True)

        filtered = self.filter_embeddings(embeddings=tomo_embeddings,
                                          map_results=tomo_map,
                                          threshold=args.threshold,
                                          discard_lower=args.lower,
                                          concat=args.concat)
        embedding_filename = os.path.splitext(os.path.basename(args.input))[0]
        if not args.concat:
            for emb_index, emb in enumerate(filtered):
                ref_name = os.path.splitext(os.path.basename(tomo_map.attrs['references'][emb_index]))[0]
                out = os.path.join(args.output, f"{embedding_filename}_filtered_{ref_name}.temb")
                emb.to_pickle(out)
                print(f"Wrote {out} - removed {100 - len(emb) / len(tomo_embeddings) * 100:.2f}% embedding points")
        else:
            emb = filtered[0]
            out = os.path.join(args.output, f"{embedding_filename}_filtered_allrefs.temb")
            emb.to_pickle(out)
            print(f"Wrote {out} - removed {100 - len(emb) / len(tomo_embeddings) * 100:.2f}% embedding points")
