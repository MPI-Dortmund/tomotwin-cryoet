import argparse
from argparse import ArgumentParser
import os
import numpy as np
import pandas as pd
import mrcfile



class SegmentationMaskTool(TomoTwinTool):
    def get_command_name(self) -> str:
        return 'embedding mask'

    def create_parser(self, parentparser : ArgumentParser) -> ArgumentParser:
        '''
        :param parentparser: ArgumentPaser where the subparser for this tool needs to be added.
        :return: Argument parser that was added to the parentparser
        '''

        parser = parentparser.add_parser(
            self.get_command_name(),
            help="Calculates a umap for the lasso  tool",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument('-i', '--input', type=str, required=True,
                            help='Embeddings file to use for clustering')

        parser.add_argument('-o', '--output', type=str, required=True,
                            help='Output folder')

        return parser

    def create_segmentation_map(embeddings):
        print("Create segmentation map")
        Z = embeddings.attrs['tomogram_input_shape'][0]
        Y = embeddings.attrs['tomogram_input_shape'][1]
        X = embeddings.attrs['tomogram_input_shape'][2]
        embeddings = embeddings.reset_index(drop=True)
        segmentation_mask = embeddings[['Z', 'Y', 'X']].copy()
        segmentation_mask = segmentation_mask.reset_index()
        empty_array = np.zeros(shape=(Z, Y, X))
        for row in segmentation_mask.itertuples(index=True, name='Pandas'):
            X = int(row.X)
            Y = int(row.Y)
            Z = int(row.Z)
            label = int(row.index)
            empty_array[(Z):(Z + 2), (Y):(Y + 2), (X):(X + 2)] = label + 1
        segmentation_array = empty_array.astype(np.float32)

        return segmentation_array

    def run(self, args):
        print("Read data")
        embeddings = pd.read_pickle(args.input)
        print("Generate segmentation mask")
        segmentation_layer = create_segmentation_map(embeddings=embeddings)
        print("Write results to disk")
        os.makedirs(args.output, exist_ok=True)
        with mrcfile.new(
                os.path.join(args.output, os.path.splitext(os.path.basename(args.input))[0] + "_embedding_mask.mrci"),
                overwrite=True) as mrc:
            mrc.set_data(segmentation_layer)
        print("Done")






