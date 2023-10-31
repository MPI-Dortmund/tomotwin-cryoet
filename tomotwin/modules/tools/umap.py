import argparse
import os
import pickle
import typing
from argparse import ArgumentParser

import cuml
import mrcfile
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from tqdm import tqdm

from tomotwin.modules.tools.tomotwintool import TomoTwinTool


class UmapTool(TomoTwinTool):

    def get_command_name(self) -> str:
        return 'umap'

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
                            help='Embeddings file')

        parser.add_argument('-o', '--output', type=str, required=True,
                            help='Output folder')
        parser.add_argument('-m', '--model', type=str, required=False, default=None,
                            help='Previously fitted model')
        parser.add_argument('-n', '--ncomponents', type=int, required=False, default=2,
                            help='Number of components')
        parser.add_argument('--neighbors', type=int, required=False, default=200,
                            help='Previously fitted model')
        parser.add_argument('--fit_sample_size', type=int, default=400000,
                            help='Sample size using for the fit of the umap')

        parser.add_argument('--chunk_size', type=int, default=400000,
                            help='Chunk size for transform all data')

        parser.add_argument('--cosine',
                            action='store_true',
                            help="Use cosine metric for estimating metric",
                            default=False)

        return parser

    def calcuate_umap(
            self, embeddings : pd.DataFrame,
            fit_sample_size: int,
            transform_chunk_size: int,
            reducer: cuml.UMAP = None,
            ncomponents=2,
            neighbors: int = 200,
            metric: str = "euclidean") -> typing.Tuple[ArrayLike, cuml.UMAP]:
        print("Prepare data")

        fit_sample = embeddings.sample(n=min(len(embeddings),fit_sample_size), random_state=17)
        fit_sample = fit_sample.drop(['filepath', 'Z', 'Y', 'X'], axis=1, errors='ignore')
        all_data = embeddings.drop(['filepath', 'Z', 'Y', 'X'],axis=1, errors='ignore')
        if reducer is None:
            reducer = cuml.UMAP(
                n_neighbors=neighbors,
                n_components=ncomponents,
                n_epochs=None,  # means automatic selection
                min_dist=0.0,
                random_state=19,
                metric=metric
            )
            print(f"Fit umap on {len(fit_sample)} samples")
            reducer.fit(fit_sample)
        else:
            print("Use provided model. Don't fit.")

        num_chunks = max(1, int(len(all_data) / transform_chunk_size))
        print(f"Transform complete dataset in {num_chunks} chunks with a chunksize of ~{int(len(all_data)/num_chunks)}")

        chunk_embeddings = []
        for chunk in tqdm(np.array_split(all_data, num_chunks),desc="Transform"):
            embedding = reducer.transform(chunk)
            chunk_embeddings.append(embedding)

        embedding = np.concatenate(chunk_embeddings)

        return embedding, reducer

    def create_embedding_mask(self, embeddings: pd.DataFrame):
        """
        Creates mask where each individual subvolume of the running windows gets an individual ID
        """
        print("Create embedding mask")
        Z = embeddings.attrs["tomogram_input_shape"][0]
        Y = embeddings.attrs["tomogram_input_shape"][1]
        X = embeddings.attrs["tomogram_input_shape"][2]
        stride = embeddings.attrs["stride"][0]
        segmentation_array = np.zeros(shape=(Z, Y, X), dtype=np.float32)
        z = np.array(embeddings["Z"], dtype=int)
        y = np.array(embeddings["Y"], dtype=int)
        x = np.array(embeddings["X"], dtype=int)

        values = np.array(range(1, len(x) + 1))
        for stride_x in tqdm(list(range(stride))):
            for stride_y in range(stride):
                for stride_z in range(stride):
                    index = (z + stride_z, y + stride_y, x + stride_x)
                    segmentation_array[index] = values

        return segmentation_array

    def run(self, args):
        print("Read data")
        embeddings = pd.read_pickle(args.input)
        out_pth = args.output
        model = None
        if args.model:
            model = pickle.load(open(args.model, "rb"))
        metric = "euclidean"
        if args.cosine:
            metric = "cosine"
        umap_embeddings, fitted_umap = self.calcuate_umap(embeddings=embeddings,
                                                          fit_sample_size=args.fit_sample_size,
                                                          transform_chunk_size=args.chunk_size,
                                                          reducer=model,
                                                          neighbors=args.neighbors,
                                                          ncomponents=args.ncomponents,
                                                          metric=metric)



        os.makedirs(out_pth,exist_ok=True)
        fname = os.path.splitext(os.path.basename(args.input))[0]
        df_embeddings = pd.DataFrame(umap_embeddings)

        print("Write embeedings to disk")
        df_embeddings.columns = [f"umap_{i}" for i in range(umap_embeddings.shape[1])]
        df_embeddings.to_pickle(os.path.join(out_pth,fname+".tumap"))

        print("Write umap model to disk")
        pickle.dump(fitted_umap, open(os.path.join(out_pth, fname + "_umap_model.pkl"), "wb"))

        print("Calculate label mask and write it to disk")
        embedding_mask = self.create_embedding_mask(embeddings)
        with mrcfile.new(
                os.path.join(
                    args.output,
                    fname + "_label_mask.mrci",
                ),
                overwrite=True,
        ) as mrc:
            mrc.set_data(embedding_mask)

        print("Done")
