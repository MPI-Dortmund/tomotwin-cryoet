import argparse
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from tomotwin.modules.tools.tomotwintool import TomoTwinTool


class MakeTargetEmbeddings(TomoTwinTool):
    def get_command_name(self) -> str:
        '''
        :return: The name of the command
        '''
        return "make_target"

    def create_parser(self, parentparser: ArgumentParser) -> ArgumentParser:
        """
        :param parentparser: ArgumentPaser where the subparser for this tool needs to be added.
        :return: Argument parser that was added to the parentparser
        """

        parser = parentparser.add_parser(
            self.get_command_name(),
            help="Create target embeddings from clsutering results",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        parser.add_argument(
            "-i",
            "--input",
            type=str,
            required=True,
            help="Embeddings that were used for clustering",
        )

        parser.add_argument(
            "-c",
            "--clusters",
            type=str,
            required=True,
            help="Output csv file from clustering",
        )

        parser.add_argument(
            "-o", "--output", type=str, required=True, help="Output folder"
        )

        return parser

    def make_targets(
            self, embeddings: pd.DataFrame, clusters: pd.DataFrame
    ) -> pd.DataFrame:
        '''
        Calculates the average embedding for each target (Cluster)
        :param embeddings: Embeddings
        :param clusters: Label for each embedding to row a cluster
        :return: List of average embeddings
        '''
        embeddings = embeddings.drop(
            columns=["X", "Y", "Z", "filepath"], errors="ignore"
        )
        targets = []
        target_names = []
        for cluster in set(clusters):
            if cluster == 0:
                continue
            target = (
                embeddings.loc[clusters == cluster, :].astype(np.float32).mean(axis=0)
            )
            target = target.to_frame().T
            targets.append(target)
            target_names.append(f"cluster_{cluster}")

        targets = pd.concat(targets, ignore_index=True)
        targets["filepath"] = target_names
        return targets

    def run(self, args):
        '''
        runs the tool :-)
        '''

        print("Read embeddings")
        embeddings = pd.read_pickle(args.input)

        print("Read clusters")
        clusters = pd.read_csv(args.clusters)["MANUAL_CLUSTER_ID"]

        assert len(embeddings) == len(
            clusters
        ), "Cluster and embedding file are not compatible."

        print("Make targets")
        targets = self.make_targets(embeddings, clusters)

        print("Write targets")
        os.makedirs(args.output, exist_ok="True")
        pth_ref = os.path.join(args.output, "cluster.temb")

        targets.to_pickle(pth_ref)
