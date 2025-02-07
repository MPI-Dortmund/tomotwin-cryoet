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
import sys

from tomotwin.modules.inference.embed_ui import EmbedUI, EmbedConfiguration, EmbedMode, DistrMode


class EmbedArgParseUI(EmbedUI):
    """
    Argparse interface to embedor
    """
    def __init__(self):
        self.modelpth = None
        self.volumes = None
        self.batchsize = None
        self.output = None
        self.stride = None
        self.mode = None
        self.zrange = None
        self.maskpth = None
        self.distr_mode = None
        self.padding = None

    def run(self, args=None) -> None:
        parser = self.create_parser()
        args = parser.parse_args(args)

        self.modelpth = args.modelpth
        self.volumes = args.volumes
        self.batchsize = args.batchsize
        self.output = args.outpath


        if "subvolumes" in sys.argv[1]:
            self.mode = EmbedMode.VOLUMES
            self.distr_mode = DistrMode.DP

        if "tomogram" in sys.argv[1]:
            self.mode = EmbedMode.TOMO
            self.stride = args.stride
            self.zrange = args.zrange
            if len(self.stride) == 1:
                self.stride = self.stride * 3
            self.maskpth = args.mask
            self.distr_mode = DistrMode.DDP
            self.padding = args.padding
            if args.distribution_mode == 0:
                self.distr_mode = DistrMode.DP
            self.padding = args.padding

    def get_embed_configuration(self) -> EmbedConfiguration:
        conf = EmbedConfiguration(
            model_path=self.modelpth,
            volumes_path=self.volumes,
            output_path=self.output,
            mode=self.mode,
            batchsize=self.batchsize,
            stride=self.stride,
            zrange=self.zrange,
            maskpth=self.maskpth,
            distr_mode=self.distr_mode,
            padding = self.padding
        )
        return conf

    @staticmethod
    def create_volume_parser(parser):
        """
        Creates parser for the volume command
        """

        parser.add_argument(
            "-m",
            "--modelpth",
            type=str,
            required=True,
            help="Path to the model with saved weights",
        )

        parser.add_argument(
            "-v",
            "--volumes",
            type=str,
            required=True,
            nargs="+",
            help="Path to volumes. That can be either a folder containing volumes or one or multiple volumes seperated by whitespace",
        )

        parser.add_argument(
            "-b",
            "--batchsize",
            type=int,
            default=128,
            required=False,
            help="Batch size during calculating the embeddings",
        )

        parser.add_argument(
            "-o",
            "--outpath",
            type=str,
            required=True,
            help="All output files are written in that path.",
        )



    @staticmethod
    def create_tomo_parser(parser):
        """
        Creates parser for the tomo command
        """

        parser.add_argument(
            "-m",
            "--modelpth",
            type=str,
            required=True,
            help="Path to the tomotwin model",
        )

        parser.add_argument(
            "-v",
            "--volumes",
            type=str,
            required=True,
            help="Path to a single tomogram file",
        )

        parser.add_argument(
            "-b",
            "--batchsize",
            type=int,
            default=64,
            required=False,
            help="Batch size during calculating the embeddings",
        )

        parser.add_argument(
            "-o",
            "--outpath",
            type=str,
            required=True,
            help="All output files are written in that path.",
        )

        parser.add_argument(
            "-s",
            "--stride",
            type=int,
            default=[2],
            nargs='+',
            help="Stride of the sliding window. Either an integer or a tuple of 3 numbers representing the slides in x,y,z",
        )

        parser.add_argument(
            "-z",
            "--zrange",
            type=int,
            default=None,
            nargs=2,
            help="Minimum z and maximum z for to run the sliding window on. Handy to skip the void volume in order to speed up the embedding.",
        )

        parser.add_argument(
            "--mask",
            type=str,
            required=False,
            default=None,
            help="Path to binary mask to define embedding region (mrc format). All values != 0 are interpreted as 'True'.",
        )

        parser.add_argument(
            "-d",
            "--distribution_mode",
            type=int,
            required=False,
            choices=[0, 1],
            default=1,
            help="0: DataParallel,  1: Faster parallelism mode using DistributedDataParallel"
        )

        parser.add_argument(
            "-p",
            "--padding",
            type=bool,
            required=False,
            default=True,
            help="padding value added to all axis of the tomogram from both sides, can be useful to pick particles at the edges"
        )

    def create_parser(self) -> argparse.ArgumentParser:
        """
        Create the embedor parser
        """

        parser_parent = argparse.ArgumentParser(
            description="Interface to calculate Embeddings for TomoTwin"
        )
        subparsers = parser_parent.add_subparsers(help="sub-command help")
        parser_embed_volume = subparsers.add_parser(
            "subvolumes",
            help="Embed one or multiple subvolumes.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self.create_volume_parser(parser_embed_volume)

        parser_embed_tomogram = subparsers.add_parser(
            "tomogram",
            help="Embed a tomogram",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self.create_tomo_parser(parser_embed_tomogram)

        return parser_parent
