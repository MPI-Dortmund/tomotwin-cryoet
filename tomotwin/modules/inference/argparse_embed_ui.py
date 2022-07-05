"""
MIT License

Copyright (c) 2021 MPI-Dortmund

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import argparse
import sys

from tomotwin.modules.inference.embed_ui import EmbedUI, EmbedConfiguration, EmbedMode



class EmbedArgParseUI(EmbedUI):
    """
    Argparse interface to embedor
    """
    def __init__(self):
        self.modelpth = None
        self.volumes = None
        self.batchsize = None
        self.output = None
        self.window_size = None
        self.stride = None
        self.mode = None

    def run(self, args=None) -> None:
        parser = self.create_parser()
        args = parser.parse_args(args)

        self.modelpth = args.modelpth
        self.volumes = args.volumes
        self.batchsize = args.batchsize
        self.output = args.outpath


        if "subvolumes" in sys.argv[1]:
            self.mode = EmbedMode.VOLUMES

        if "tomogram" in sys.argv[1]:
            self.mode = EmbedMode.TOMO
            self.window_size = args.windowsize
            self.stride = args.stride
            if len(self.stride) == 1:
                self.stride = self.stride*3

    def get_embed_configuration(self) -> EmbedConfiguration:
        conf = EmbedConfiguration(
            model_path=self.modelpth,
            volumes_path=self.volumes,
            output_path=self.output,
            mode=self.mode,
            batchsize=self.batchsize,
            window_size=self.window_size,
            stride=self.stride,
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
            "-w",
            "--windowsize",
            type=int,
            default=37,
            help="Size of the sliding window",
        )

        parser.add_argument(
            "-s",
            "--stride",
            type=int,
            default=2,
            nargs='+',
            help="Stride of the sliding window. Either an integer or a tuple of 3 numbers representing the slides in x,y,z",
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
