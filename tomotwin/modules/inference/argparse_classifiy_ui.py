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
from tomotwin.modules.inference.classify_ui import (
    ClassifyUI,
    ClassifyConfiguration,
    ClassifyMode,
)


class ClassifiyArgParseUI(ClassifyUI):
    """
    Argparse interface for the classify command
    """

    def __init__(self):
        self.reference_pth = None
        self.volume_pth = None
        self.output_pth = None
        self.threshold = None
        self.mode = None

    def run(self, args=None) -> None:
        parser = self.create_parser()
        args = parser.parse_args(args)
        self.reference_pth = args.references
        self.volume_pth = args.volumes
        self.output_pth = args.output
        if "distance" in sys.argv[1]:
            self.mode = ClassifyMode.DISTANCE
            self.threshold = args.threshold

    def get_classification_configuration(self) -> ClassifyConfiguration:
        conf = ClassifyConfiguration(
            reference_embeddings_path=self.reference_pth,
            volume_embeddings_path=self.volume_pth,
            output_path=self.output_pth,
            mode=self.mode,
            threshold=self.threshold,
        )
        return conf

    @staticmethod
    def create_distance_parser(parser):
        """
        Create parser for distance subcommand
        """
        parser.add_argument(
            "-r",
            "--references",
            type=str,
            required=True,
            help="Path to reference embeddings file",
        )

        parser.add_argument(
            "-v",
            "--volumes",
            type=str,
            required=True,
            help="Path to volume embeddings file",
        )

        parser.add_argument(
            "-o",
            "--output",
            type=str,
            required=True,
            help="Path to output folder.",
        )

        parser.add_argument(
            "-t",
            "--threshold",
            type=float,
            help="Cut-off threshold for evaluation. Embeddings with distance to a reference greater than this threshold get assign a zero probability for that reference.",
        )

    def create_parser(self) -> argparse.ArgumentParser:
        """
        Create parser for the classify command
        """

        parser_parent = argparse.ArgumentParser(
            description="Interface to calculate Embeddings for TomoTwin"
        )
        subparsers = parser_parent.add_subparsers(help="sub-command help")
        parser_embed_volume = subparsers.add_parser(
            "distance",
            help="Classify volumes by distance to the references.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self.create_distance_parser(parser_embed_volume)

        return parser_parent
