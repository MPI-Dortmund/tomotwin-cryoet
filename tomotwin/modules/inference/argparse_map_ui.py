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

from tomotwin.modules.inference.map_ui import (
    MapUI,
    MapConfiguration,
    MapMode,
)


class MapArgParseUI(MapUI):
    """
    Argparse interface for the map command
    """

    def __init__(self):
        self.reference_pth = None
        self.volume_pth = None
        self.output_pth = None
        self.mode = None

    def run(self, args=None) -> None:
        parser = self.create_parser()
        args = parser.parse_args(args)
        self.reference_pth = args.references
        self.volume_pth = args.volumes
        self.output_pth = args.output
        if "distance" in sys.argv[1]:
            self.mode = MapMode.DISTANCE
            self.skip_refinement = not args.refine

    def get_map_configuration(self) -> MapConfiguration:
        conf = MapConfiguration(
            reference_embeddings_path=self.reference_pth,
            volume_embeddings_path=self.volume_pth,
            output_path=self.output_pth,
            mode=self.mode,
            skip_refinement=self.skip_refinement
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
            "--refine",
            action='store_true',
            default=False,
            help="Do reference refinement",
        )

        parser.add_argument(
            "-o",
            "--output",
            type=str,
            required=True,
            help="Path to output folder.",
        )

    def create_parser(self) -> argparse.ArgumentParser:
        """
        Create parser for the map command
        """

        parser_parent = argparse.ArgumentParser(
            description="Interface to calculate embeddings for TomoTwin"
        )
        subparsers = parser_parent.add_subparsers(help="sub-command help")
        parser_embed_volume = subparsers.add_parser(
            "distance",
            help="Map volumes by distance to the references.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self.create_distance_parser(parser_embed_volume)

        return parser_parent
