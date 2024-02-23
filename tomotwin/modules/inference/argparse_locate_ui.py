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

from tomotwin.modules.inference.locate_ui import LocateUI, LocateConfiguration, LocateMode


class LocateArgParseUI(LocateUI):

    def __init__(self):
        self.map_path = None
        self.output_path = None
        self.tolerance = None
        self.boxsize = None
        self.mode = None
        self.global_min = None
        self.processes = 4
        self.write_heatmaps = False

    def run(self, args=None) -> None:
        parser = self.create_parser()
        args = parser.parse_args(args)

        if "findmax" in sys.argv[1]:
            self.map_path = args.map
            self.tolerance = args.tolerance
            self.output_path = args.output
            self.boxsize = args.boxsize
            self.mode = LocateMode.FINDMAX
            self.global_min = args.global_min
            self.processes = args.processes
            self.write_heatmaps = args.write_heatmaps




    def get_locate_configuration(self) -> LocateConfiguration:
        conf = LocateConfiguration(
            map_path=self.map_path,
            output_path=self.output_path,
            mode=self.mode,
            boxsize=self.boxsize,
            tolerance=self.tolerance,
            global_min=self.global_min,
            processes=self.processes,
            write_heatmaps=self.write_heatmaps
        )
        return conf

    def create_findmax_parser(self, parser):
        parser.add_argument(
            "-m",
            "--map",
            type=str,
            required=True,
            help="Path to the map file ( *.tmap, output of map command)",
        )

        parser.add_argument(
            "-o",
            "--output",
            type=str,
            required=True,
            help="Path to the output folder",
        )

        parser.add_argument(
            "-t",
            "--tolerance",
            type=float,
            default=0.2,
            help="Tolerance value",
        )

        parser.add_argument(
            "-b",
            "--boxsize",
            default=37,
            help="Provide the box size you want to use for picking or a json file with reference_filename as keys and box sizes as values.",
        )

        parser.add_argument(
            "-g",
            "--global_min",
            type=float,
            default=0.5,
            help="Global minimum of the find max procedure. Maximums below value will be ignored. Higher values give faster runtime.",
        )

        parser.add_argument(
            "--processes",
            type=int,
            default=4,
            help="Number of parallel processed references. More processes require more memory.",
        )

        parser.add_argument(
            "--write_heatmaps",
            action='store_true',
            default=False,
            help="If true, a heatmap is written for every reference. Default is false, as they require some space and are not necessary for the further process.",
        )

    def create_parser(self) -> argparse.ArgumentParser:
        """Create locate parser"""
        parser_parent = argparse.ArgumentParser(
            description="Interface to locate particles with TomoTwin"
        )
        subparsers = parser_parent.add_subparsers(help="sub-command help")

        findmax_parser = subparsers.add_parser(
            "findmax",
            help="Locate your particles with a find maxima locator.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self.create_findmax_parser(findmax_parser)

        return parser_parent
