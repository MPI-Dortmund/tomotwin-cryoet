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

from tomotwin.modules.inference.pick_ui import PickUI, PickConfiguration


class PickArgParseUI(PickUI):

    def __init__(self):
        self.locate_results_path: str = None
        self.target_reference: str = None
        self.output_path: str = None
        self.min_metric: float = None
        self.max_metric: float = None
        self.min_size: float = None
        self.max_size: float = None

    def get_pick_configuration(self) -> PickConfiguration:
        return PickConfiguration(
            locate_results_path = self.locate_results_path,
            target_reference = self.target_reference,
            output_path = self.output_path,
            min_metric = self.min_metric,
            max_metric = self.max_metric,
            min_size = self.min_size,
            max_size = self.max_size,
            )

    def run(self, args=None) -> None:
        parser_parent = argparse.ArgumentParser(
            description="Interface to locate particles with TomoTwin"
        )
        self.create_parser(parser_parent)

        args = parser_parent.parse_args(args)

        self.locate_results_path = args.locate_results
        self.target_reference = args.target
        self.output_path = args.output
        self.min_size = args.minsize
        self.max_size = args.maxsize
        self.min_metric = args.minmetric
        self.max_metric = args.maxmetric


    def create_parser(self, parser):

        parser.add_argument(
            "-l",
            "--locate_results",
            type=str,
            required=True,
            help="Path to the locate results file (output of locate command)",
        )

        parser.add_argument(
            "--target",
            type=str,
            default=None,
            nargs="+",
            help="Optional. Name of one or multiple target references",
        )

        parser.add_argument(
            "-o",
            "--output",
            type=str,
            required=True,
            help="Path to the output folder",
        )

        parser.add_argument(
            "--minmetric",
            type=float,
            help="Minimum value for metric",
        )

        parser.add_argument(
            "--maxmetric",
            type=float,
            help="Maximum value for metric",
        )

        parser.add_argument(
            "--minsize",
            type=float,
            help="Minimum value for size",
        )

        parser.add_argument(
            "--maxsize",
            type=float,
            help="Maximum value for size",
        )

