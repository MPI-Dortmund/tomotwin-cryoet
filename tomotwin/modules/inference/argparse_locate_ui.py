import argparse
import sys

from tomotwin.modules.inference.locate_ui import LocateUI, LocateConfiguration, LocateMode

class LocateArgParseUI(LocateUI):

    def __init__(self):
        self.probability_path = None
        self.output_path = None
        self.pthresh = None
        self.dthresh = None
        self.tolerance = None
        self.boxsize = None
        self.mode = None

    def run(self, args=None) -> None:
        parser = self.create_parser()
        args = parser.parse_args(args)

        if "findmax" in sys.argv[1]:
            self.probability_path = args.probability
            self.tolerance = args.tolerance
            self.output_path = args.output
            self.boxsize = args.boxsize
            self.mode = LocateMode.FINDMAX




    def get_locate_configuration(self) -> LocateConfiguration:
        conf = LocateConfiguration(
            probability_path=self.probability_path,
            output_path=self.output_path,
            probability_threshold=self.pthresh,
            distance_threshold=self.dthresh,
            mode=self.mode,
            boxsize=self.boxsize,
            tolerance=self.tolerance
        )
        return conf

    def create_findmax_parser(self, parser):
        parser.add_argument(
            "-p",
            "--probability",
            type=str,
            required=True,
            help="Path to the probability file (output of classify command)",
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
