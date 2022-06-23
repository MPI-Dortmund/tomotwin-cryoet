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
        self.stride = None
        self.mode = None

    def run(self, args=None) -> None:
        parser = self.create_parser()
        args = parser.parse_args(args)

        if "simple" in sys.argv[1]:
            self.probability_path = args.probability
            self.output_path = args.output
            self.pthresh = args.pthresh
            self.dthresh = args.dthresh
            self.mode = LocateMode.SIMPLE

        if "findmax" in sys.argv[1]:
            self.probability_path = args.probability
            self.tolerance = args.tolerance
            self.output_path = args.output
            self.stride = args.stride
            self.boxsize = args.boxsize
            self.mode = LocateMode.FINDMAX



    def get_locate_configuration(self) -> LocateConfiguration:
        conf = LocateConfiguration(
            probability_path=self.probability_path,
            output_path=self.output_path,
            probability_threshold=self.pthresh,
            distance_threshold=self.dthresh,
            mode=self.mode,
            stride=self.stride,
            boxsize=self.boxsize,
            tolerance=self.tolerance
        )
        return conf

    def create_naive_parser(self, parser):
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
            "-pt",
            "--pthresh",
            type=float,
            default=None,
            help="Only keep particles with a probability above that threshold",
        )

        parser.add_argument(
            "-dt",
            "--dthresh",
            type=float,
            default=None,
            help="Only keep particles with a distance blow that threshold",
        )

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
            "-s",
            "--stride",
            type=float,
            default=None,
            nargs='+',
            help="(Optional) Provide the same stride a during embedding",
        )

        parser.add_argument(
            "-b",
            "--boxsize",
            default=None,
            help="Provide the box size you want to use for picking or a json file with reference_filename as keys and box sizes as values.",
        )

    def create_parser(self) -> argparse.ArgumentParser:
        """Create locate parser"""
        parser_parent = argparse.ArgumentParser(
            description="Interface to locate particles with TomoTwin"
        )
        subparsers = parser_parent.add_subparsers(help="sub-command help")

        naive_parser = subparsers.add_parser(
            "simple",
            help="Locate your particles with a quite naive locator.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self.create_naive_parser(naive_parser)

        findmax_parser = subparsers.add_parser(
            "findmax",
            help="Locate your particles with a find maxima locator.",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        self.create_findmax_parser(findmax_parser)

        return parser_parent
