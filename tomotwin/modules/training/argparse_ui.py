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

from .training_ui import TrainingUI, TrainingConfiguration
import argparse
from tomotwin.modules.common.distances import DistanceManager


class TrainingArgParseUI(TrainingUI):
    def __init__(self):
        self.batchsize = None
        self.learningrate = None
        self.num_epochs = None
        self.pdb_path = None
        self.volume_path = None
        self.max_neg = None
        self.outpath = None
        self.margin = None
        self.config = None
        self.checkpoint = None
        self.distance = None

    def create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Training interface for TomoTwin",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "-p",
            "--pdbpath",
            type=str,
            help="Path to PDB files that should be use for training.",
            default=None
        )
        parser.add_argument(
            "-v",
            "--volpath",
            type=str,
            required=True,
            help="Path to subtomogram volumes that should be used for training",
        )

        parser.add_argument(
            "--validvolumes",
            type=str,
            default=None,
            help="Optional path for validation volumes. If not given, it will be generated from the training data.",
        )

        parser.add_argument(
            "-o",
            "--outpath",
            type=str,
            required=True,
            help="All output files are written in that path.",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=10,
            metavar="N",
            help="Number of epochs to train for (default 10)",
        )
        parser.add_argument(
            "--max_neg",
            type=int,
            default=1,
            metavar="MAX",
            help="Maximum number of triplets generated for a anchor-positive pair. Large value can lead to a combinatorial explosion.",
        )
        parser.add_argument(
            "-nc",
            "--netconfiguration",
            type=str,
            required=True,
            metavar="NETWORK_JSON_FILE",
            help="Network configuration json file",
        )

        parser.add_argument(
            "--checkpoint",
            type=str,
            default=None,
            help="Path to a model where the training should start from.",
        )

        parser.add_argument(
            "-d",
            "--distance",
            type=str,
            choices=DistanceManager().distances.keys(),
            default="COSINE",
            help="Distance function that should be used",
        )

        parser.add_argument(
            "--save_after_improvement",
            action='store_true',
            default=False,
            help="Save seperate model for each epoch where the F1 score improves",
        )



        return parser

    def run(self, args=None) -> None:
        parser = self.create_parser()
        args = parser.parse_args(args)

        self.num_epochs = args.epochs
        self.pdb_path = args.pdbpath
        self.volume_path = args.volpath
        self.max_neg = args.max_neg
        self.outpath = args.outpath
        self.config = args.netconfiguration
        self.checkpoint = args.checkpoint
        self.distance = args.distance
        self.validvolumes = args.validvolumes
        self.save_after_improvement = args.save_after_improvement

    def get_training_configuration(self) -> TrainingConfiguration:
        tconf = TrainingConfiguration(
            pdb_path=self.pdb_path,
            volume_path=self.volume_path,
            num_epochs=self.num_epochs,
            max_neg=self.max_neg,
            output_path=self.outpath,
            netconfig=self.config,
            checkpoint=self.checkpoint,
            distance=self.distance,
            validvolumes=self.validvolumes,
            save_after_improvement=self.save_after_improvement
        )
        return tconf
