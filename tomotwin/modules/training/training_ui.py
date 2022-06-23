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

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TrainingConfiguration:
    """
    Represents the configuration for training

    :param pdb_path: Path to folder with PDB volumes
    :param volume_path: Path to folder with subvolumes
    :param output_path: All output files are written here.
    :param num_epochs: Number of training epochs
    :param max_neg: Maximum number of triplets generated for a anchor-positive pair.
    :param threshold: Used for printing
    :param netconfig: Configuration file for the network.
    """

    pdb_path: str
    volume_path: str
    output_path: str
    num_epochs: int
    max_neg: int
    netconfig: str
    checkpoint: str
    distance: str
    validvolumes: str
    save_after_improvement: bool


class TrainingUI(ABC):
    """Interface to define"""

    @abstractmethod
    def run(self, args=None) -> None:
        """
        Runs the UI.
        :param args: Optional arguments that might need to pass to the parser. Can also be used for testing.
        :return: None
        """

    @abstractmethod
    def get_training_configuration(self) -> TrainingConfiguration:
        """
        Creates the training configuration and returns it.
        :return: A training configuration instance
        """
