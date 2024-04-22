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
