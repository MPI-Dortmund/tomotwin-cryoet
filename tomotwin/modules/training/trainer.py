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
from typing import Any
from typing import List

from tomotwin.modules.training.filepathtriplet import FilePathTriplet


class Trainer(ABC):
    @abstractmethod
    def set_training_data(self, training_data: List[FilePathTriplet]) -> None:
        """
        :param training_data: List of file triplets used for training
        """

    @abstractmethod
    def set_test_data(self, test_data: List[FilePathTriplet]) -> None:
        """
        :param test_data: List of test data.
        :return: None
        """

    @abstractmethod
    def set_seed(self, seed) -> None:
        """
        Sets the seed value for the trainer.
        :param seed: Seed value
        :return: None
        """

    @abstractmethod
    def train(self) -> Any:
        """
        Starts the training.
        """

    @abstractmethod
    def get_model(self) -> Any:
        """
        Returns the trained model
        :return: The trained model
        """

    @abstractmethod
    def write_results_to_disk(self, path: str):
        """
        Writes results to give at the specified path
        :param path: Path
        :return: None
        """