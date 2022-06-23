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
from typing import Any
from tomotwin.modules.training.filepathtriplet import FilePathTriplet
from typing import List


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
        pass

    @abstractmethod
    def write_results_to_disk(self, path: str):
        """
        Writes results to give at the specified path
        :param path: Path
        :return: None
        """
        pass
