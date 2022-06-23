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
from typing import List
from enum import Enum, auto


class EmbedMode(Enum):
    """
    Enumeration of embedding modes
    """
    TOMO = auto()
    VOLUMES = auto()


@dataclass
class EmbedConfiguration:
    """
    Represents the configuration for embedding calculation

    :param model_path: Path to the model
    :param volumes: Path to folder with subvolumes, subvolumes itself or path to a tomogram
    :param mode: EmbedMode to run
    :param window_size: Running window size
    :param stride: Stride for running window
    """

    model_path: str
    volumes_path: List[str]
    output_path: str
    mode: EmbedMode
    batchsize: int
    window_size: bool = None
    stride: int = None


class EmbedUI(ABC):
    """Interface to define"""

    @abstractmethod
    def run(self, args=None) -> None:
        """
        Runs the UI.
        :param args: Optional arguments that might need to pass to the parser. Can also be used for testing.
        :return: None
        """

    @abstractmethod
    def get_embed_configuration(self) -> EmbedConfiguration:
        """
        Creates the embed configuration and returns it.
        :return: A embed configuration instance
        """
