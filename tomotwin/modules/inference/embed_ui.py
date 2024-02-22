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
from enum import Enum, auto
from typing import List, Tuple, Union


class EmbedMode(Enum):
    """
    Enumeration of embedding modes
    """
    TOMO = auto()
    VOLUMES = auto()


class DistrMode(Enum):
    """
    Enumeration of Distribution mode
    """
    DP = 0
    DDP = 1


@dataclass
class EmbedConfiguration:
    """
    Represents the configuration for embedding calculation
    """

    model_path: str
    volumes_path: Union[str, List[str]]
    output_path: str
    mode: EmbedMode
    batchsize: int
    stride: int = None
    zrange: Tuple[int, int] = None
    maskpth: str = None
    distr_mode: DistrMode = None


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
        :return: An embed configuration instance
        """
