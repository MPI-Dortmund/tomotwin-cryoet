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
from typing import Union

class LocateMode(Enum):
    """
    Enumeration of embedding modes
    """
    FINDMAX = auto()

@dataclass
class LocateConfiguration:
    """
    Represents the configuration for classification calculation

    :param probability_path: Kee
    :param output_path: Path to embedded volumes
    :param probability_threshold: Only keep picks with a probability above that threshold
    :param distance_threshold: Only keep picks with a distance below that threshold
    """

    map_path: str
    output_path: str
    tolerance: float
    boxsize: Union[str,int]
    mode: LocateMode
    global_min: float
    processes: int
    write_heatmaps: bool

class LocateUI(ABC):
    """Interface to define"""

    @abstractmethod
    def run(self, args=None) -> None:
        """
        Runs the UI.
        :param args: Optional arguments that might need to pass to the parser. Can also be used for testing.
        :return: None
        """

    @abstractmethod
    def get_locate_configuration(self) -> LocateConfiguration:
        """
        Creates the locate configuration and returns it.
        :return: LocateConfiguration
        """