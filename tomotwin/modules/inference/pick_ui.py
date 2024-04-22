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
class PickConfiguration:
    """
    Represents the configuration for picking

    :param locate_results_path: Path to locate results
    :param target_reference: Name of the target reference
    :param output_path: Path where to write the coordinate files
    :param min_metric: Minimum metric
    :param max_metric: Maximum metric
    :param min_size: Minimum size of a maxima
    :param max_size: Maximum size of a maxima

    """

    locate_results_path: str
    target_reference: str
    output_path: str
    min_metric: float
    max_metric: float
    min_size: float
    max_size: float

class PickUI(ABC):
    """Interface to define"""

    @abstractmethod
    def run(self, args=None) -> None:
        """
        Runs the UI.
        :param args: Optional arguments that might need to pass to the parser. Can also be used for testing.
        :return: None
        """

    @abstractmethod
    def get_pick_configuration(self) -> PickConfiguration:
        """
        Creates the pick configuration and returns it.
        :return: PickConfiguration
        """