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