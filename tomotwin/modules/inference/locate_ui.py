from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Union

class LocateMode(Enum):
    """
    Enumeration of embedding modes
    """
    SIMPLE = auto()
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

    probability_path: str
    output_path: str
    tolerance: float
    boxsize: Union[str,int]
    stride: int
    mode: LocateMode
    probability_threshold: float = 0
    distance_threshold: float = 0

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