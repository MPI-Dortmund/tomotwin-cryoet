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
import itertools
from abc import ABC, abstractmethod
from typing import Tuple, List, Callable

import numpy as np


class VolumeDataset(ABC):
    """
    Abstract representation of a volume dataset
    """

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of volumes"""

    @abstractmethod
    def __getitem__(self, itemindex) -> np.array:
        """Return the an item with a certain index"""

    @abstractmethod
    def get_localization(self, itemindex) -> Tuple[int, int, int]:
        """Returns the center positions of an item"""


class FileNameVolumeDataset(VolumeDataset):
    """Here the subvolumes are representent by individual files"""

    def __init__(self, volumes: List[str], filereader: Callable[[str], np.array]):
        self.volumes = volumes
        self.filereader = filereader

    def __getitem__(self, itemindex) -> np.array:
        pth = self.volumes[itemindex]
        volume = self.filereader(pth)
        return volume

    def get_localization(self, itemindex) -> Tuple[float, float, float]:
        return None

    def __len__(self) -> int:
        return len(self.volumes)


class SlidingWindowVolumeData(VolumeDataset):
    """
    Represents a volume datset that came from a sliding window.
    """

    def __init__(self, volumes: np.array, stride: Tuple, boxsize: int):
        """
        :param volumes: array with shape (X,Y,Z,BS,BS,BS), where was BS is the box size and X,Y,Z relate
        to the position of the subvolume.
        """
        self.volumes = volumes
        self.stride = stride
        self.boxsize = boxsize

        self.center_coords = {}
        self.indicies = {}

        self.indicies = np.array(list(itertools.product(range(volumes.shape[0]),range(volumes.shape[1]),range(volumes.shape[2]))))
        self.center_coords = self.indicies * self.stride + (self.boxsize-1)/2


    def __len__(self) -> int:
        return self.volumes.shape[0] * self.volumes.shape[1] * self.volumes.shape[2]

    def __getitem__(self, itemindex) -> np.array:
        location = self.indicies[itemindex]
        vol = self.volumes[tuple(location)]
        return vol

    def get_localization(self, itemindex) -> Tuple[float, float, float]:
        """Return the center position"""
        return (self.center_coords[itemindex][0],self.center_coords[itemindex][1],self.center_coords[itemindex][2])
