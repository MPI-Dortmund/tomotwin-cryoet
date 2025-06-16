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
from typing import Tuple, List, Callable

import numpy as np


@dataclass
class VolumeROI:
    """
    Represents a region of interesion within a volume
    """

    center_coords: np.array
    box_size: int


class VolumeDataset(ABC):
    """
    Abstract representation of a volume dataset
    """

    @abstractmethod
    def __len__(self) -> int:
        """Returns the number of volumes"""

    @abstractmethod
    def __getitem__(self, itemindex) -> np.array:
        """Return an item with a certain index"""

    @abstractmethod
    def get_localization(self, itemindex) -> Tuple[int, int, int]:
        """Returns the center positions of an item"""


class FileNameVolumeDataset(VolumeDataset):
    """Here the subvolumes are represented by individual files"""

    def __init__(self, volumes: List[str], filereader: Callable[[str], np.array]):
        self.volumes = volumes
        self.filereader = filereader

    def __getitem__(self, itemindex) -> np.array:
        pth = self.volumes[itemindex]
        volume = self.filereader(pth)
        return volume

    def __len__(self) -> int:
        return len(self.volumes)

    def get_localization(self, itemindex) -> None:
        return None


class SimpleVolumeData(VolumeDataset):
    """
    Represents a volume dataset that came from a sliding window.
    """

    def __init__(self,
                 volumes: np.array,
                 roi: VolumeROI):
        """
        :param volumes: array with shape (X,Y,Z,BS,BS,BS), where was BS is the box size and X,Y,Z relate
        to the position of the subvolume.
        """
        self.volumes = volumes
        self.roi = roi

    def __len__(self) -> int:
        return len(self.roi.center_coords)

    def __getitem__(self, itemindex) -> np.array:
        p = self.roi.center_coords[itemindex].astype(int)
        p_min = p - int((self.roi.box_size - 1) / 2)
        p_max = p + int((self.roi.box_size - 1) / 2 + 1)
        v = self.volumes[p_min[0]:p_max[0], p_min[1]:p_max[1], p_min[2]:p_max[2]]

        return v

    def get_localization(self, itemindex) -> Tuple[int, int, int]:
        return self.roi.center_coords[itemindex]
