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

import itertools
from abc import ABC, abstractmethod
from typing import Union, Tuple

import numpy as np
import numpy.lib.stride_tricks as tricks
from numpy.typing import NDArray

from tomotwin.modules.inference.volumedata import SimpleVolumeData, VolumeROI


class Boxer(ABC):
    """
    Abstract representation of a boxer
    """
    @abstractmethod
    def box(self, tomogram: NDArray) -> SimpleVolumeData:
        """Transforms tomogram into a set of boxes. Returns an array of arrays (boxes)"""


class InvalidZRangeConfiguration(Exception):
    ...


class SlidingWindowBoxer(Boxer):
    """
    Sliding window boxer
    """

    def __init__(self, box_size: int,
                 stride: Union[int, Tuple],
                 zrange: Tuple[int, int] = None,
                 mask: np.array = None):
        self.box_size = box_size
        self.stride = stride
        if isinstance(self.stride, int):
            self._stride_x = stride
            self._stride_y = stride
            self._stride_z = stride
        else:
            self._stride_x, self._stride_y, self._stride_z = stride

        self.zrange = zrange
        self.center_coords = None
        self.indicies = None
        self.mask = mask

    @staticmethod
    def _calc_volume_roi(tomogram: np.array,
                         stride: Tuple[int, int, int],
                         box_size: int,
                         zrange=None,
                         mask: np.array = None) -> VolumeROI:

        sliding_window_strides = SlidingWindowBoxer._calc_sliding_volumes(
            tomogram=tomogram,
            stride=stride,
            window_shape=(box_size, box_size, box_size)
        )  # this is actuallz a bit to complicated.

        # Calculate center coordinates
        indicies = np.array(
            list(itertools.product(
                range(sliding_window_strides.shape[0]),
                range(sliding_window_strides.shape[1]),
                range(sliding_window_strides.shape[2]))
            )
        )

        odd_factor = box_size % 2
        center_coords = indicies * stride + (box_size - odd_factor) / 2
        print(center_coords[:,0:10])
        if zrange is not None:
            center_coords[:, 0] = zrange[0] + center_coords[:, 0]
        relevant_center_coords = []

        # Apply mask if necessary
        if mask is not None:
            for c in center_coords:
                if mask[tuple(c.astype(int).tolist())]:
                    relevant_center_coords.append(c)
            relevant_center_coords = np.vstack(relevant_center_coords)
        else:
            relevant_center_coords = center_coords
        r = VolumeROI(center_coords=relevant_center_coords, box_size=box_size)
        return r

    @staticmethod
    def _calc_sliding_volumes(tomogram: np.array,
                              stride: Tuple[int, int, int],
                              window_shape: Tuple[int, int, int],
                              ) -> np.array:
        """
        Calculate the sliding subvolumes (views) and returns their center coordinates
        :return:
        """

        sliding_window_views = tricks.sliding_window_view(
            tomogram, window_shape=window_shape
        )

        sliding_window_strides = sliding_window_views[
                                 ::stride[2], :: stride[1], :: stride[0]
                                 ]

        return sliding_window_strides

    def box(self, tomogram: NDArray) -> SimpleVolumeData:
        """
        Transforms tomogram into a set of boxes
        """
        if self.zrange:
            if self.zrange[0] < 0 or (self.zrange[0] >= self.zrange[1]) or self.zrange[1] > tomogram.shape[0]:
                raise InvalidZRangeConfiguration()

            tomogram = tomogram[self.zrange[0]:self.zrange[1]]

        roi = SlidingWindowBoxer._calc_volume_roi(
            tomogram,
            stride=(self._stride_x, self._stride_y, self._stride_z),
            box_size=self.box_size,
            zrange=self.zrange,
            mask=self.mask

        )

        data = SimpleVolumeData(
            volumes=tomogram,
            roi=roi
        )

        return data
