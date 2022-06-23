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
import numpy.lib.stride_tricks as tricks
from numpy.typing import NDArray
from typing import Union, Tuple
from tomotwin.modules.inference.volumedata import SlidingWindowVolumeData


class Boxer(ABC):
    """
    Abstract representation of a boxer
    """
    @abstractmethod
    def box(self, tomogram: NDArray) -> SlidingWindowVolumeData:
        """Transforms tomogram into a set of boxes. Returns an array of arrays (boxes)"""


class SlidingWindowBoxer(Boxer):
    """
    Sliding window boxer
    """

    def __init__(self, box_size: int, stride: Union[int, Tuple]):
        self.box_size = box_size
        self.stride = stride
        if isinstance(self.stride,int):
            self._stride_x = stride
            self._stride_y = stride
            self._stride_z = stride
        else:
            self._stride_x, self._stride_y, self._stride_z = stride

        self.center_coords = None

    def box(self, tomogram: NDArray) -> SlidingWindowVolumeData:
        """
        Transforms tomogram into a set of boxes
        """

        window_shape = (self.box_size, self.box_size, self.box_size)
        sliding_window_views = tricks.sliding_window_view(
            tomogram, window_shape=window_shape
        )

        sliding_window_strides = sliding_window_views[
            :: self._stride_z, :: self._stride_y, :: self._stride_x
        ]

        data = SlidingWindowVolumeData(
            volumes=sliding_window_strides, boxsize=self.box_size, stride=(self._stride_x,self._stride_y,self._stride_z)
        )

        return data
