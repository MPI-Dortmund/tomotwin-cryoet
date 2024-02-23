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

from tomotwin.modules.training.filepathtriplet import FilePathTriplet
from tomotwin.modules.training.numpytriplet import NumpyTriplet


class TripletHandler(ABC):
    """
    Base class for handling a FilePathTriplets. It specific implementation creates numpytriplets out of FilePathTriplets
    """

    @abstractmethod
    def handle(self, triplet: FilePathTriplet) -> NumpyTriplet:
        """
        Read information from triplets and provides a tuble of numpy arrays. One for anchor, one for negativ and on for positive
        :param triplet: A FileTriplet
        :return: tuble of numpy arrays
        """
