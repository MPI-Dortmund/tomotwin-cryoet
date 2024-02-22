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

import numpy as np

import tomotwin.modules.common.preprocess as preprocess
from tomotwin.modules.common.io.mrc_format import MrcFormat
from tomotwin.modules.training.numpytriplet import NumpyTriplet
from tomotwin.modules.training.triplethandler import TripletHandler, FilePathTriplet


class MRCTripletHandler(TripletHandler):
    """
    Handles MRC images to provide numpy triplets. We could refactor it into general Image handler out of it.
    """

    @staticmethod
    def read_mrc_and_norm(pth: str) -> np.array:
        vol = MrcFormat.read(pth)
        vol = preprocess.norm(vol)
        return vol

    def handle(self, triplet: FilePathTriplet) -> NumpyTriplet:

        try:
            anchor = self.read_mrc_and_norm(triplet.anchor)
        except ValueError as e:
            print(f"Can't read {triplet.anchor}")
            raise e
        try:
            positive = self.read_mrc_and_norm(triplet.positive)
        except ValueError as e:
            print(f"Can't read {triplet.positive}")
            raise e
        try:
            negative = self.read_mrc_and_norm(triplet.negative)

        except ValueError as e:
            print(f"Can't read {triplet.negative}")
            raise e

        triplet = NumpyTriplet(
            anchor=anchor,
            positive=positive,
            negative=negative,
        )

        return triplet
