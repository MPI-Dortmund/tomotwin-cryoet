import numpy as np

from tomotwin.modules.training.triplethandler import TripletHandler, FilePathTriplet
from tomotwin.modules.training.numpytriplet import NumpyTriplet
import tomotwin.modules.common.io as io
import tomotwin.modules.common.preprocess as preprocess



class MRCTripletHandler(TripletHandler):
    """
    Handles MRC images to provide numpy triplets. We could refactor it into general Image handler out of it.
    """

    @staticmethod
    def read_mrc_and_norm(pth: str) -> np.array:
        vol = io.read_mrc(pth)
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
