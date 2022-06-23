from tomotwin.modules.training.triplethandler import *
from tomotwin.modules.training.numpytriplet import NumpyTriplet
class TripletFakeHandler(TripletHandler):

    def __init__(self, pos_arr, neg_arr, anchor_arr):
        self.pos_arr = pos_arr
        self.neg_arr = neg_arr
        self.anchor = anchor_arr

    def handle(self, triplet : FilePathTriplet) -> NumpyTriplet:
        triplet = NumpyTriplet(positive=self.pos_arr, negative=self.neg_arr,anchor=self.anchor)
        return triplet