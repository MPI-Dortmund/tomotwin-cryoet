import unittest
from tomotwin.modules.training.filepathtriplet import FilePathTriplet
from tomotwin.modules.training.mrctriplethandler import MRCTripletHandler
import os
import numpy as np
class MyTestCase(unittest.TestCase):
    def test_handle(self):
        pthvol = os.path.join(os.path.dirname(__file__), "../resources/tests/mrctriplethandler/model_0_1BXN_1.mrc")

        triplet = FilePathTriplet(pthvol,pthvol,pthvol)
        handler = MRCTripletHandler()
        nptripplet = handler.handle(triplet)

        np.testing.assert_array_equal(nptripplet.anchor.shape, (37, 37, 37))
        np.testing.assert_array_equal(nptripplet.positive.shape, (37, 37, 37))
        np.testing.assert_array_equal(nptripplet.negative.shape, (37, 37, 37))


if __name__ == '__main__':
    unittest.main()
