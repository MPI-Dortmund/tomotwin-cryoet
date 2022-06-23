import unittest
import numpy as np
from tomotwin.modules.common.findmax.findmax import find_maxima


class FindMaxTests(unittest.TestCase):

    def test_findmax_locator_find2(self):
        import os
        import mrcfile
        mrcpdb = os.path.join(os.path.dirname(__file__), "../resources/tests/FindMaxLocator/5MRC.mrc")
        with mrcfile.open(mrcpdb, permissive=True) as mrc:
            vol = mrc.data

        max, _ = find_maxima(vol, tolerance=0.2, global_min=0.9)
        self.assertEqual(len(max), 2)

    def test_findmax_locator_find3(self):

        vol = np.zeros(shape=(30, 30, 30))

        vol[5, 5, 5] = 1
        vol[10, 10, 10] = 1
        vol[20, 20, 20] = 1
        max, _ = find_maxima(vol, tolerance=0.2, global_min=0.9)
        self.assertEqual(len(max), 3)

    def test_findmax_locator_1d(self):
        '''
        at least one should be found
        '''

        data = np.array([0.93, 0.32, 0.18, 0.2, 0.57, 0.6, 0.96, 0.65, 0.75, 0.65])
        max, _ = find_maxima(data, tolerance=0.3, global_min=0.5)
        self.assertGreaterEqual(len(max), 1)


if __name__ == '__main__':
    unittest.main()
