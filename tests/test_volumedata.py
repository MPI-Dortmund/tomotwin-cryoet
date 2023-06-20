import unittest
from tomotwin.modules.inference.volumedata import SimpleVolumeData
from tomotwin.modules.inference.boxer import SlidingWindowBoxer
import numpy as np
import numpy.lib.stride_tricks as tricks

class MyTestCase(unittest.TestCase):
    def test_something(self):

        vol = np.zeros(shape=(10,10,10))
        pos0 = 7
        pos1 = 3
        pos2 = 5
        val = 1
        vol[pos0,pos1,pos2] = val
        box_size = 3
        stride = 2

        sliding_window_strides, center_coords = SlidingWindowBoxer._calc_sliding_volumes(
            tomogram=vol,
            stride=(stride,stride,stride),
            box_size=box_size


        )
        dat = SimpleVolumeData(volumes=sliding_window_strides)

        for i in range(len(dat)):
            loc = center_coords[i]

            sub = dat[i]
            if loc[0] == pos0 and loc[1] == pos1 and loc[2] == pos2:
                self.assertEqual(sub[1, 1, 1], val)
            else:
                self.assertEqual(sub[1, 1, 1], 0)





if __name__ == '__main__':
    unittest.main()
