import unittest

import numpy as np

from tomotwin.modules.inference.boxer import SlidingWindowBoxer
from tomotwin.modules.inference.volumedata import SimpleVolumeData


class MyTestCase(unittest.TestCase):
    def test_SimpleVolumeData(self):

        vol = np.zeros(shape=(10,10,10))
        pos0 = 7
        pos1 = 3
        pos2 = 5
        val = 1
        vol[pos0,pos1,pos2] = val
        box_size = 3
        stride = 2

        roi = SlidingWindowBoxer._calc_volume_roi(vol, stride=(stride, stride, stride), box_size=box_size)
        dat = SimpleVolumeData(volumes=vol, roi=roi)

        for i in range(len(dat)):
            loc = roi.center_coords[i]

            sub = dat[i]
            print(sub.shape)
            if loc[0] == pos0 and loc[1] == pos1 and loc[2] == pos2:
                self.assertEqual(sub[1, 1, 1], val)
            else:
                print(sub[1, 1, 1])
                self.assertEqual(sub[1, 1, 1], 0)





if __name__ == '__main__':
    unittest.main()
