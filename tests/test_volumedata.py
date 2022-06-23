import unittest
from tomotwin.modules.inference.volumedata import SlidingWindowVolumeData
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

        window_shape = (box_size, box_size, box_size)
        sliding_window_views = tricks.sliding_window_view(
            vol, window_shape=window_shape
        )

        sliding_window_strides = sliding_window_views[
                                 :: stride, :: stride, :: stride
                                 ]
        dat = SlidingWindowVolumeData(volumes=sliding_window_strides, stride=stride, boxsize=box_size)

        for i in range(len(dat)):
            loc = dat.get_localization(i)
            sub = dat[i]
            if loc[0] == pos0 and loc[1] == pos1 and loc[2] == pos2:
                self.assertEqual(sub[1, 1, 1], val)
            else:
                self.assertEqual(sub[1, 1, 1], 0)





if __name__ == '__main__':
    unittest.main()
