import unittest
from tomotwin.modules.inference import naivelocator
from tomotwin.modules.inference.locator import Locator
from tomotwin.modules.inference.FindMaximaLocator import FindMaximaLocator
from tomotwin.modules.common.findmax.findmax import find_maxima
import pandas as pd
import numpy as np
import tomotwin.locate_main as lm

class MyTestCase(unittest.TestCase):
    def test_naivelocator(self):

        dict = {"predicted_class": [0,0,0,0,1,1,1,1,2,2,2,2]}
        "predicted_prob"
        d = pd.DataFrame(dict)
        nloc = naivelocator.NaiveLocator()
        located_dfs = nloc.locate(d)

        np.testing.assert_almost_equal(located_dfs[0].to_numpy().flatten(), [0, 0, 0, 0])
        np.testing.assert_almost_equal(located_dfs[1].to_numpy().flatten(), [1, 1, 1, 1])
        np.testing.assert_almost_equal(located_dfs[2].to_numpy().flatten(), [2, 2, 2, 2])

    def test_naivelocator_probthresh(self):

        dict = {
            "predicted_class": [0,0,0,0,1,1,1,1,2,2,2,2],
            "predicted_prob": [1,1,1,0.5,1,1,1,0.5,1,1,1,0.5]
        }

        d = pd.DataFrame(dict)
        nloc = naivelocator.NaiveLocator(pthresh=0.6)
        located_dfs = nloc.locate(d)
        np.testing.assert_almost_equal(located_dfs[0]['predicted_class'].to_numpy().flatten(), [0, 0, 0])
        np.testing.assert_almost_equal(located_dfs[1]['predicted_class'].to_numpy().flatten(), [1, 1, 1])
        np.testing.assert_almost_equal(located_dfs[2]['predicted_class'].to_numpy().flatten(), [2, 2, 2])

    def test_locater_nms(self):
        dict = {
            "X": [100, 100, 100, 300, 300, 300],
            "Y": [100, 100, 100, 300, 300, 300],
            "Z": [100, 100, 100, 300, 300, 300],
            "predicted_class": [1,1,1,1,1,1],
            "metric_best": [0.1, 0.2, 0.3, 0.3, 0.2, 0.1],
        }
        d = pd.DataFrame(dict)
        r = Locator.nms(d,37)

        self.assertEqual(len(r),2)

    def test_locater_nms_closest(self):
        dict = {
            "X": [100, 100, 100, 300, 300, 300],
            "Y": [100, 100, 100, 300, 300, 300],
            "Z": [100, 100, 100, 300, 300, 300],
            "predicted_class": [1, 1, 1, 1, 1, 1],
            "d_class_1": [0.1, 0.2, 0.3, 0.3, 0.2, 0.1],
            "metric_best": [0.1, 0.2, 0.3, 0.3, 0.2, 0.1],
        }
        d = pd.DataFrame(dict)
        r = Locator.nms(d,37)
        r = r["d_class_1"].to_numpy()
        s = np.sum(r==0.1)
        self.assertEqual(s,2)

    def test_locator_iou(self):
        a = np.zeros(shape=[1, 6])
        b = np.zeros(shape=[1, 6])

        sizea=37
        a[0, 0] = 100
        a[0, 1] = 100
        a[0, 2] = 100
        a[0, 3:7] = sizea

        sizeb = 37
        b[0, 0] = 100
        b[0, 1] = 100
        b[0, 2] = 100
        b[0, 3:7] = sizeb

        iou = Locator._bbox_iou_vec_3d(a,b)
        self.assertEqual(1.0,iou)

if __name__ == '__main__':
    unittest.main()
