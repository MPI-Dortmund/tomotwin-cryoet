import unittest
from tomotwin.modules.inference.locator import Locator
import pandas as pd
import numpy as np

class MyTestCase(unittest.TestCase):

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
