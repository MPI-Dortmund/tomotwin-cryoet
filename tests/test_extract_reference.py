import tempfile
import unittest

import mrcfile
import numpy as np
import pandas as pd

from tomotwin.modules.tools.extract_reference import ExtractReference


class MyTestCase(unittest.TestCase):
    def test_extract_and_save_pos_even(self):
        zeros = np.zeros(shape=(
            10, 10, 10))
        zeros[2,2,2] = 5
        pos = {"X":[2], "Y":[2], "Z": [2]}
        with tempfile.TemporaryDirectory() as tmpdirname:
            written = ExtractReference.extract_and_save(volume=zeros, positions=pd.DataFrame(pos), box_size=4,
                                                        out_pth=tmpdirname, basename="ref")
            vol = mrcfile.mmap(written[0], permissive=True, mode='r').data

        self.assertEqual(-5, vol[2, 2, 2])

    def test_extract_and_save_pos_odd(self):
        zeros = np.zeros(shape=(10, 10, 10))
        zeros[2, 2, 2] = 5
        pos = {"X":[3], "Y":[3], "Z": [3]}
        with tempfile.TemporaryDirectory() as tmpdirname:
            written = ExtractReference.extract_and_save(volume=zeros,positions=pd.DataFrame(pos),box_size=3,out_pth=tmpdirname,basename="ref")
            vol = mrcfile.mmap(written[0], permissive=True, mode='r').data

        self.assertEqual(-5, vol[0, 0, 0])


if __name__ == '__main__':
    unittest.main()
