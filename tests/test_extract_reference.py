import unittest
from tomotwin.modules.tools.extract_reference import ExtractReference, EvenBoxSizeException
import numpy as np
import pandas as pd
import tempfile
import mrcfile
import os
class MyTestCase(unittest.TestCase):
    def test_extract_and_save_pos_even(self):
        zeros = np.zeros(shape=(5,5,5))
        zeros[2,2,2] = 5
        pos = {"X":[2], "Y":[2], "Z": [2]}
        with tempfile.TemporaryDirectory() as tmpdirname:
            written = ExtractReference.extract_and_save(volume=zeros,positions=pd.DataFrame(pos),box_size=3,out_pth=tmpdirname,basename="ref")
            vol = mrcfile.mmap(os.path.join(tmpdirname,"ref_0.mrc"), permissive=True, mode='r').data

        self.assertEqual(-5, vol[1,1,1])

    def test_extract_and_save_pos_odd(self):
        zeros = np.zeros(shape=(5,5,5))
        zeros[3,3,3] = 5
        pos = {"X":[3], "Y":[3], "Z": [3]}
        with tempfile.TemporaryDirectory() as tmpdirname:
            written = ExtractReference.extract_and_save(volume=zeros,positions=pd.DataFrame(pos),box_size=3,out_pth=tmpdirname,basename="ref")
            vol = mrcfile.mmap(os.path.join(tmpdirname,"ref_0.mrc"), permissive=True, mode='r').data

        self.assertEqual(-5, vol[1,1,1])

    def test_extract_and_save_even_boxsize_raise(self):
        with self.assertRaises(EvenBoxSizeException):
            zeros = np.zeros(shape=(5,5,5))
            zeros[2,2,2] = 5
            pos = {"X":[2], "Y":[2], "Z": [2]}
            with tempfile.TemporaryDirectory() as tmpdirname:
                written = ExtractReference.extract_and_save(volume=zeros,positions=pd.DataFrame(pos),box_size=2,out_pth=tmpdirname,basename="ref")


if __name__ == '__main__':
    unittest.main()
