import os
import unittest

import pandas as pd

from tomotwin.modules.common.io.coords_format import CoordsFormat


class TestCoordsFormat(unittest.TestCase):

    def setUp(self):
        self.test_file_path = "test_coords.txt"
        self.test_write_path = "test_output_coords.txt"
        self.test_df = pd.DataFrame({"X": [1.0, 4.0, 7.0], "Y": [2.0, 5.0, 8.0], "Z": [3.0, 6.0, 9.0]})
        with open(self.test_file_path, "w") as file:
            file.write("1.0 2.0 3.0\n4.0 5.0 6.0\n7.0 8.0 9.0")

    def tearDown(self):
        if os.path.exists(self.test_file_path):
            os.remove(self.test_file_path)
        if os.path.exists(self.test_write_path):
            os.remove(self.test_write_path)

    def test_read_valid_file(self):
        result = CoordsFormat.read(self.test_file_path)
        expected_df = pd.DataFrame({"X": [1.0, 4.0, 7.0], "Y": [2.0, 5.0, 8.0], "Z": [3.0, 6.0, 9.0]})
        pd.testing.assert_frame_equal(result, expected_df)

    def test_read_nonexistent_file(self):
        with self.assertRaises(FileNotFoundError):
            CoordsFormat.read("nonexistent_file.txt")

    def test_read_invalid_format_file(self):
        with open("invalid_file.txt", "w") as file:
            file.write("1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0")
        with self.assertRaises(ValueError):
            CoordsFormat.read("invalid_file.txt")
        os.remove("invalid_file.txt")

    def test_write_valid_file(self):
        CoordsFormat().write(self.test_df, self.test_write_path)
        with open(self.test_write_path, "r") as file:
            content = file.read()
        self.assertEqual(content.strip(), "1.0 2.0 3.0\n4.0 5.0 6.0\n7.0 8.0 9.0")

    def test_write_empty_dataframe(self):
        empty_df = pd.DataFrame(columns=["X", "Y", "Z"])
        CoordsFormat().write(empty_df, self.test_write_path)
        with open(self.test_write_path, "r") as file:
            content = file.read()
        self.assertEqual(content.strip(), "")
