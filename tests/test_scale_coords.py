import unittest
import tempfile
import pandas as pd
import os
import argparse
from tomotwin.modules.tools.scale_coordinates import ScaleCoordinates
from tomotwin.pick_main import write_coords


class MyTestCase(unittest.TestCase):
    def test_no_scaling(self):
        with tempfile.TemporaryDirectory() as tmpdirname:

            # Create Sample data
            coords = {"X": [5], "Y": [8], "Z": [10]}
            df = pd.DataFrame(coords)
            filepath = os.path.join(tmpdirname,"coords.coords")
            write_coords(df,filepath)

            # Setup parser
            tool = ScaleCoordinates()
            parser = argparse.ArgumentParser()
            subparsers = parser.add_subparsers()
            tool.create_parser(subparsers)
            out_filename = os.path.join(tmpdirname, "coords_scaled.coords")

            #Run it
            args = parser.parse_args(["scale_coordinates", "--coords", filepath, "--tomotwin_pixel_size", "1.0", "--extraction_pixel_size", "1.0", "--out", out_filename])
            tool.run(args)

            #Check results
            coords_df = pd.read_csv(out_filename, sep=' ', header=None)


        self.assertEqual(5, coords_df.iloc[0, 0])
        self.assertEqual(8, coords_df.iloc[0, 1])
        self.assertEqual(10, coords_df.iloc[0, 2])

    def test_scale_factor_2(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Create Sample data
            coords = {"X": [5], "Y": [10], "Z": [15]}
            df = pd.DataFrame(coords)
            filepath = os.path.join(tmpdirname, "coords.coords")
            write_coords(df, filepath)

            # Setup parser
            tool = ScaleCoordinates()
            parser = argparse.ArgumentParser()
            subparsers = parser.add_subparsers()
            tool.create_parser(subparsers)
            out_filename = os.path.join(tmpdirname, "coords_scaled.coords")

            # Run it
            args = parser.parse_args(["scale_coordinates", "--coords", filepath, "--tomotwin_pixel_size", "10.0",
                                      "--extraction_pixel_size", "5.0", "--out", out_filename])
            tool.run(args)

            # Check results
            coords_df = pd.read_csv(out_filename, sep=' ', header=None)

        self.assertEqual(10, coords_df.iloc[0, 0])
        self.assertEqual(20, coords_df.iloc[0, 1])
        self.assertEqual(30, coords_df.iloc[0, 2])


if __name__ == '__main__':
    unittest.main()
