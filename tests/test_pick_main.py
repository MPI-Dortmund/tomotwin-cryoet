import os
import tempfile
import unittest

import pandas as pd

from tomotwin.modules.common.io.coords_format import CoordsFormat
from tomotwin.modules.inference.argparse_pick_ui import PickConfiguration
from tomotwin.pick_main import write_results, InvalidLocateResults, run


class MyTestCase(unittest.TestCase):
    def test_something(self):
        with self.assertRaises(InvalidLocateResults):
            d = {"X": [], "Y": [], "Z": [], "width": []}
            df = pd.DataFrame(d)
            with tempfile.TemporaryDirectory() as tmpdirname:
                write_results(df, [CoordsFormat()], tmpdirname, "blub")

    def test_pick_main(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            locate_path = os.path.join(os.path.dirname(__file__), "../resources/tests/pick_main/located.tloc")
            conf = PickConfiguration(
                locate_results_path=locate_path,
                target_reference=["gen01_t01_2df7_089.mrc"],
                output_path=tmpdirname,
                min_metric=0.8,
                max_metric=1,
                min_size=None,
                max_size=None

            )
            run(conf)
            self.assertEqual(True, os.path.exists(os.path.join(tmpdirname, "gen01_t01_2df7_089_relion3.star")))
            self.assertEqual(True, os.path.exists(os.path.join(tmpdirname, "gen01_t01_2df7_089.coords")))


if __name__ == '__main__':
    unittest.main()
