import unittest
from tomotwin.pick_main import write_results, InvalidLocateResults
from tomotwin.modules.common.io.coords_format import CoordsFormat
import pandas as pd
import tempfile


class MyTestCase(unittest.TestCase):
    def test_something(self):
        with self.assertRaises(InvalidLocateResults):
            d = {"X":[],"Y":[],"Z":[], "width": []}
            df = pd.DataFrame(d)
            with tempfile.TemporaryDirectory() as tmpdirname:
                write_results(df,[CoordsFormat()], tmpdirname,"blub")


if __name__ == '__main__':
    unittest.main()
