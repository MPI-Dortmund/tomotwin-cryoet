import unittest
from tomotwin.modules.inference.argparse_pick_ui import PickArgParseUI
from unittest.mock import patch
import sys

class MyTestCase(unittest.TestCase):
    def test_required(self):
        locate_results = '/my/locate/pos.tloc'
        outpth = '/my/out/path/'

        ui = PickArgParseUI()
        ui.run(['-l', locate_results, '-o', outpth])
        conf = ui.get_pick_configuration()

        self.assertEqual(conf.locate_results_path, locate_results)
        self.assertEqual(conf.output_path, outpth)


if __name__ == '__main__':
    unittest.main()
