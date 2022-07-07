import unittest
from tomotwin.modules.inference.argparse_locate_ui import LocateArgParseUI
from unittest.mock import patch
import sys

class MyTestCase(unittest.TestCase):
    def test_parse_args_only_required(self):
        mappth = '/my/map/map.tmap'
        outpth = '/my/out/path/'

        ui = LocateArgParseUI()
        cmd = "findmax"
        testargs = ["", cmd]
        with patch.object(sys, 'argv', testargs):
            ui.run([cmd, '-m', mappth, '-o', outpth])
        conf = ui.get_locate_configuration()

        self.assertEqual(conf.map_path, mappth)
        self.assertEqual(conf.output_path, outpth)
        self.assertEqual(conf.tolerance, 0.2)
        self.assertEqual(conf.boxsize, 37)


if __name__ == '__main__':
    unittest.main()
