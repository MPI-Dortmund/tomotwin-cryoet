import unittest
from tomotwin.modules.inference.argparse_map_ui import MapArgParseUI
from unittest.mock import patch
import sys

class MyTestCase(unittest.TestCase):
    def test_something(self):
        refpth = '/my/emb/emb.temb'
        volpth = '/my/emb/vol.temb'
        outpth = '/my/out/path/'

        ui = MapArgParseUI()
        cmd = "distance"
        testargs = ["", cmd]
        with patch.object(sys, 'argv', testargs):
            ui.run([cmd, '-r', refpth, '-v', volpth, '-o', outpth])
        conf = ui.get_map_configuration()

        self.assertEqual(conf.reference_embeddings_path, refpth)
        self.assertEqual(conf.output_path, outpth)
        self.assertEqual(conf.volume_embeddings_path, volpth)


if __name__ == '__main__':
    unittest.main()
