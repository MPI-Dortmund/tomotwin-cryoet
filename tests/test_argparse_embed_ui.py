import unittest
from tomotwin.modules.inference.argparse_embed_ui import EmbedArgParseUI
from unittest.mock import patch
import sys
class MyTestCase(unittest.TestCase):

    def test_parse_args_only_required(self):
        modelpth = '/my/model/path.h5'
        volpth = '/my/vol/path/vol.mrc'
        outpth = '/my/out/path/'

        ui = EmbedArgParseUI()
        testargs = ["", "tomogram"]
        with patch.object(sys, 'argv', testargs):
            ui.run(['tomogram','-m', modelpth, '-v', volpth, '-o', outpth])
        conf = ui.get_embed_configuration()

        self.assertEqual(conf.model_path, modelpth)
        self.assertEqual(conf.volumes_path, volpth)
        self.assertEqual(conf.output_path, outpth)
        self.assertEqual(conf.window_size, 37)
        self.assertEqual(conf.stride, [2, 2, 2])
