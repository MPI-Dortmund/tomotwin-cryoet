import multiprocessing
import os
import tempfile
import unittest

from tomotwin.locate_main import run
from tomotwin.modules.inference.locate_ui import LocateConfiguration, LocateMode


class MyTestCase(unittest.TestCase):
    def test_locate_main(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            map_pth = os.path.join(os.path.dirname(__file__), "../resources/tests/locate_main/map.tmap")
            conf = LocateConfiguration(processes=multiprocessing.cpu_count(),
                                output_path=tmpdirname,
                                map_path=map_pth,
                                mode=LocateMode.FINDMAX,
                                tolerance=0.2,
                                boxsize=37,
                                global_min=0.5,
                                write_heatmaps=True)
            run(conf)
            self.assertEqual(True, os.path.exists(os.path.join(tmpdirname, "located.tloc")))
            self.assertEqual(True, os.path.exists(os.path.join(tmpdirname, "gen01_t01_2df7_089.mrc.mrc")))


if __name__ == '__main__':
    unittest.main()
