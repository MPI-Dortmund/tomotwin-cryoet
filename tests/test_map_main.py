import os
import tempfile
import unittest

from tomotwin.map_main import run
from tomotwin.modules.inference.map_ui import MapMode, MapConfiguration


class MyTestCase(unittest.TestCase):
    def test_map_main(self):
        # map_pth = os.path.join(os.path.dirname(__file__), "../resources/tests/locate_main/map.tmap")
        with tempfile.TemporaryDirectory() as tmpdirname:
            ref_emb_pth = os.path.join(os.path.dirname(__file__),
                                       "../resources/tests/map_main/embed/refs/embeddings.temb")
            vol_emb_pth = os.path.join(os.path.dirname(__file__),
                                       "../resources/tests/map_main/embed/tomo/tiltseries_cropped_embeddings.temb")
            conf = MapConfiguration(
                reference_embeddings_path=ref_emb_pth,
                volume_embeddings_path=vol_emb_pth,
                output_path=tmpdirname,
                mode=MapMode.DISTANCE,
                skip_refinement=True
            )
            run(conf)
            self.assertEqual(True, os.path.exists(os.path.join(tmpdirname, "map.tmap")))


if __name__ == '__main__':
    unittest.main()
