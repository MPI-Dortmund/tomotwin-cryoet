import os.path
import unittest
from tomotwin.modules.inference.embedor import Embedor
import numpy as np
from tomotwin.modules.inference.argparse_embed_ui import EmbedConfiguration
import tempfile
from tomotwin.modules.inference.volumedata import VolumeDataset

class DummyEmbedor(Embedor):

    def __init__(self):
        self.tomotwin_config={}

    def embed(self, volume_data: VolumeDataset) -> np.array:
        embeddings = np.random.randn(2,32)
        return embeddings

class TestsEmbedMain(unittest.TestCase):

    def test_embed_tomogram(self):
        from tomotwin.embed_main import embed_tomogram
        tomo = np.random.randn(50,50,50)
        with tempfile.TemporaryDirectory() as tmpdirname:
            embed_conf = EmbedConfiguration(
                model_path=None,
                volumes_path="my/fake/volume.mrc",
                output_path=tmpdirname,
                mode=None,
                batchsize=3,
                stride=1,
                zrange=None
            )
            embed_tomogram(
                tomo=tomo,
                embedor=DummyEmbedor(),
                window_size=10,
                conf=embed_conf
            )
            import os

            self.assertEqual(True, os.path.exists(os.path.join(tmpdirname,"volume_embeddings.temb")))




if __name__ == '__main__':
    unittest.main()
