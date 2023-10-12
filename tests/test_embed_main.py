import os.path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import mrcfile
import numpy as np
import pytest
import torch

import tomotwin.embed_main
from tomotwin.modules.inference.argparse_embed_ui import EmbedConfiguration, EmbedMode, DistrMode
from tomotwin.modules.inference.embedor import Embedor
from tomotwin.modules.inference.embedor import TorchEmbedor, TorchEmbedorDistributed
from tomotwin.modules.inference.volumedata import VolumeDataset
from tomotwin.modules.networks import networkmanager, SiameseNet3D


class DummyEmbedor(Embedor):
    def __init__(self):
        self.tomotwin_config = {}

    def embed(self, volume_data: VolumeDataset) -> np.array:
        embeddings = np.random.randn(2, 32)
        return embeddings


class TestsEmbedMain(unittest.TestCase):
    def test_get_file_md5(self):
        from tomotwin.embed_main import get_file_md5

        mrcpdb = os.path.join(
            os.path.dirname(__file__), "../resources/tests/FindMaxLocator/5MRC.mrc"
        )
        md5_checksum = get_file_md5(mrcpdb)
        self.assertEqual(md5_checksum, "dc78623e5f17d5eff2f80d254d6f1b92")

    @patch(
        "tomotwin.modules.networks.networkmanager.NetworkManager.create_network",
        MagicMock(
            return_value=SiameseNet3D.SiameseNet3D(
                norm_name="GroupNorm",
                norm_kwargs={"num_channels": 1024, "num_groups": 64},
            )
        ),
    )
    def test_embed_main_real_subvol_torchembeddor(self):
        from tomotwin.embed_main import run as embed_main_func

        tomo = np.random.randn(37, 37, 37).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdirname:
            volumes = [os.path.join(tmpdirname, "vola.mrc")]
            for v in volumes:
                with mrcfile.new(v) as mrc:
                    mrc.set_data(tomo)
            embed_conf = EmbedConfiguration(
                model_path=None,
                volumes_path=volumes,
                output_path=tmpdirname,
                mode=EmbedMode.VOLUMES,
                batchsize=1,
                stride=1,
            )
            with patch(
                    "tomotwin.embed_main.make_embeddor",
                    MagicMock(return_value=TorchEmbedor(batchsize=1, weightspth=None)),
            ), patch("tomotwin.embed_main.get_window_size", MagicMock(return_value=37)):
                embed_conf.distr_mode = DistrMode.DP
                embed_main_func(None, embed_conf, None)
                networkmanager.NetworkManager.create_network.reset_mock()
            self.assertEqual(
                True, os.path.exists(os.path.join(tmpdirname, "embeddings.temb"))
            )

    @patch(
        "tomotwin.modules.networks.networkmanager.NetworkManager.create_network",
        MagicMock(
            return_value=SiameseNet3D.SiameseNet3D(
                norm_name="GroupNorm",
                norm_kwargs={"num_channels": 1024, "num_groups": 64},
            )
        ),
    )
    @pytest.mark.skipif(torch.cuda.is_available() == False, reason="Skipped because CUDA is not available")
    def test_embed_main_real_subvol_distributedtorchembeddor(self):
        import random
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29' + str(random.randint(1, 500)).zfill(3)

        from tomotwin.embed_main import run as embed_main_func

        tomo = np.random.randn(37, 37, 37).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmpdirname:
            volumes = [os.path.join(tmpdirname, "vola.mrc")]
            for v in volumes:
                with mrcfile.new(v) as mrc:
                    mrc.set_data(tomo)
            embed_conf = EmbedConfiguration(
                model_path=None,
                volumes_path=volumes,
                output_path=tmpdirname,
                mode=EmbedMode.VOLUMES,
                batchsize=1,
                stride=1,
            )
            with patch(
                    "tomotwin.embed_main.make_embeddor",
                    MagicMock(return_value=TorchEmbedorDistributed(batchsize=1, weightspth=None, world_size=1, rank=0)),
            ), patch("tomotwin.embed_main.get_window_size", MagicMock(return_value=37)):
                embed_conf.distr_mode = DistrMode.DP
                embed_main_func(rank=0, conf=embed_conf, world_size=1)
                networkmanager.NetworkManager.create_network.reset_mock()
            self.assertEqual(
                True, os.path.exists(os.path.join(tmpdirname, "embeddings.temb"))
            )

    def test_embed_main_volumes(self):
        from tomotwin.embed_main import run as embed_main_func

        tomo = np.random.randn(50, 50, 50).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdirname:
            volumes = [
                os.path.join(tmpdirname, "vola.mrc"),
                os.path.join(tmpdirname, "volb.mrc"),
            ]
            for v in volumes:
                with mrcfile.new(v) as mrc:
                    mrc.set_data(tomo)

            tomotwin.embed_main.make_embeddor = MagicMock(return_value=DummyEmbedor())
            tomotwin.embed_main.get_window_size = MagicMock(return_value=37)
            embed_conf = EmbedConfiguration(
                model_path=None,
                volumes_path=volumes,
                output_path=tmpdirname,
                mode=EmbedMode.VOLUMES,
                batchsize=3,
                stride=1,
                zrange=None,
            )
            embed_conf.distr_mode = DistrMode.DP
            embed_main_func(None, embed_conf, None)
            self.assertEqual(
                True, os.path.exists(os.path.join(tmpdirname, "embeddings.temb"))
            )

    def test_embed_main_tomo(self):
        from tomotwin.embed_main import run as embed_main_func

        tomo = np.random.randn(50, 50, 50).astype(np.float32)

        with tempfile.TemporaryDirectory() as tmpdirname:
            volume_path = os.path.join(tmpdirname, "vola.mrc")

            with mrcfile.new(volume_path) as mrc:
                mrc.set_data(tomo)
            tomotwin.embed_main.make_embeddor = MagicMock(return_value=DummyEmbedor())
            tomotwin.embed_main.get_window_size = MagicMock(return_value=37)
            embed_conf = EmbedConfiguration(
                model_path=None,
                volumes_path=volume_path,
                output_path=tmpdirname,
                mode=EmbedMode.TOMO,
                batchsize=3,
                stride=1,
                zrange=None,
            )
            embed_conf.distr_mode = DistrMode.DP
            embed_main_func(None, embed_conf, None)
            self.assertEqual(
                True, os.path.exists(os.path.join(tmpdirname, "vola_embeddings.temb"))
            )

    def test_embed_tomogram(self):
        from tomotwin.embed_main import embed_tomogram

        tomo = np.random.randn(50, 50, 50)
        with tempfile.TemporaryDirectory() as tmpdirname:
            embed_conf = EmbedConfiguration(
                model_path=None,
                volumes_path="my/fake/volume.mrc",
                output_path=tmpdirname,
                mode=None,
                batchsize=3,
                stride=1,
                zrange=None,
            )
            embed_tomogram(
                tomo=tomo, embedor=DummyEmbedor(), window_size=10, conf=embed_conf
            )
            import os

            self.assertEqual(
                True, os.path.exists(os.path.join(tmpdirname, "volume_embeddings.temb"))
            )

    def test_embed_subvolume(self):
        from tomotwin.embed_main import embed_subvolumes

        with tempfile.TemporaryDirectory() as tmpdirname:
            embed_conf = EmbedConfiguration(
                model_path=None,
                volumes_path="my/fake/volume.mrc",
                output_path=tmpdirname,
                mode=None,
                batchsize=3,
                stride=1,
                zrange=None,
            )
            paths = [
                os.path.join(tmpdirname, "vola.mrc"),
                os.path.join(tmpdirname, "volb.mrc"),
            ]

            for p in paths:
                with mrcfile.new(p) as mrc:
                    mrc.set_data(np.random.rand(50, 50).astype(np.float32))

            embed_subvolumes(paths=paths, embedor=DummyEmbedor(), conf=embed_conf)
            self.assertEqual(
                True, os.path.exists(os.path.join(tmpdirname, "embeddings.temb"))
            )


if __name__ == "__main__":
    unittest.main()
