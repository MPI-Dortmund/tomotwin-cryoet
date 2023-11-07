import os
import tempfile
import unittest

import pandas as pd
import pytest
import torch

from tomotwin.modules.tools.umap import UmapTool


# from cuml.common.device_selection import using_device_type


class MyTestCase(unittest.TestCase):

    @pytest.mark.skipif(torch.cuda.is_available() == False, reason="Skipped because CUDA is not available")
    def test_something(self):
        tool = UmapTool()
        in_pth = os.path.join(os.path.dirname(__file__),
                              "../resources/tests/map_main/embed/tomo/tiltseries_cropped_embeddings.temb")
        from types import SimpleNamespace

        with tempfile.TemporaryDirectory() as tmpdirname:
            # with using_device_type('cpu'):
            dat = pd.read_pickle(in_pth)
            pd.to_pickle(dat[:500], os.path.join(tmpdirname, "emb.temb"))
            args = {
                "input": os.path.join(tmpdirname, "emb.temb"),
                "fit_sample_size": 500,
                "chunk_size": 500,
                "neighbors": 200,
                "ncomponents": 2,
                "output": tmpdirname,
                "model": None,
                "cosine": None,
            }

            tool.run(SimpleNamespace(**args))

            self.assertEqual(True,
                             os.path.exists(os.path.join(tmpdirname, "emb_label_mask.mrci")))  # add assertion here

            self.assertEqual(True, os.path.exists(os.path.join(tmpdirname, "emb.tumap")))  # add assertion here


if __name__ == '__main__':
    unittest.main()
