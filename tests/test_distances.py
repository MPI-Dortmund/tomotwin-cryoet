import unittest
from tomotwin.modules.common.distances import Geodesic
import numpy as np
class MyTestCase(unittest.TestCase):
    def test_geodesic(self):
        embeddings = np.zeros(shape=(1, 3))
        reference = np.zeros(shape=(1, 3))
        embeddings[0, 2] = 1
        reference[0, 2] = -1
        dist = Geodesic().calc_np(embeddings, reference)
        self.assertAlmostEqual(np.pi, dist[0])


if __name__ == '__main__':
    unittest.main()
