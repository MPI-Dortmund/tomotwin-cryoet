import unittest
from tomotwin.modules.inference.boxer import SlidingWindowBoxer

class MyTestCase(unittest.TestCase):
    def test_SlidingWindowBoxer_stride1(self):
        boxer = SlidingWindowBoxer(
            stride=1,
            box_size=3,
        )
        import numpy as np
        tomo = np.random.randn(9,9,9)
        boxes = boxer.box(tomogram=tomo)

        self.assertEqual(boxes.volumes.shape[0]*boxes.volumes.shape[1]*boxes.volumes.shape[2], 7*7*7)

    def test_SlidingWindowBoxer_stride2(self):
        boxer = SlidingWindowBoxer(
            stride=2,
            box_size=3,
        )
        import numpy as np
        tomo = np.random.randn(9,9,9)
        boxes = boxer.box(tomogram=tomo)
        self.assertEqual(boxes.volumes.shape[0]*boxes.volumes.shape[1]*boxes.volumes.shape[2], 4*4*4)


if __name__ == '__main__':
    unittest.main()
