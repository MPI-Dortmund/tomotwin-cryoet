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

        self.assertEqual(7*7*7,len(boxes))

    def test_SlidingWindowBoxer_stride2(self):
        boxer = SlidingWindowBoxer(
            stride=2,
            box_size=3,
        )
        import numpy as np
        tomo = np.random.randn(9,9,9)
        boxes = boxer.box(tomogram=tomo)
        self.assertEqual(len(boxes), 4*4*4)

    def test_SlidingWindowBoxer_stride2_mask(self):

        import numpy as np
        tomo = np.random.randn(9, 9, 9)
        mask = np.zeros(shape=(9,9,9))
        mask[3:6,3:6,3:6] = 1
        mask = mask != 0

        boxer = SlidingWindowBoxer(
            stride=3,
            box_size=3,
            mask=mask
        )

        boxes = boxer.box(tomogram=tomo)
        self.assertEqual(len(boxes), 1)


if __name__ == '__main__':
    unittest.main()
