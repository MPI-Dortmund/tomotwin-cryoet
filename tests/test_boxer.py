import unittest

from tomotwin.modules.inference.boxer import SlidingWindowBoxer, CoordsBoxer


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

    def test_CoordsBoxer_with_mask(self):
        from numpy import array, zeros, random
        mask = zeros((9, 9, 9), dtype=bool)
        mask[3:6, 3:6, 3:6] = True
        coords = array([[4, 4, 4]])
        tomo = random.randn(9, 9, 9)

        boxer = CoordsBoxer(coordspth=None, box_size=3, mask=mask)
        boxer.coords = coords  # Adding the coordinates manually
        boxes = boxer.box(tomogram=tomo)

        self.assertEqual(len(boxes), 1)

    def test_CoordsBoxer_without_mask(self):
        from numpy import array, random
        coords = array([[2, 2, 2], [5, 5, 5]])
        tomo = random.randn(9, 9, 9)

        boxer = CoordsBoxer(coordspth=None, box_size=3, mask=None)
        boxer.coords = coords  # Adding the coordinates manually
        boxes = boxer.box(tomogram=tomo)

        self.assertEqual(len(boxes), 2)


if __name__ == '__main__':
    unittest.main()
