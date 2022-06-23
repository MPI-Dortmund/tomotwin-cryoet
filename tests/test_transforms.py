import unittest
from tomotwin.modules.training.transforms import AugmentationPipeline, AddNoise, Rotate, Blur, Shift,BlockDropout
import numpy as np


class MyTestCase(unittest.TestCase):

    def test_apply_no_transofrm(self):
        aug_anchor = AugmentationPipeline(
            augs=[
                AddNoise(sigma=1),
                Blur(sigma=0.1),
                Rotate(),
                Shift()
            ],
            probs=[0.0, 0.0, 0.0, 0.0]
        )
        initial = np.random.rand(37, 37, 37)

        augmented = AugmentationPipeline.transform(aug_anchor, initial)

        np.testing.assert_array_equal(augmented, initial)

    def test_addnoise(self):
        trans = AddNoise(sigma=(1,1))
        input = np.zeros(shape=(10,10,10))
        ouput = trans(input)

        std = np.std(ouput)
        np.testing.assert_almost_equal(std,1,decimal=0.1)

    def test_blockout(self):

        trans = BlockDropout(nblocks=(1,2),blocksize=(10,11))
        input = np.zeros(shape=(40, 40, 40))+1
        output = trans(input)

        blocked = np.sum(output==0)
        np.testing.assert_equal(blocked,1000)

    def test_shift(self):

        trans = Shift(min_shift=1,max_shift=1)
        input = np.zeros(shape=(40, 40, 40))
        input[0,0,0] = 1
        output = trans(input)

        np.testing.assert_equal(input[0,0,0],output[1,0,0])

if __name__ == '__main__':
    unittest.main()
