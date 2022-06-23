"""
MIT License

Copyright (c) 2021 MPI-Dortmund

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import scipy.ndimage
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple
from numpy.typing import NDArray


class Augmentation(ABC):
    @abstractmethod
    def __call__(self, volume: NDArray) -> NDArray:
        pass

    @abstractmethod
    def __str__(self):
        pass


class AugmentationPipeline:
    def __init__(self, augs: List[Augmentation], probs: List[float] = None):
        self.augs = augs
        self.probs = probs
        if probs is not None:
            if len(probs) != len(self.augs):
                raise RuntimeError(
                    "Probabilities must have the same length as the number of augmentations"
                )

    def __str__(self):
        for aug_index, aug in enumerate(self.augs):
            postfix = ""
            if self.probs is not None:
                postfix = f" (P {self.probs[aug_index]})"
            return str(aug) + postfix

    def transform(self, volume: NDArray) -> NDArray:
        for aug_index, aug in enumerate(self.augs):
            if self.probs is None:
                volume = aug(volume)
            else:
                if np.random.rand() < self.probs[aug_index]:
                    volume = aug(volume)
                    if volume is None:
                        print("NONE AUG:", aug)

        return volume

class VoxelDropout(Augmentation):

    def __init__(self, ratio : Tuple[float,float]):
        self.ratio = ratio

    def __call__(self, volume: NDArray):
        if isinstance(self.ratio, float):
            rand_ratio = self.ratio
        else:
            rand_ratio = self.ratio[0] + np.random.rand() * (self.ratio[1] - self.ratio[0])
        mean_val = 0 #np.mean(volume)
        drop = np.random.binomial(
            n=1, p=1 - rand_ratio, size=(volume.shape[0], volume.shape[1],volume.shape[2])
        )
        volume[drop == 0] = mean_val
        return volume
    def __str__(self):
        return f"Voxeldropout (Ratio {self.ratio})"

class BlockDropout(Augmentation):

    def __init__(self, blocksize : Tuple[int,int], nblocks : Tuple[int,int]):
        self.blocksize = blocksize
        self.nblocks = nblocks

    def __call__(self, volume: np.array):

        rand_nblock = np.random.randint(self.nblocks[0],self.nblocks[0]+1)

        for _ in range(rand_nblock):
            rand_blocksize = np.random.randint(self.blocksize[0],self.blocksize[0]+1)

            pos0 = np.random.randint(rand_blocksize, volume.shape[0] - rand_blocksize)
            pos1 = np.random.randint(rand_blocksize, volume.shape[1] - rand_blocksize)
            pos2 = np.random.randint(rand_blocksize, volume.shape[2] - rand_blocksize)
            volume[
            int(pos0 - rand_blocksize//2):int(pos0 + rand_blocksize//2),
            int(pos1 - rand_blocksize//2):int(pos1 + rand_blocksize//2),
            int(pos2 - rand_blocksize // 2):int(pos2 + rand_blocksize // 2),
            ] = 0
        return volume.copy()

    def __str__(self):
        return f"BlockDropout (blocksize: {self.blocksize}, nblocks: {self.nblocks})"

class AddNoise(Augmentation):
    def __init__(self, sigma : Tuple[float,float]):
        self.sigma = sigma

    def __call__(self, volume: NDArray):
        noise = np.random.randn(*volume.shape) * np.random.uniform(*self.sigma)
        noisy_volume = np.add(volume, noise)
        return noisy_volume.copy()

    def __str__(self):
        return f"AddNoise (Sigma: {self.sigma})"

class AddMultiplicativeNoise(Augmentation):
    def __init__(self, sigma : Tuple[float,float]):
        self.sigma = sigma

    def __call__(self, volume: NDArray):
        noise = 1 + np.random.randn(*volume.shape) * np.random.uniform(*self.sigma)
        noisy_volume = np.multiply(volume, noise)
        return noisy_volume.copy()
    def __str__(self):
        return f"AddMultiplicativeNoise (Sigma: {self.sigma})"

class Blur(Augmentation):
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, volume: NDArray):
        blurred = scipy.ndimage.gaussian_filter(volume, self.sigma)
        return blurred.copy()

    def __str__(self):
        return f"Blur (Sigma: {self.sigma})"


class Rotate(Augmentation):
    def __init__(self, axes=(0, 1)):
        self.axes = axes

    def __call__(self, volume: NDArray):
        num_rotations = np.random.randint(0,4)
        rotated = np.rot90(volume, k=num_rotations, axes=self.axes)
        return rotated.copy()
    def __str__(self):
        return f"Rotate (Axes: {self.axes})"

class RotateFull(Augmentation):
    def __init__(self, axes=(0, 1)):
        self.axes = axes

    def __call__(self, volume: NDArray):

        angle = np.random.rand()*360
        rotated = scipy.ndimage.rotate(volume, angle=angle, axes=self.axes, reshape=False, mode='reflect')
        return rotated.copy()
    def __str__(self):
        return f"RotateFull (Axes: {self.axes})"

class Shift(Augmentation):
    def __init__(self, axis=0, min_shift=0, max_shift=4):
        self.axis = axis
        self.min_shift = min_shift
        self.max_shift = max_shift
        self._rnd_shift = None

    def __call__(self, volume: NDArray):
        x = np.random.randint(
            self.min_shift, self.max_shift + 1
        )  # random integer between 0 and 4
        self._rnd_shift = x
        shifted = np.roll(volume, x, axis=self.axis)
        return shifted.copy()
    def __str__(self):
        return f"Shift (Axis: {self.axis}, MinShift: {self.min_shift}, MaxShift: {self.max_shift})"


class Transpose(Augmentation):
    # could be replaced by the rotate class
    def __init__(self):
        self.rot = Rotate(axes=(0, 2))

    def __call__(self, volume: NDArray):
        transposed = self.rot(volume)
        return transposed.copy()
    def __str__(self):
        return f"Transpose"
