"""
Copyright (c) 2022 MPI-Dortmund
SPDX-License-Identifier: MPL-2.0

This file is subject to the terms of the Mozilla Public License, Version 2.0 (MPL-2.0).
The full text of the MPL-2.0 can be found at http://mozilla.org/MPL/2.0/.

For files that are Incompatible With Secondary Licenses, as defined under the MPL-2.0,
additional notices are required. Refer to the MPL-2.0 license for more details on your
obligations and rights under this license and for instructions on how secondary licenses
may affect the distribution and modification of this software.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np
import scipy.ndimage
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
        '''
        Applies an augmentation pipeline to a volume
        :return: Augmented subvolume
        '''
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
    '''
    Replaces a random amount of voxels with the volume mean
    '''

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
    '''
    Sets blocks of random size to 0
    '''

    def __init__(self, blocksize : Tuple[int,int], nblocks : Tuple[int,int]):
        self.blocksize = blocksize
        self.nblocks = nblocks

    def __call__(self, volume: np.array):

        rand_nblock = np.random.randint(self.nblocks[0],self.nblocks[0]+1)

        for _ in range(rand_nblock):
            rand_blocksize = np.random.randint(self.blocksize[0],self.blocksize[1]+1)

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
    '''
    Sets blocks of random size to 0
    '''
    def __init__(self, sigma : Tuple[float,float]):
        self.sigma = sigma

    def __call__(self, volume: NDArray):
        noise = np.random.randn(*volume.shape) * np.random.uniform(*self.sigma)
        noisy_volume = np.add(volume, noise)
        return noisy_volume.copy()

    def __str__(self):
        return f"AddNoise (Sigma: {self.sigma})"

class AddMultiplicativeNoise(Augmentation):
    '''
    Adds multiplicative noise to a subvolume
    '''

    def __init__(self, sigma : Tuple[float,float]):
        self.sigma = sigma

    def __call__(self, volume: NDArray):
        noise = 1 + np.random.randn(*volume.shape) * np.random.uniform(*self.sigma)
        noisy_volume = np.multiply(volume, noise)
        return noisy_volume.copy()
    def __str__(self):
        return f"AddMultiplicativeNoise (Sigma: {self.sigma})"

class Blur(Augmentation):
    '''
    Adds blurring to a subvolume
    '''

    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def __call__(self, volume: NDArray):
        blurred = scipy.ndimage.gaussian_filter(volume, self.sigma)
        return blurred.copy()

    def __str__(self):
        return f"Blur (Sigma: {self.sigma})"


class Rotate(Augmentation):
    '''
    Does 90 degree rotations around specified axis
    '''

    def __init__(self, axes=(0, 1)):
        self.axes = axes

    def __call__(self, volume: NDArray):
        num_rotations = np.random.randint(0,4)
        rotated = np.rot90(volume, k=num_rotations, axes=self.axes)
        return rotated.copy()
    def __str__(self):
        return f"Rotate (Axes: {self.axes})"

class RotateFull(Augmentation):
    '''
    Does custrom rotations around the specified axis
    '''

    def __init__(self, axes=(0, 1)):
        self.axes = axes

    def __call__(self, volume: NDArray):

        angle = np.random.rand()*360
        rotated = scipy.ndimage.rotate(volume, angle=angle, axes=self.axes, reshape=False, mode='reflect')
        return rotated.copy()
    def __str__(self):
        return f"RotateFull (Axes: {self.axes})"

class Shift(Augmentation):
    '''
    Shifts the image by a random amount along the specified axis
    '''

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
