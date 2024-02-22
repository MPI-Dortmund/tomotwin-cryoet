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

import numpy as np
import pytorch_metric_learning.distances as dist
import torch
from numpy.typing import ArrayLike


class DistanceDoesNotExistError(Exception):
    '''Exception when distance function does not exist'''


class Distance(ABC):
    '''
    Abstract base class for distance metrics
    '''

    @abstractmethod
    def calc(self, x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
        '''Calculates the criterion'''

    @abstractmethod
    def calc_np(self, x_1: ArrayLike, x_2: ArrayLike) -> np.array:
        '''Numpy version of the criterion'''

    @abstractmethod
    def name(self) -> str:
        '''returns the name of the criterion'''

    @abstractmethod
    def is_similarity(self) -> bool:
        '''
        True if it actually measures similarity/
        '''

class DistanceManager():
    '''
    Manages all distance metrics
    '''

    def __init__(self):
        self.distances = {}
        dlist = [Euclidean(), Geodesic(), CosineSimilarty(), SNRSimilarty()]
        for d in dlist:
            self.distances[d.name().upper()] = d


    def get_distance(self, identifier: str) -> Distance:
        if identifier.upper() in self.distances:
            return self.distances[identifier]
        else:
            raise DistanceDoesNotExistError()


class SNRSimilarty(Distance, dist.SNRDistance):
    '''
    Signal to Noise Ratio distance metric
    '''

    def calc(self, x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
        """
        Calculates the pairwise cosine similarty
        :param x_1: First array
        :param x_2: Second array
        :return: Pairwise Cosine similarty
        """
        snr_dist = dist.SNRDistance()
        return snr_dist.pairwise_distance(x_1, x_2)

    def calc_np(self, x_1: np.array, x_2: np.array) -> np.array:
        """
        Calculates the pairwise cosine similarty
        :param x_1: First array
        :param x_2: Second array
        :return: Pairwise Cosine similarty
        """
        return np.var(x_2-x_1,axis=1)/np.var(x_2,axis=1)

    def name(self) -> str:
        return "SNR"

    def is_similarity(self) -> bool:
        return True

class CosineSimilarty(Distance, dist.CosineSimilarity):
    '''
    Cosine similarty metric
    '''

    def calc(self, x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
        """
        Calculates the pairwise cosine similarty
        :param x_1: First array
        :param x_2: Second array
        :return: Pairwise Cosine similarty
        """
        cos_sim = dist.CosineSimilarity()
        return cos_sim.pairwise_distance(x_1, x_2)

    def calc_np(self, x_1: ArrayLike, x_2: ArrayLike) -> np.array:
        """
        Calculates the pairwise cosine similarty
        :param x_1: First array
        :param x_2: Second array
        :return: Pairwise Cosine similarty
        """
        # no need for additional normalization, as our embeddings are already l2 normalized.
        res = (x_1 * x_2).sum(1)
        return res

    def name(self) -> str:
        return "COSINE"

    def is_similarity(self) -> bool:
        return True

class Euclidean(Distance):
    '''
    Euclidian distance metric
    '''

    def calc(self, x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
        """
        Calculates the euclidian distance between two tensor
        :param x_1: First tensor
        :param x_2: Second tensor
        :return: Euclidean distance
        """
        return (x_1 - x_2).pow(2).sum(1)

    def calc_np(self, x_1: np.array, x_2: np.array) -> np.array:
        """
        Calculates the squared euclidian distance between two tensors
        :param x_1: First tensor
        :param x_2: Second tensor
        :return: Euclidean distance
        """

        return np.sum(np.square(x_1 - x_2), axis=1)

    def name(self) -> str:
        return "EUCLIDEAN"

    def is_similarity(self) -> bool:
        return False

class Geodesic(Distance):
    '''
    Geodesic distance metric
    '''

    def calc(self, x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
        """
        Calculates the geodesic distance between two sensor
        :param x_1: First tensor
        :param x_2: Second tensor
        :return: Geodesic distance
        """
        product = (x_1 * x_2).sum(1)

        return torch.acos(product)

    def calc_np(self, x_1: np.array, x_2: np.array) -> np.array:
        """
        Calculates the geodesic distance between two sensor
        :param x_1: First tensor
        :param x_2: Second tensor
        :return: Geodesic distance
        """
        product = (x_1 * x_2).sum(1)
        res = np.arccos(product)
        return res

    def name(self) -> str:
        return "GEODESIC"

    def is_similarity(self) -> bool:
        return False