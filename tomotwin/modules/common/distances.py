import numpy as np
import torch
from abc import ABC, abstractmethod
import pytorch_metric_learning.distances as dist


class DistanceDoesNotExistError(Exception):
    '''Exception when distance function does not exist'''

class Distance(ABC):

    @abstractmethod
    def calc(self, x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
        '''Calculates the criterion'''

    @abstractmethod
    def calc_np(self, x_1: np.array, x_2: np.array) -> np.array:
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

    def calc(self, x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
        """
        Calculates the pairwise cosine similarty
        :param x_1: First array
        :param x_2: Second array
        :return: Pairwise Cosine similarty
        """
        return dist.SNRDistance.pairwise_distance(x_1, x_2)

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

    def calc(self, x_1: torch.Tensor, x_2: torch.Tensor) -> torch.Tensor:
        """
        Calculates the pairwise cosine similarty
        :param x_1: First array
        :param x_2: Second array
        :return: Pairwise Cosine similarty
        """
        return dist.CosineSimilarity.pairwise_distance(x_1, x_2)

    def calc_np(self, x_1: np.array, x_2: np.array) -> np.array:
        """
        Calculates the pairwise cosine similarty
        :param x_1: First array
        :param x_2: Second array
        :return: Pairwise Cosine similarty
        """
        # no need for additional normalization, as our embeddings are already l2 normalized.
        return (x_1 * x_2).sum(1)

    def name(self) -> str:
        return "COSINE"

    def is_similarity(self) -> bool:
        return True

class Euclidean(Distance):

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
        return np.arccos(product)

    def name(self) -> str:
        return "GEODESIC"

    def is_similarity(self) -> bool:
        return False