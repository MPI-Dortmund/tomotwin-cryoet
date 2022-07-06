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

from typing import Callable
import multiprocessing
from multiprocessing import Pool
from functools import partial
import numpy as np
import tqdm
from tomotwin.modules.inference.classifier import Classifier


class DistanceClassifier(Classifier):
    """
    Classifier that is using a distance measure for classification.
    """

    def __init__(
        self, distance_function: Callable[[np.array, np.array], float], similarty=False
    ):
        """
        :param distance_function: Distance function that takes one row of the references (first array) and
        calulates the distances to a array of embeddings (second array).
        :param threshold:
        """
        self.distance_function = distance_function
        self.distances = None
        self.is_similarty = similarty

    def classify(
        self,
        embeddings: np.array,
        references: np.array,
    ) -> np.array:
        """
        Will calculate the distance between evey embedding and every reference. If threshold is given
        it will ignore all distances which are bigger than this threshold. Then it applies
        a software to the remaining negative distances.

        It returns a 2D array, where the columns contain the probabilities for all references for a specific embedding.
        """
        distances = np.empty(shape=(references.shape[0], embeddings.shape[0]), dtype=np.float16)
        num_cores = multiprocessing.cpu_count()
        embedding_chunks = np.array_split(embeddings,num_cores)

        for ref_index, ref in tqdm.tqdm(enumerate(references), "Calculate distances"):
            with Pool() as pool:
                results_chunks = pool.map(
                    partial(self.distance_function, x_2=ref),
                    embedding_chunks)
            result = np.concatenate(results_chunks)
            distances[ref_index, :] = result

        self.distances = distances

        return distances

    def get_distances(self) -> np.array:
        return self.distances
