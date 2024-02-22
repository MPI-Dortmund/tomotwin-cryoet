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

import multiprocessing
from typing import Callable, List

import numpy as np
import tqdm

from tomotwin.modules.inference.mapper import Mapper


class DistanceMapper(Mapper):
    """
    Classifier that is using a distance measure for classification.
    """

    def __init__(
        self, distance_function: Callable[[np.array, np.array], float], similarty: bool=False, quiet: bool=False
    ):
        """
        :param distance_function: Distance function that takes one row of the references (first array) and
        calulates the distances to a array of embeddings (second array).
        :param threshold:
        """
        self.distance_function = distance_function
        self.is_similarty = similarty
        self.quiet = quiet



    def map_reference(self, reference: np.array, embedding_chunks: List[np.array]) -> np.array:

        def map_multiprocessing_map(embedding_chunks):
            from multiprocessing.pool import Pool
            from itertools import repeat

            with Pool() as pool:
                results_chunks = pool.starmap(self.distance_function,zip(embedding_chunks, repeat(reference)))

            return results_chunks

        results_chunks = map_multiprocessing_map(embedding_chunks)

        return np.concatenate(results_chunks)

    def map(
        self,
        embeddings: np.array,
        references: np.array,
    ) -> np.array:
        """
        Will calculate the distance between evey embedding and every reference.
        It returns a 2D array, where the columns contain the distances metric for all references for a specific embedding.
        """
        embedding_chunks = np.array_split(embeddings,multiprocessing.cpu_count())
        from multiprocessing import Manager
        import ctypes
        manager = Manager()
        shared_memory_chunks = []

        for chunks in embedding_chunks:
            k = manager.Value(ctypes.py_object, chunks)
            shared_memory_chunks.append(k.value)

        distances = np.empty(shape=(references.shape[0], embeddings.shape[0]), dtype=np.float16)
        del embeddings
        del embedding_chunks

        for ref_index in tqdm.tqdm(range(references.shape[0]), desc="Map references", disable=self.quiet):
            distances[ref_index, :] = self.map_reference(reference=references[ref_index,:], embedding_chunks=shared_memory_chunks)

        return distances
