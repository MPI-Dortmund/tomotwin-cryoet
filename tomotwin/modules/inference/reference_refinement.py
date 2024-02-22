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

import numpy as np
import tqdm

from tomotwin.modules.inference.distance_mapper import DistanceMapper


class ReferenceRefiner:

    def __init__(
            self,
            mapper: DistanceMapper,
            sample_size: int = 500
    ):
        self.mapper = mapper
        self.sample_size = sample_size

    def sample_embeddings_indicis(self, distances: np.array, N: int=500) -> np.array:
        distances = distances.squeeze()
        return np.argsort(distances)[-N:]

    def sim_matrix(self,embeddings: np.array) -> np.array:
        """
        Calculates the pairwise similarity matrix
        """
        mat = np.zeros(shape=[embeddings.shape[0],embeddings.shape[0]])
        for i in range(embeddings.shape[0]):
            for j in range(i+1,embeddings.shape[0]):
                sim = self.mapper.distance_function(np.atleast_2d(embeddings[i,:]),np.atleast_2d(embeddings[j,:]))
                mat[i,j] = sim
                mat[j,i] = sim

        return mat

    def refine_reference(self, reference: np.array, embeddings: np.array) -> np.array:
        """
        # Refines a single reference in 3 steps
        # 1. calculate distance to embeddings
        # 2. take the N closest embeddings
        # 3. select the least minimal average distance to all others
        """

        distances = self.mapper.map(embeddings=embeddings, references=reference)
        embeddings_indicis = self.sample_embeddings_indicis(distances=distances,N=self.sample_size)
        embeddings_subset = embeddings[embeddings_indicis]
        mat = self.sim_matrix(embeddings_subset)
        mat[mat == 0] = np.nan
        avgs = np.nanmean(mat,axis=0)
        refined_reference = embeddings_subset[np.argmax(avgs)]

        return np.atleast_2d(refined_reference)

    def refine_references(self, references: np.array, embeddings: np.array, iterations: int = 5) -> np.array:
        '''
        Refines multiple references at once
        '''

        refined = np.copy(references)
        converged = [False]*len(refined)
        for _ in tqdm.tqdm(range(iterations),desc="Medoid reference refinement"):
            for ref_i, ref in enumerate(refined):
                if not converged[ref_i]:
                    refined_i = self.refine_reference(np.atleast_2d(ref), embeddings)
                    old_dist_ref = self.mapper.distance_function(np.atleast_2d(refined_i), np.atleast_2d(refined[ref_i, :]))
                    if np.abs(old_dist_ref-1.0)<0.0001:
                        converged[ref_i] = True
                    refined[ref_i, :] = refined_i
        return refined

