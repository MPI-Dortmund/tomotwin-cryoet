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


class Mapper(ABC):
    @abstractmethod
    def map(
        self,
        embeddings: np.array,
        references: np.array,
    ) -> np.array:
        """
        Given a set of embeddings and references, it calculate the pairwise distance of each references to the embeddings
        :param embeddings: 2D array of embeddings. Each row correspond to different subvolume.
        :param references: 2D array of reference embeddings. Each row correspond to different reference.
        :return: It returns a 2D array, where the columns contain the distance metric for all references for a specific embedding.
        """