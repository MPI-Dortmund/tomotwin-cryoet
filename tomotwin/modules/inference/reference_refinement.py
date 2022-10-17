import pandas as pd
import numpy as np
from tomotwin.modules.inference.distance_mapper import DistanceMapper

class ReferenceRefiner:

    def __init__(
            self, mapper: DistanceMapper
    ):
        self.mapper = mapper

    def refine_reference(self, reference: np.array, embeddings: np.array):
        # calculate distance to embeddings
        # sample according distance
        # select the one that is minimal to all others

        distances = self.mapper.map(embeddings=embeddings, references=reference)

        print(distances.shape)

        pass

