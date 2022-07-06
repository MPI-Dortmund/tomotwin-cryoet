import unittest
from tomotwin.modules.inference.distance_mapper import DistanceMapper
from tomotwin.modules.common.distances import Euclidean
import numpy as np

class ClassifierTestCases(unittest.TestCase):
    def test_distance_mapper(self):
        '''
        distance to first reference should be 0
        '''
        # Generate test data
        size_embedding = 30
        num_embeddings = 1
        num_reference_embeddings = 4
        embeddings = np.zeros(shape=(num_embeddings,size_embedding))
        reference_embeddings = np.zeros(shape=(num_reference_embeddings,size_embedding))

        embeddings[0, 5] = 1.0
        reference_embeddings[0, 5] = 1.0 # Same as embedding

        for i in range(1,num_reference_embeddings):
            reference_embeddings[i,:] = np.random.randn(size_embedding)
            reference_embeddings[i,:] = reference_embeddings[i,:]/np.linalg.norm(reference_embeddings[i,:])



        # Setup classifer
        distances_func = Euclidean().calc_np
        classifier = DistanceMapper(distance_function=distances_func)

        result = classifier.map(embeddings=embeddings, references=reference_embeddings)

        self.assertEqual(0, result[0,0])

    def test_distance_mapper_max0(self):
        '''
        distance should be zero
        '''
        # Generate test data
        size_embedding = 30
        num_embeddings = 1
        num_reference_embeddings = 4
        embeddings = np.zeros(shape=(num_embeddings,size_embedding))
        reference_embeddings = np.zeros(shape=(num_reference_embeddings,size_embedding))

        embeddings[0, 5] = 1.0
        reference_embeddings[0, 5] = 1.0 # Same as embedding

        for i in range(1,num_reference_embeddings):
            reference_embeddings[i,:] = np.random.randn(size_embedding)
            reference_embeddings[i,:] = reference_embeddings[i,:]/np.linalg.norm(reference_embeddings[i,:])



        # Setup classifer

        distances_func = Euclidean().calc_np
        classifier = DistanceMapper(distance_function=distances_func)

        result = classifier.map(embeddings=embeddings, references=reference_embeddings)
        self.assertEqual(0, result[0,0])


if __name__ == '__main__':
    unittest.main()
