import unittest
from tomotwin.modules.inference.distance_classifier import DistanceClassifier
from tomotwin.modules.common.distances import Euclidean
import numpy as np

class ClassifierTestCases(unittest.TestCase):
    def test_distance_classifier(self):

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
        classifier = DistanceClassifier(distance_function=distances_func,threshold=None)

        result = classifier.classify(embeddings=embeddings,references=reference_embeddings)

        self.assertEqual(0, np.argmax(result))

    def test_distance_classifier_max0(self):

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
        classifier = DistanceClassifier(distance_function=distances_func,threshold=0)

        result = classifier.classify(embeddings=embeddings,references=reference_embeddings)

        self.assertEqual(0, np.argmax(result))


if __name__ == '__main__':
    unittest.main()
