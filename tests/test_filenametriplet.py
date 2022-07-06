import unittest
from tomotwin.modules.training.filenametripletprovider import FilenameMatchingTripletProvider
import os

class MyTestCase(unittest.TestCase):

    def test_generate_triplets_correct_number(self):

        volumes = ["/a/b/c/module_0_1BXN_0.mrc","/a/b/c/module_0_1BXN_1.mrc","/a/b/c/module_0_3GL1_0.mrc"]
        pdbs = ["/x/y/z/1bxn.mrc","/x/y/z/3gl1.mrc"]
        ftp = FilenameMatchingTripletProvider(path_pdb="",path_volume="", max_neg=5)
        triplets = ftp.generate_triplets(pdbs,volumes)
        for t in triplets:
            print(t.anchor,t.positive,t.negative)

        self.assertEqual(len(triplets),15)

    def test_generate_triplets_correct_number_relpath(self):

        volumes = ["a/b/c/module_0_1BXN_0.mrc","a/b/c/module_0_1BXN_1.mrc","a/b/c/module_0_3GL1_0.mrc"]
        pdbs = ["x/y/z/1bxn.mrc","x/y/z/3gl1.mrc"]
        ftp = FilenameMatchingTripletProvider(path_pdb="",path_volume="", max_neg=5)
        triplets = ftp.generate_triplets(pdbs,volumes)

        self.assertEqual(len(triplets),15)

    def test_generate_triplets_get_triplets(self):
        pathpdb = os.path.join(os.path.dirname(__file__), "../resources/tests/tripletgenerator/pdb/")
        pathvolumes = os.path.join(os.path.dirname(__file__),
                               "../resources/tests/tripletgenerator/volumes/")
        ftp = FilenameMatchingTripletProvider(path_pdb=pathpdb, path_volume=pathvolumes, max_neg=5)
        triplets = ftp.get_triplets()

        self.assertEqual(15, len(triplets))
if __name__ == '__main__':
    unittest.main()
