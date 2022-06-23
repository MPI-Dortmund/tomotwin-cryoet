import unittest
from tomotwin.modules.training.tripletprovider import TripletProvider
from tomotwin.modules.training.filenametripletprovider import FilenameMatchingTripletProvider
from tomotwin.modules.training.filenamematchingtripletprovidernopdb import FilenameMatchingTripletProviderNoPDB
class MyTestCase(unittest.TestCase):

    def test_triplet_to_dataframe(self):
        volumes = ["/a/b/c/module_0_1BXN_0.mrc", "/a/b/c/module_0_1BXN_1.mrc",
                   "/a/b/c/module_0_3GL1_0.mrc"]
        pdbs = ["/x/y/z/1bxn.mrc", "/x/y/z/3gl1.mrc"]
        ftp = FilenameMatchingTripletProvider(path_pdb="", path_volume="",max_neg=5)
        triplets = ftp.generate_triplets(pdbs, volumes)
        df = TripletProvider.triplets_to_df(triplets)

        self.assertEqual(len(df), 15)
        self.assertEqual(df.isnull().values.any(), False)

    def test_triplet_to_dataframe_max_neg(self):
        volumes = ["/a/b/c/module_0_1BXN_0.mrc",
                   "/a/b/c/module_0_3GL1_0.mrc",
                   "/a/b/c/module_0_3GL1_1.mrc",
                   "/a/b/c/module_0_3GL1_2.mrc",
                   "/a/b/c/module_0_3GL1_3.mrc",
                   "/a/b/c/module_0_3GL1_4.mrc",
                   "/a/b/c/module_0_3GL1_5.mrc",
                   "/a/b/c/module_0_3GL1_6.mrc",
                   "/a/b/c/module_0_3GL1_7.mrc"]
        pdbs = ["/x/y/z/1bxn.mrc"]
        ftp = FilenameMatchingTripletProvider(path_pdb="",
                                              path_volume="",
                                              max_neg=5)
        triplets = ftp.generate_triplets(pdbs, volumes)
        df = TripletProvider.triplets_to_df(triplets)

        self.assertEqual(5,len(df))
        self.assertEqual(df.isnull().values.any(), False)

    def test_triplet_to_dataframe_NoPDB(self):
        volumes = ["/a/b/c/module_0_1BXN_0.mrc",
                   "/a/b/c/module_0_5MRC_0.mrc",
                   "/a/b/c/module_0_5MRC_1.mrc",
                   "/a/b/c/module_0_3GL1_2.mrc",
                   "/a/b/c/module_0_3GL1_3.mrc",
                   "/a/b/c/module_0_3GL1_4.mrc",
                   "/a/b/c/module_0_3GL1_5.mrc",
                   "/a/b/c/module_0_3GL1_5.mrc",]

        ftp = FilenameMatchingTripletProviderNoPDB(path_volume="")
        triplets = ftp.generate_triplets(volumes)
        df = TripletProvider.triplets_to_df(triplets)
        print(triplets)
        self.assertEqual(3,len(df))
        self.assertEqual(df.isnull().values.any(), False)



if __name__ == '__main__':
    unittest.main()
