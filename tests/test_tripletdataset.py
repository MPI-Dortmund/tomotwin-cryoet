import unittest
from tomotwin.modules.training.tripletdataset import TripletDataset
from tomotwin.modules.training.filenametripletprovider import FilenameMatchingTripletProvider
from tomotwin.modules.common.preprocess import label_filename

from tests.faketriplethandler import TripletFakeHandler
import numpy as np

class MyTestCase(unittest.TestCase):

    def test_init_dataset_not_none(self):

        # Generate dummy triplet data
        volumes = ["/a/b/c/module_0_1BXN_0.mrc", "/a/b/c/module_0_1BXN_1.mrc",
                   "/a/b/c/module_0_3GL1_0.mrc"]
        pdbs = ["/x/y/z/1bxn.mrc", "/x/y/z/3gl1.mrc"]



        ftp = FilenameMatchingTripletProvider(path_pdb="", path_volume="")

        triplets = ftp.generate_triplets(pdbs, volumes)

        dataset = TripletDataset(triplets, handler=None,label_ext_func=label_filename)

        self.assertTrue(dataset.training_data is not None)

    def test_init_dataset_get_item(self):

        # Generate dummy triplet data
        volumes = ["/a/b/c/module_0_1BXN_0.mrc", "/a/b/c/module_0_1BXN_1.mrc",
                   "/a/b/c/module_0_3GL1_0.mrc"]
        pdbs = ["/x/y/z/1bxn.mrc", "/x/y/z/3gl1.mrc"]


        tripletHandler = TripletFakeHandler(
            anchor_arr=np.zeros(shape=(37,37)),
            pos_arr=np.zeros(shape=(37,37)),
            neg_arr=np.zeros(shape=(37,37)),
        )
        ftp = FilenameMatchingTripletProvider(path_pdb="", path_volume="")

        triplets = ftp.generate_triplets(pdbs, volumes)

        from tomotwin.modules.common.preprocess import label_filename
        dataset = TripletDataset(triplets, tripletHandler, label_ext_func=label_filename)
        item = dataset.__getitem__(0)

        self.assertTrue(item is not None)
        self.assertTrue(item["anchor"] is not None)
        self.assertTrue(item["positive"] is not None)
        self.assertTrue(item["negative"] is not None)




if __name__ == '__main__':
    unittest.main()
