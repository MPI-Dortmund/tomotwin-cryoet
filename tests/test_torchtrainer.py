import unittest
from tomotwin.modules.training.torchtrainer import TorchTrainer
from tomotwin.modules.common.distances import DistanceManager
from tomotwin.modules.training.filenametripletprovider import FilenameMatchingTripletProvider
from tomotwin.modules.training.tripletdataset import TripletDataset
from tomotwin.modules.common.preprocess import label_filename
from tomotwin.modules.networks.SiameseNet3D import SiameseNet3D
from pytorch_metric_learning import miners, losses
from tests.faketriplethandler import TripletFakeHandler
import numpy as np
from tomotwin.modules.training.LossPyML import LossPyML


class MyTestCase(unittest.TestCase):
    def test_init_working(self):
        model = SiameseNet3D(output_channels=128, norm_name=SiameseNet3D.NORM_BATCHNORM)
        trainer = TorchTrainer(epochs=1,
                               batchsize=2,
                               learning_rate=0.3,
                               network=model,
                               criterion=None,
                               )

        self.assertEqual(1, trainer.epochs)
        self.assertEqual(2, trainer.batchsize)
        self.assertEqual(0.3, trainer.learning_rate)

    def test_seed(self):
        model = SiameseNet3D(output_channels=128, norm_name=SiameseNet3D.NORM_BATCHNORM)

        trainer = TorchTrainer(epochs=1,
                               batchsize=2,
                               learning_rate=0.3,
                               network=model,
                               criterion=None,
                               )
        failed=False
        try:
            trainer.set_seed(1)
        except:
            failed=True
        self.assertFalse(failed)

    def test_tripletloss(self):
        import torch
        batch_size=2
        out_head=3
        x_1 = np.zeros(shape=[batch_size,out_head])
        x_2 = np.zeros(shape=[batch_size, out_head])
        x_2[0,:] = 1
        x_2[1,:] = 2
        expected_distance = np.array([np.sqrt(3)**2, (2*np.sqrt(3))**2])
        dm = DistanceManager()
        result = dm.get_distance("EUCLIDEAN").calc(torch.tensor(x_1), torch.tensor(x_2)).numpy()

        np.testing.assert_array_almost_equal(expected_distance,result)


    def test_set_train_data(self):
        # Generate dummy triplet data
        volumes = ["/a/b/c/module_0_1BXN_0.mrc",
                   "/a/b/c/module_0_3GL1_0.mrc"]
        pdbs = ["/x/y/z/1bxn.mrc"]

        ftp = FilenameMatchingTripletProvider(path_pdb="", path_volume="")

        triplets = ftp.generate_triplets(pdbs, volumes)
        network = SiameseNet3D(output_channels=128, norm_name=SiameseNet3D.NORM_BATCHNORM)
        trainer = TorchTrainer(epochs=1,
                               batchsize=1,
                               learning_rate=0.01,
                               workers=0,
                               network=network,
                               criterion=None)
        self.assertEqual(None, trainer.training_data)
        trainer.set_training_data(triplets)
        self.assertIsNotNone(trainer.training_data)

    def test_set_test_data(self):
        # Generate dummy triplet data
        volumes = ["/a/b/c/module_0_1BXN_0.mrc",
                   "/a/b/c/module_0_3GL1_0.mrc"]
        pdbs = ["/x/y/z/1bxn.mrc"]

        ftp = FilenameMatchingTripletProvider(path_pdb="", path_volume="")

        triplets = ftp.generate_triplets(pdbs, volumes)
        network = SiameseNet3D(output_channels=128, norm_name=SiameseNet3D.NORM_BATCHNORM)
        trainer = TorchTrainer(epochs=1,
                               batchsize=1,
                               learning_rate=0.01,
                               workers=0,
                               network=network,
                               criterion=None)
        self.assertEqual(None, trainer.test_data)
        trainer.set_test_data(triplets)
        self.assertIsNotNone(trainer.test_data)

    def test_train_get_best_f1_all_correct(self):
        anchor_label = "ABC"
        similarities = np.array([1,0,1,0,1,0])
        sim_labels = ["ABC", "DEF", "ABC", "DEF", "ABC", "DEF"]
        f1, _ = TorchTrainer.get_best_f1(anchor_label, similarities, sim_labels)
        self.assertEqual(1.0,f1)

    def test_train_get_best_f1_all_wrong(self):
        anchor_label = "ABC"
        similarities = np.array([1,0,1,0,1,0])
        sim_labels = ["ABC", "DEF", "ABC", "DEF", "ABC", "DEF"]
        sim_labels = sim_labels[::-1]
        f1, _ = TorchTrainer.get_best_f1(anchor_label, similarities, sim_labels)
        self.assertEqual(0.0,f1)

    def test_train_get_best_f1_lowprecision(self):
        anchor_label = "ABC"
        similarities = np.array([1,1,1,1,1,1])
        sim_labels = ["ABC", "DEF", "ABC", "DEF", "ABC", "DEF"]
        sim_labels = sim_labels[::-1]
        f1, _ = TorchTrainer.get_best_f1(anchor_label, similarities, sim_labels)
        self.assertAlmostEquals(1/1.5,f1)

    def test_train_get_best_f1_besttreshold(self):
        anchor_label = "ABC"
        similarities = np.array([1, 0.7, 1, 0.7, 1, 0.7])
        sim_labels = ["ABC", "DEF", "ABC", "DEF", "ABC", "DEF"]
        f1, best_t = TorchTrainer.get_best_f1(anchor_label, similarities, sim_labels)
        self.assertTrue(best_t>=0.7)
        self.assertAlmostEquals(1.0,f1)

    #@unittest.skip("Because it fails in CI, but not locally")
    def test_train_bs1(self):
        # Generate dummy triplet data
        volumes = ["/a/b/c/module_0_1BXN_0.mrc",
                   "/a/b/c/module_0_3GL1_0.mrc"]
        pdbs = ["/x/y/z/1bxn.mrc"]


        ftp = FilenameMatchingTripletProvider(path_pdb="", path_volume="")

        triplets = ftp.generate_triplets(pdbs, volumes)

        size=37
        tripletHandler = TripletFakeHandler(
            anchor_arr=np.zeros(shape=(size, size, size), dtype=np.float32),
            pos_arr=np.zeros(shape=(size, size, size), dtype=np.float32),
            neg_arr=np.zeros(shape=(size, size, size) , dtype=np.float32),
        )
        train_dataset = TripletDataset(training_data=triplets, handler=tripletHandler,label_ext_func=label_filename)
        model = SiameseNet3D(norm_name = "BatchNorm", output_channels =16)
        dm = DistanceManager()
        loss_func = losses.TripletMarginLoss(
            margin=0.7,
            distance=dm.get_distance(identifier="COSINE")
        )
        loss = LossPyML(
            loss_func=loss_func, miner=None
        )
        trainer = TorchTrainer(epochs=1,
                               batchsize=1,
                               learning_rate=0.01,
                               workers=1,
                               network=model,
                               criterion=loss)
        trainer.set_training_data(train_dataset)
        m = trainer.train()

        self.assertTrue(m is not None)


if __name__ == '__main__':
    unittest.main()
