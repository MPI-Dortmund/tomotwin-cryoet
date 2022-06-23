import unittest
from tomotwin.modules.training.filenametripletprovider import FilenameMatchingTripletProvider, FilePathTriplet
from tomotwin.train_main import train_test_split_anchor_positive
from tomotwin.train_main import get_augmentations, get_loss_func

class MyTestCase(unittest.TestCase):
    def test_train_test_split_anchor_positive(self):
        volumes = ["/a/b/c/module_0_1BXN_0.mrc",
                   "/a/b/c/module_0_1BXN_1.mrc",
                   "/a/b/c/module_0_3GL1_0.mrc",
                   "/a/b/c/module_0_3GL1_2.mrc",
                   ]
        pdbs = ["/x/y/z/1bxn.mrc","/x/y/z/3gl1.mrc"]
        ftp = FilenameMatchingTripletProvider(path_pdb="",
                                              path_volume="",
                                              max_neg=5)
        triplets = ftp.generate_triplets(pdbs, volumes)
        train_split, valid_split = train_test_split_anchor_positive(triplets,split=0.75)


        self.assertEqual(20*0.75, len(train_split))
        self.assertEqual(20*(1-0.75), len(valid_split))


    def test_get_augmentations_two_piplines(self):
        augs = get_augmentations(aug_train_shift_distance=3)
        self.assertEqual(2, len(augs))

    def test_get_loss_func_arcface(self):
        net_conf = {
            "output_channels": 4
        }
        train_conf = {
            "loss": "ArcFaceLoss",
            "num_classes" : 12,
            "af_margin": 3,
            "af_scale": 3

        }
        from pytorch_metric_learning import losses
        loss = get_loss_func(distance=None,train_conf=train_conf,net_conf=net_conf)
        self.assertIsInstance(loss,losses.ArcFaceLoss)

    def test_get_loss_func_sphereface(self):
        net_conf = {
            "output_channels": 4
        }
        train_conf = {
            "loss": "SphereFaceLoss",
            "num_classes" : 12,
            "sf_margin": 3,
            "sf_scale": 3

        }
        from pytorch_metric_learning import losses
        loss = get_loss_func(distance=None,train_conf=train_conf,net_conf=net_conf)
        self.assertIsInstance(loss,losses.SphereFaceLoss)

    def test_get_loss_func_TripletLoss(self):
        net_conf = {
        }
        train_conf = {
            "loss": "TripletLoss",
            "tl_margin": 3,
        }
        from pytorch_metric_learning import losses
        loss = get_loss_func(distance=None,train_conf=train_conf,net_conf=net_conf)
        self.assertIsInstance(loss,losses.TripletMarginLoss)

    def test_get_loss_func_unknown(self):
        net_conf = {
        }
        train_conf = {
            "loss": "Blub",
            "tl_margin": 3,
        }
        from tomotwin.modules.common import exceptions
        with self.assertRaises(exceptions.UnknownLoss):
            loss = get_loss_func(distance=None,train_conf=train_conf,net_conf=net_conf)



if __name__ == '__main__':
    unittest.main()
