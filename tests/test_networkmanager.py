import unittest
import os
import json
from tomotwin.modules.networks.resnet import Resnet
from tomotwin.modules.networks.SiameseNet3D import SiameseNet3D
from tomotwin.modules.networks.networkmanager import NetworkManager, NetworkNotExistError, MalformedConfigError

class MyTestCase(unittest.TestCase):
    def test_create_network_resnet(self):
        nw = NetworkManager()

        config_path = os.path.join(os.path.dirname(__file__), "../resources/configs/config_resnet.json")
        config = nw.load_configuration(config_path)

        model = nw.create_network(config)

        is_correct_network = isinstance(model, Resnet)
        self.assertEqual(True, is_correct_network)

    def test_create_network_siamesenet(self):
        nw = NetworkManager()

        config_path = os.path.join(os.path.dirname(__file__), "../resources/configs/config_siamese.json")
        config = nw.load_configuration(config_path)

        model = nw.create_network(config)

        is_correct_network = isinstance(model, SiameseNet3D)
        self.assertEqual(True, is_correct_network)

    def test_create_network_unknown(self):
        nw = NetworkManager()


        config = {
            "identifier": "UnknownNetwork",
            "network_config": {}
        }

        with self.assertRaises(NetworkNotExistError):
            nw.create_network(config)

    def test_create_malformed_config(self):
        nw = NetworkManager()

        config = {
            "network_config": {}
        }

        with self.assertRaises(MalformedConfigError):
            nw.check_format(config)



if __name__ == '__main__':
    unittest.main()
