"""
MIT License

Copyright (c) 2021 MPI-Dortmund

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from tomotwin.modules.networks.SiameseNet3D import SiameseNet3D
from tomotwin.modules.networks.resnet import Resnet
from tomotwin.modules.networks.densenet import DenseNet3D
from tomotwin.modules.networks.customtestmodels import FacebookNet
from tomotwin.modules.networks.customtestmodels import Resnet3D
from tomotwin.modules.networks.torchmodel import TorchModel
from tomotwin.modules.networks.dnet16 import DNet16
from typing import Dict
import json


class NetworkNotExistError(Exception):
    """Exception when network does not exist"""

    pass


class MalformedConfigError(Exception):
    """Expection when there when using malformed configuration files"""

    pass


class NetworkManager:
    """
    Factory for all networks.
    """

    def __init__(self):
        self.network_identifier_map = {}
        self.add_network("SiameseNet", SiameseNet3D)
        self.add_network("ResNet", Resnet)
        self.add_network("Facebook", FacebookNet)
        self.add_network("ResNet3D", Resnet3D)
        self.add_network("DenseNet3D", DenseNet3D)
        self.add_network("DNet16", DNet16)

    def add_network(self, key: str, netclass: TorchModel) -> None:
        """
        Add a network to the network identifier map.

        :param key: Identifier for the network
        :param netclass: Class for the network
        :return: None
        """
        self.network_identifier_map[key.upper()] = netclass

    def check_format(self, config: Dict) -> None:
        """
        Check if all necessary fields are in the configuration file.
        :param config: Configuration dictionary
        :return: None
        """
        if "identifier" not in config:
            raise MalformedConfigError(
                f"The keyword 'identifier' must be in the config file"
            )
        if "network_config" not in config:
            raise MalformedConfigError(
                f"The keyword 'network_config' must be in the config file"
            )
        if "train_config" not in config:
            raise MalformedConfigError(
                f"The keyword 'train_config' must be in the config file"
            )

    def load_configuration(self, config_path: str) -> Dict:
        """
        Load the configuration
        :param config_path: Path to config file.
        :return: None
        """
        with open(config_path) as json_file:
            config = json.load(json_file)
            self.check_format(config)

        return config

    def create_network(self, configuration: Dict) -> TorchModel:
        """
        Create the networking given in the configuration
        :param configuration:
        :return: A TorchModel
        """

        identifier = configuration["identifier"].upper()

        if identifier not in self.network_identifier_map:
            raise NetworkNotExistError(f"Network '{identifier}' does not exist")
        else:
            modelclass = self.network_identifier_map[identifier]
            config = configuration["network_config"]
            if "groups" in config:
                '''
                This can be removed at some point. I only keept it here to make it compatible with older models.
                '''
                config["norm_name"] = "GroupNorm"
                config["norm_kwargs"] = {"num_groups": config["groups"]}
                del config["groups"]
            model = modelclass(**config)

            return model
