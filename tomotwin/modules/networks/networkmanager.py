"""
Copyright (c) 2022 MPI-Dortmund
SPDX-License-Identifier: MPL-2.0

This file is subject to the terms of the Mozilla Public License, Version 2.0 (MPL-2.0).
The full text of the MPL-2.0 can be found at http://mozilla.org/MPL/2.0/.

For files that are Incompatible With Secondary Licenses, as defined under the MPL-2.0,
additional notices are required. Refer to the MPL-2.0 license for more details on your
obligations and rights under this license and for instructions on how secondary licenses
may affect the distribution and modification of this software.
"""

import json
from typing import Dict

from tomotwin.modules.networks.SiameseNet3D import SiameseNet3D
from tomotwin.modules.networks.resnet import Resnet
from tomotwin.modules.networks.torchmodel import TorchModel
from tomotwin.modules.networks.Unet import UNet
from tomotwin.modules.networks.Unet_GN import UNet_GN
from tomotwin.modules.networks.resunet import resunet


class NetworkNotExistError(Exception):
    """Exception when network does not exist"""


class MalformedConfigError(Exception):
    """Expection when there when using malformed configuration files"""


class NetworkManager:
    """
    Factory for all networks.
    """
    network_identifier_map = {
        "SiameseNet".upper(): SiameseNet3D,
        "ResNet".upper(): Resnet,
        "UNet".upper(): UNet,
        "UNet_GN".upper(): UNet_GN,
        "resunet".upper(): resunet
    }


    @staticmethod
    def add_network(key: str, netclass: TorchModel) -> None:
        """
        Add a network to the network identifier map.

        :param key: Identifier for the network
        :param netclass: Class for the network
        :return: None
        """
        NetworkManager.network_identifier_map[key.upper()] = netclass


    @staticmethod
    def check_format(config: Dict) -> None:
        """
        Check if all necessary fields are in the configuration file.
        :param config: Configuration dictionary
        :return: None
        """
        if "identifier" not in config:
            raise MalformedConfigError(
                "The keyword 'identifier' must be in the config file"
            )
        if "network_config" not in config:
            raise MalformedConfigError(
                "The keyword 'network_config' must be in the config file"
            )
        if "train_config" not in config:
            raise MalformedConfigError(
                "The keyword 'train_config' must be in the config file"
            )
    @staticmethod
    def load_configuration(config_path: str) -> Dict:
        """
        Load the configuration
        :param config_path: Path to config file.
        :return: None
        """
        with open(config_path) as json_file:
            config = json.load(json_file)
            NetworkManager.check_format(config)

        return config

    @staticmethod
    def create_network(configuration: Dict) -> TorchModel:
        """
        Create the networking given in the configuration
        :param configuration:
        :return: A TorchModel
        """

        identifier = configuration["identifier"].upper()

        if identifier not in NetworkManager.network_identifier_map:
            raise NetworkNotExistError(f"Network '{identifier}' does not exist")
        else:
            modelclass = NetworkManager.network_identifier_map[identifier]
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
