from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from tomotwin.modules.networks.torchmodel import TorchModel

class AutoEncoder(TorchModel):

    NORM_BATCHNORM = "BatchNorm"
    NORM_GROUPNORM = "GroupNorm"

    class Model(nn.Module):
        def make_norm(self, norm: Dict, num_channels: int) -> nn.Module:
            if norm["module"] == nn.BatchNorm3d:
                norm["kwargs"]["num_features"] = num_channels
                return norm["module"](**norm["kwargs"])
            elif norm["module"] == nn.GroupNorm:
                norm["kwargs"]["num_channels"] = num_channels
                return norm["module"](**norm["kwargs"])
            else:
                raise ValueError("Not supported norm", norm["module"])


        def __init__(
            self,
            output_channels: int,
            norm: Dict,
            dropout: float = 0.5,
            repeat_layers=0,
            gem_pooling = None,
        ):
            super().__init__()
            norm_func = self.make_norm(norm, 64)
            self.en_layer0 = self._make_conv_layer(1, 64, norm=norm_func)

            norm_func = self.make_norm(norm, 128)
            self.en_layer1 = self._make_conv_layer(64, 128, norm=norm_func)

            norm_func = self.make_norm(norm, 256)
            self.en_layer2 = self._make_conv_layer(128, 256, norm=norm_func)

            norm_func = self.make_norm(norm, 512)
            self.en_layer3 = self._make_conv_layer(256, 512, norm=norm_func)


            self.max_pooling = nn.MaxPool3d((2, 2, 2))
            if gem_pooling:
                self.adap_max_pool = gem_pooling
            else:
                self.adap_max_pool = nn.AdaptiveAvgPool3d((2, 2, 2))
            
            self.headnet = self._make_headnet(
                 512, 256, 64, 1, dropout=dropout
            )

            norm_func = self.make_norm(norm, 256)
            self.de_layer0 = self._make_deconv_layer(512, 256, norm=norm_func)

            norm_func = self.make_norm(norm, 128)
            self.de_layer1 = self._make_deconv_layer(256, 128, norm=norm_func)

            norm_func = self.make_norm(norm, 64)
            self.de_layer2 = self._make_deconv_layer(128, 64, norm=norm_func)

            #norm_func = self.make_norm(norm, 64)
            #self.de_layer3 = self._make_conv_layer(128, 64, norm=norm_func)

            #norm_func = self.make_norm(norm, 1)
            #self.de_layer4 = self._make_conv_layer(64, 1, norm=norm_func)
            self.de_layer4 = nn.Sequential(
                nn.ConvTranspose3d(64, 1, kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.ConvTranspose3d(1, 1, kernel_size=3, padding=1),
                nn.Identity() 
            )

            self.up_sampling = nn.Upsample(scale_factor =2)

        @staticmethod
        def _make_conv_layer(in_c: int, out_c: int, norm: nn.Module, padding: int = 1, kernel_size: int =3):
            conv_layer = nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=padding),
                norm,
                nn.LeakyReLU(),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=padding),
                norm,
                nn.LeakyReLU(),
            )
            return conv_layer
        
        @staticmethod
        def _make_deconv_layer(in_c: int, out_c: int, norm: nn.Module, padding: int = 1, kernel_size: int =3):
            conv_layer = nn.Sequential(
                nn.ConvTranspose3d(in_c, out_c, kernel_size=3, padding=padding),
                norm,
                nn.LeakyReLU(),
                nn.ConvTranspose3d(out_c, out_c, kernel_size=3, padding=padding),
                norm,
                nn.LeakyReLU(),
            )
            return conv_layer

        @staticmethod
        def _make_headnet(
            in_c1: int, in_c2: int,out_c1: int, out_head: int, dropout: float
        ) -> nn.Sequential:
            headnet = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Conv3d(in_c1, in_c2, kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.Conv3d(in_c2, out_c1, kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.Conv3d(out_c1, out_head, kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.ConvTranspose3d(out_head, out_c1, kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.ConvTranspose3d(out_c1, in_c2, kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.ConvTranspose3d(in_c2,in_c1, kernel_size=3, padding=1),
                nn.LeakyReLU(),


            )
            return headnet

        def forward(self, inputtensor):
            """
            Forward pass through the network
            :param inputtensor: Input tensor
            """
            inputtensor = F.pad(inputtensor, (1, 2, 1, 2, 1, 2))

            out = self.en_layer0(inputtensor)
            out = self.max_pooling(out)
            out = self.en_layer1(out)
            out = self.max_pooling(out)
            out = self.en_layer2(out)
            out = self.max_pooling(out)
            out = self.en_layer3(out)
            #out = self.max_pooling(out)

            #out = self.en_layer4(out)
            #out = self.adap_max_pool(out)
            #out = out.reshape(out.size(0), -1)  # flatten
            out = self.headnet(out)
            #out = out.reshape(-1,512,5,5,5)
            out = self.de_layer0(out)
            out = self.up_sampling(out)
            out = self.de_layer1(out)
            out = self.up_sampling(out)
            out = self.de_layer2(out)
            out = self.up_sampling(out)
            #out = self.de_layer3(out)
            #out = self.up_sampling(out)
            out = self.de_layer4(out)
            #out = F.normalize(out, p=2, dim=1)

            return out

    """
    Custom 3D convnet, nothing fancy
    """

    def setup_norm(self, norm_name : str, norm_kwargs: dict) -> Dict:
        norm = {}
        if norm_name == AutoEncoder.NORM_BATCHNORM:
            norm["module"] = nn.BatchNorm3d
        if norm_name == AutoEncoder.NORM_GROUPNORM:
            norm["module"] = nn.GroupNorm
        norm["kwargs"] = norm_kwargs

        return norm


    def setup_gem_pooling(self,gem_pooling_p : float) -> Union[None, nn.Module]:
        gem_pooling = None
        if gem_pooling_p > 0:
            from tomotwin.modules.networks.GeneralizedMeanPooling import GeneralizedMeanPooling
            gem_pooling = GeneralizedMeanPooling(norm=gem_pooling_p, output_size=(2, 2, 2))
        return gem_pooling

    def __init__(
        self,
        norm_name: str,
        norm_kwargs: Dict = {},
        output_channels: int = 128,
        dropout: float = 0.5,
        gem_pooling_p: float = 0,
        repeat_layers=0,
    ):
        super().__init__()
        norm = self.setup_norm(norm_name, norm_kwargs)
        gem_pooling = self.setup_gem_pooling(gem_pooling_p)


        self.model = self.Model(
            output_channels=output_channels,
            dropout=dropout,
            repeat_layers=repeat_layers,
            norm=norm,
            gem_pooling=gem_pooling
        )

    def init_weights(self):
        def _init_weights(model):
            if isinstance(model, nn.Conv3d):
                torch.nn.init.kaiming_normal_(model.weight)

        self.model.apply(_init_weights)

    def get_model(self) -> nn.Module:
        return self.model
    
