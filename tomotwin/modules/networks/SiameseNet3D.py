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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Union
from tomotwin.modules.networks.torchmodel import TorchModel


class ResnetBlock(nn.Module):
    def __init__(self, in_c, out_c, norm: nn.Module, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_c,
            out_c,
            kernel_size=3,
            stride=(stride, stride, stride),
            padding=(1, 1, 1),
            bias=False,
        )
        self.norm = norm

        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            out_c, out_c, kernel_size=3, stride=(1, 1, 1), padding=(1, 1, 1), bias=False
        )
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.norm(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SiameseNet3D(TorchModel):

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

        def make_repeat(self, r: int, inc: int, outc: int, norm: Dict):
            if r == 0:
                return None
            rep = []
            for r in range(r):
                rep.append(ResnetBlock(inc, outc, norm=norm))
            return nn.Sequential(*rep)

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
            self.conv_layer0 = self._make_conv_layer(1, 64, norm=norm_func)
            self.rep_layers0 = self.make_repeat(
                r=repeat_layers, inc=64, outc=64, norm=norm_func
            )

            norm_func = self.make_norm(norm, 128)
            self.conv_layer1 = self._make_conv_layer(64, 128, norm=norm_func)
            self.rep_layers1 = self.make_repeat(
                r=repeat_layers, inc=128, outc=128, norm=norm_func
            )

            norm_func = self.make_norm(norm, 256)
            self.conv_layer2 = self._make_conv_layer(128, 256, norm=norm_func)
            self.rep_layers2 = self.make_repeat(
                r=repeat_layers, inc=256, outc=256, norm=norm_func
            )

            norm_func = self.make_norm(norm, 512)
            self.conv_layer3 = self._make_conv_layer(256, 512, norm=norm_func)
            self.rep_layers3 = self.make_repeat(
                r=repeat_layers, inc=512, outc=512, norm=norm_func
            )

            norm_func = self.make_norm(norm, 1024)
            self.conv_layer4 = self._make_conv_layer(512, 1024, norm=norm_func)
            self.rep_layers4 = self.make_repeat(
                r=repeat_layers, inc=1024, outc=1024, norm=norm_func
            )

            self.max_pooling = nn.MaxPool3d((2, 2, 2))
            if gem_pooling:
                self.adap_max_pool = gem_pooling
            else:
                self.adap_max_pool = nn.AdaptiveAvgPool3d((2, 2, 2))
            self.headnet = self._make_headnet(
                2 * 2 * 2 * 1024, 2048, output_channels, dropout=dropout
            )

        @staticmethod
        def _make_conv_layer(in_c: int, out_c: int, norm: nn.Module, padding: int = 0):
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
        def _make_headnet(
            in_c1: int, out_c1: int, out_head: int, dropout: float
        ) -> nn.Sequential:
            headnet = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_c1, out_c1),
                nn.LeakyReLU(),
                nn.Linear(out_c1, out_c1),
                nn.LeakyReLU(),
                nn.Linear(out_c1, out_head),
            )
            return headnet

        def forward(self, inputtensor):
            """
            Forward pass through the network
            :param inputtensor: Input tensor
            """

            inputtensor = nn.functional.pad(inputtensor, (1, 2, 1, 2, 1, 2))

            out = self.conv_layer0(inputtensor)
            out = self.max_pooling(out)
            if self.rep_layers0:
                out = self.rep_layers0(out)

            out = self.conv_layer1(out)
            if self.rep_layers1:
                out = self.rep_layers1(out)

            out = self.conv_layer2(out)
            if self.rep_layers2:
                out = self.rep_layers2(out)

            out = self.conv_layer3(out)
            if self.rep_layers3:
                out = self.rep_layers3(out)

            out = self.conv_layer4(out)
            if self.rep_layers4:
                out = self.rep_layers4(out)

            out = self.adap_max_pool(out)
            # print("P", out.shape)
            out = out.reshape(out.size(0), -1)  # flatten
            out = self.headnet(out)
            out = F.normalize(out, p=2, dim=1)

            return out

    """
    Custom 3D convnet, nothing fancy
    """

    def setup_norm(self, norm_name : str, norm_kwargs: dict) -> Dict:
        norm = {}
        if norm_name == SiameseNet3D.NORM_BATCHNORM:
            norm["module"] = nn.BatchNorm3d
        if norm_name == SiameseNet3D.NORM_GROUPNORM:
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
