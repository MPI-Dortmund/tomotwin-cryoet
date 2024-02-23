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

"""
Implementation was taken from here:
https://github.com/kenshohara/3D-ResNets-PyTorch/blob/master/models/resnet.py

"""
from functools import partial
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from tomotwin.modules.networks.torchmodel import TorchModel


def make_norm(norm: Dict, num_channels: int) -> nn.Module:
    if norm["module"] == nn.BatchNorm3d:
        norm["kwargs"]["num_features"] = num_channels
        return norm["module"](**norm["kwargs"])
    elif norm["module"] == nn.GroupNorm:
        norm["kwargs"]["num_channels"] = num_channels
        return norm["module"](**norm["kwargs"])
    else:
        raise ValueError("Not supported norm", norm["module"])


def get_in_c():
    return [64, 128, 256, 512]


def conv1x1x1(in_c, out_c, stride=1):
    return nn.Conv3d(in_c, out_c, kernel_size=1, stride=stride, bias=False)


def conv3x3x3(in_c, out_c, stride=1):
    return nn.Conv3d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_c, channels, norm: Dict, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x3x3(in_c, channels, stride)
        self.norm1 = make_norm(norm, channels)
        self.relu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3x3(channels, channels)
        self.norm2 = make_norm(norm, channels)
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_c, channels, norm: Dict, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_c, channels)
        self.norm1 = make_norm(norm, channels)
        self.conv2 = conv3x3x3(channels, channels, stride)
        self.norm2 = make_norm(norm, channels)
        self.conv3 = conv1x1x1(channels, channels * self.expansion)
        self.norm3 = make_norm(norm, channels * self.expansion)
        self.relu = nn.LeakyReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Resnet(TorchModel):

    NORM_BATCHNORM = "BatchNorm"
    NORM_GROUPNORM = "GroupNorm"

    class Model(nn.Module):


        def __init__(
            self,
            block,
            layers,
            block_inchannels,
            norm: Dict,
            n_input_channels=1,
            conv1_t_size=7,
            conv1_t_stride=1,
            no_max_pool=False,
            shortcut_type="B",
            widen_factor=1.0,
            out_head=128,
            dropout=0.2,
        ):
            super().__init__()

            block_inchannels = [int(x * widen_factor) for x in block_inchannels]

            self.in_c = block_inchannels[0]
            self.no_max_pool = no_max_pool
            self.norm = norm

            self.conv1 = nn.Conv3d(
                n_input_channels,
                self.in_c,
                kernel_size=(conv1_t_size, 7, 7),
                stride=(conv1_t_stride, 2, 2),
                padding=(conv1_t_size // 2, 3, 3),
                bias=False,
            )
            self.norm1 = make_norm(self.norm, self.in_c)
            self.relu = nn.LeakyReLU(inplace=True)
            self.no_max_pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
            self.layer1 = self._make_layer(
                block, block_inchannels[0], layers[0], shortcut_type
            )
            self.layer2 = self._make_layer(
                block, block_inchannels[1], layers[1], shortcut_type, stride=2
            )
            self.layer3 = self._make_layer(
                block, block_inchannels[2], layers[2], shortcut_type, stride=2
            )
            self.layer4 = self._make_layer(
                block, block_inchannels[3], layers[3], shortcut_type, stride=2
            )

            self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
            self.fc = nn.Linear(block_inchannels[3] * block.expansion, out_head)
            self.drop = nn.Dropout3d(p=dropout)

            for m in self.modules():
                if isinstance(m, nn.Conv3d):
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="leaky_relu"
                    )
                elif isinstance(m, nn.BatchNorm3d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

        def _downsample_basic_block(self, x, channels, stride):
            out = F.avg_pool3d(x, kernel_size=1, stride=stride)
            zero_pads = torch.zeros(
                out.size(0),
                channels - out.size(1),
                out.size(2),
                out.size(3),
                out.size(4),
            )
            if isinstance(out.data, torch.cuda.FloatTensor):
                zero_pads = zero_pads.cuda()

            out = torch.cat([out.data, zero_pads], dim=1)

            return out

        def _make_layer(self, block, channels, blocks, shortcut_type, stride=1):
            downsample = None
            if stride != 1 or self.in_c != channels * block.expansion:
                if shortcut_type == "A":
                    downsample = partial(
                        self._downsample_basic_block,
                        channels=channels * block.expansion,
                        stride=stride,
                    )
                else:
                    norm_func = make_norm(self.norm, channels * block.expansion)
                    downsample = nn.Sequential(
                        conv1x1x1(self.in_c, channels * block.expansion, stride),
                        norm_func,
                    )

            layers = []
            layers.append(
                block(
                    in_c=self.in_c,
                    channels=channels,
                    stride=stride,
                    downsample=downsample,
                    norm=self.norm
                )
            )
            self.in_c = channels * block.expansion
            for i in range(1, blocks):
                layers.append(block(in_c=self.in_c, channels=channels, norm=self.norm))

            return nn.Sequential(*layers)

        def forward(self, x):
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
            if not self.no_max_pool:
                x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)

            x = x.view(x.size(0), -1)
            x = self.drop(x)
            x = self.fc(x)
            x = F.normalize(x, p=2, dim=1)

            return x

    def setup_norm(self, norm_name : str, norm_kwargs: dict) -> Dict:
        norm = {}
        if norm_name == Resnet.NORM_BATCHNORM:
            norm["module"] = nn.BatchNorm3d
        if norm_name == Resnet.NORM_GROUPNORM:
            norm["module"] = nn.GroupNorm
        norm["kwargs"] = norm_kwargs

        return norm

    def __init__(
        self,
        model_depth,
        norm_name: str,
        norm_kwargs: Dict = {},
        n_input_channels=1,
        conv1_t_size=7,
        conv1_t_stride=1,
        no_max_pool=False,
        shortcut_type="B",
        widen_factor=1.0,
        out_head=128,
    ):
        super().__init__()
        assert model_depth in [10, 18, 34, 50, 101, 152, 200]
        self.model_depth = model_depth
        norm = self.setup_norm(norm_name, norm_kwargs)

        if model_depth == 10:
            self.model = self.Model(
                block=BasicBlock,
                layers=[1, 1, 1, 1],
                block_inchannels=get_in_c(),
                n_input_channels=n_input_channels,
                conv1_t_size=conv1_t_size,
                conv1_t_stride=conv1_t_stride,
                no_max_pool=no_max_pool,
                shortcut_type=shortcut_type,
                widen_factor=widen_factor,
                out_head=out_head,
                norm=norm
            )
        elif model_depth == 18:
            self.model = self.Model(
                block=BasicBlock,
                layers=[2, 2, 2, 2],
                block_inchannels=get_in_c(),
                n_input_channels=n_input_channels,
                conv1_t_size=conv1_t_size,
                conv1_t_stride=conv1_t_stride,
                no_max_pool=no_max_pool,
                shortcut_type=shortcut_type,
                widen_factor=widen_factor,
                out_head=out_head,
                norm=norm
            )
        elif model_depth == 34:
            self.model = self.Model(
                block=BasicBlock,
                layers=[3, 4, 6, 3],
                block_inchannels=get_in_c(),
                n_input_channels=n_input_channels,
                conv1_t_size=conv1_t_size,
                conv1_t_stride=conv1_t_stride,
                no_max_pool=no_max_pool,
                shortcut_type=shortcut_type,
                widen_factor=widen_factor,
                out_head=out_head,
                norm=norm
            )
        elif model_depth == 50:
            self.model = self.Model(
                block=Bottleneck,
                layers=[3, 4, 6, 3],
                block_inchannels=[64, 128, 256, 512],
                n_input_channels=n_input_channels,
                conv1_t_size=conv1_t_size,
                conv1_t_stride=conv1_t_stride,
                no_max_pool=no_max_pool,
                shortcut_type=shortcut_type,
                widen_factor=widen_factor,
                out_head=out_head,
                norm=norm
            )
        elif model_depth == 101:
            self.model = self.Model(
                block=Bottleneck,
                layers=[3, 4, 23, 3],
                block_inchannels=get_in_c(),
                n_input_channels=n_input_channels,
                conv1_t_size=conv1_t_size,
                conv1_t_stride=conv1_t_stride,
                no_max_pool=no_max_pool,
                shortcut_type=shortcut_type,
                widen_factor=widen_factor,
                out_head=out_head,
                norm=norm
            )
        elif model_depth == 152:
            self.model = self.Model(
                block=Bottleneck,
                layers=[3, 8, 36, 3],
                block_inchannels=get_in_c(),
                n_input_channels=n_input_channels,
                conv1_t_size=conv1_t_size,
                conv1_t_stride=conv1_t_stride,
                no_max_pool=no_max_pool,
                shortcut_type=shortcut_type,
                widen_factor=widen_factor,
                out_head=out_head,
                norm=norm
            )
        elif model_depth == 200:
            self.model = self.Model(
                block=Bottleneck,
                layers=[3, 24, 36, 3],
                block_inchannels=get_in_c(),
                n_input_channels=n_input_channels,
                conv1_t_size=conv1_t_size,
                conv1_t_stride=conv1_t_stride,
                no_max_pool=no_max_pool,
                shortcut_type=shortcut_type,
                widen_factor=widen_factor,
                out_head=out_head,
                norm=norm
            )

    def get_model(self) -> nn.Module:
        return self.model

    def init_weights(self):
        def _init_weights(model):
            if isinstance(model, nn.Conv3d):
                torch.nn.init.kaiming_normal_(model.weight)

        self.model.apply(_init_weights)
