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
from tomotwin.modules.networks.torchmodel import TorchModel
import torch.nn.functional as F

class Dense_Block(nn.Module):

    def __init__(self, in_c):

        super(Dense_Block, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm3d(in_c)
        self.conv1 = nn.Conv3d(in_c, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d(96, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv3d(128, 32, kernel_size=3, padding=1)

    def forward(self, inputtensor):
        bn = self.bn(inputtensor)
        out1 = self.relu(self.conv1(bn))
        out2 = self.relu(self.conv2(out1))
        c2_dense = self.relu(torch.cat([out1, out2], 1))
        out3 = self.relu(self.conv3(c2_dense))
        c3_dense = self.relu(torch.cat([out1, out2, out3], 1))
        out4 = self.relu(self.conv4(c3_dense))
        c4_dense = self.relu(torch.cat([out1, out2, out3, out4], 1))
        out5 = self.relu(self.conv5(c4_dense))
        c5_dense = self.relu(torch.cat([out1, out2, out3, out4, out5], 1))

        return c5_dense

class Transition_Layer(nn.Module):
    def __init__(self, in_c, out_c):
        super(Transition_Layer, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm3d(out_c)
        self.conv = nn.Conv3d(in_c, out_c, kernel_size=1, bias=False)
        self.avg_pool = nn.AvgPool3d(kernel_size=2, stride=2, padding=0)

    def forward(self, inputtensor):
        bn = self.bn(self.relu(self.conv(inputtensor)))
        out = self.avg_pool(bn)
        return out

class DenseNet3D(TorchModel):
    class Model(nn.Module):
        def __init__(self, output_channels):
            super().__init__()

            self.lowconv = nn.Conv3d(1, 64, kernel_size=3, padding=1)
            self.relu = nn.ReLU()

            self.denseblock1 = self._make_dense_block(Dense_Block, 64)
            self.denseblock2 = self._make_dense_block(Dense_Block, 128)
            self.denseblock3 = self._make_dense_block(Dense_Block, 128)

            self.transitionlayer1 = self._make_transition_layer(Transition_Layer, in_c=160, out_c=128)
            self.transitionlayer2 = self._make_transition_layer(Transition_Layer, in_c=160, out_c=128)
            self.transitionlayer3 = self._make_transition_layer(Transition_Layer, in_c=160, out_c=64)

            self.preclass = nn.Linear(4096, 1024)
            self.head_out = nn.Linear(1024, output_channels)

        def _make_dense_block(self, block, in_c):
            layers = []
            layers.append(block(in_c))
            return nn.Sequential(*layers)

        def _make_transition_layer(self, layer, in_c, out_c):
            modules = []
            modules.append(layer(in_c, out_c))
            return nn.Sequential(*modules)

        def forward(self, inputtensor):
            out = self.relu(self.lowconv(inputtensor))
            out = self.denseblock1(out)
            out = self.transitionlayer1(out)
            out = self.denseblock2(out)
            out = self.transitionlayer2(out)
            out = self.denseblock3(out)
            out = self.transitionlayer3(out)
            output = out.reshape(out.size(0), -1) # added flatten
            output = self.preclass(output)
            output = self.head_out(output)
            output = F.normalize(output, p=2, dim=1) # added normalize

            return output

    def __init__(self, output_channels=128):

        super().__init__()
        self.model = self.Model(output_channels)

    def init_weights(self):
        def _init_weights(model):
            if isinstance(model, nn.Conv3d):
                torch.nn.init.kaiming_normal_(model.weight)

        self.model.apply(_init_weights)

    def get_model(self) -> nn.Module:
        return self.model
