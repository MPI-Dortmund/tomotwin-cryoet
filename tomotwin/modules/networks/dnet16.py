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
SOFTWARE. SiameseNet3DOptim
"""
"""
Used this to understand better:
https://github.com/jarvislabsai/blog/blob/master/build_resnet34_pytorch/Building%20Resnet%20in%20PyTorch.ipynb
"""
import torch
import torch.nn as nn
from tomotwin.modules.networks.torchmodel import TorchModel
import torch.nn.functional as F
import torchvision.transforms as ttrans
from typing import Dict





class DNet16(TorchModel):

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

        def __init__(self, output_channels: int, dropout: float, norm: Dict):
            super().__init__()
            f=1
            self.inplane = 64
            self.norm = norm
            self.relu = nn.LeakyReLU(negative_slope=0.1)
            self.conv_layer1 = self._make_conv_layer(1, self.inplane*f, kernel_size_1=7, padding_1=5, stride_1=1)
            self.conv_layer2 = self._make_conv_layer(self.inplane*f, 64*f, kernel_size_1=3, padding_1=1, stride_1=1,
                                                     dilation=1)
            self.conv_layer3 = self._make_conv_layer(64*f, 128*f, kernel_size_1=3, padding_1=1, stride_1=1)
            self.conv_layer4 = self._make_conv_layer(128*f, 64*f, kernel_size_1=1, padding_1=0, stride_1=1)
            self.conv_layer5 = self._make_conv_layer(64*f, 128*f, kernel_size_1=3, padding_1=1, stride_1=1)
            self.conv_layer6 = self._make_conv_layer(128*f, 256*f, kernel_size_1=3, padding_1=1, stride_1=1)
            self.conv_layer7 = self._make_conv_layer(256*f, 128*f, kernel_size_1=1, padding_1=0, stride_1=1)
            self.conv_layer8 = self._make_conv_layer(128*f, 256*f, kernel_size_1=3, padding_1=1, stride_1=1)
            self.conv_layer9 = self._make_conv_layer(256*f, 512*f, kernel_size_1=3, padding_1=1, stride_1=1)

            self.conv_layer10 = self._make_conv_layer(512, 256, kernel_size_1=1, padding_1=0, stride_1=1)
            self.conv_layer11 = self._make_conv_layer(256, 512, kernel_size_1=3, padding_1=1, stride_1=1)
            self.conv_layer12 = self._make_conv_layer(512, 256, kernel_size_1=1, padding_1=0, stride_1=1)
            self.conv_layer13 = self._make_conv_layer(256, 512, kernel_size_1=3, padding_1=1, stride_1=1)
            self.conv_layer14 = self._make_conv_layer(512, 1024, kernel_size_1=3, padding_1=1, stride_1=1)
            self.conv_layer15 = self._make_conv_layer(1024, 512, kernel_size_1=1, padding_1=0, stride_1=1)
            self.conv_layer16 = self._make_conv_layer(768, 1024, kernel_size_1=3, padding_1=1, stride_1=1)

            self.upsample = nn.Upsample(scale_factor=2)
            final_pool_size = (2,2,2)
            self.max_pooling = nn.MaxPool3d(final_pool_size)
            self.avg_pooling = nn.AvgPool3d(final_pool_size)
            self.adapt_max_pool = nn.AdaptiveMaxPool3d(output_size=final_pool_size)
            self.adapt_avg_pool = nn.AdaptiveAvgPool3d(output_size=final_pool_size)

            self.headnet = self._make_headnet(final_pool_size[0] * final_pool_size[1] * final_pool_size[2] * 1024, 2048, output_channels, dropout=dropout)


        def _make_conv_layer(self, in_c: int, out_c: int, kernel_size_1=3, stride_1 = 1, padding_1=0, dilation=1):
            norm_func = self.make_norm(self.norm, out_c)
            conv_layer = nn.Sequential(
                nn.Conv3d(in_c, out_c,
                          kernel_size=kernel_size_1,
                          padding=(padding_1,padding_1,padding_1),
                          stride=(stride_1,stride_1,stride_1),
                          dilation=dilation,
                          bias=False),
                norm_func,
                nn.LeakyReLU(),
            )
            return conv_layer

        @staticmethod
        def _make_headnet(in_c1: int, out_c1: int, out_head: int, dropout: float) -> nn.Sequential:
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
            sizes =[]
            out = self.conv_layer1(inputtensor)
            #out = self.max_pooling(out)
            out = self.conv_layer2(out)
            out = self.max_pooling(out)
            out = self.conv_layer3(out)
            out = self.conv_layer4(out)
            out = self.conv_layer5(out)


            out = self.max_pooling(out)

            out = self.conv_layer6(out)
            out6 = out
            out = self.conv_layer7(out)
            out = self.conv_layer8(out)

            out = self.max_pooling(out)
            out = self.conv_layer9(out)

            out = self.conv_layer10(out)
            out = self.conv_layer11(out)
            out = self.conv_layer12(out)
            out = self.conv_layer13(out)

            out = self.conv_layer14(out)
            out = self.conv_layer15(out)
            out = self.upsample(out)
            #out = nn.functional.pad(out,(0,1,0,1,0,1))
            out = torch.cat((out, out6), dim=1)
            out = self.conv_layer16(out)

            out = self.adapt_max_pool(out)
            out = out.reshape(out.size(0), -1)
            #print("S", out.size())
            out = self.headnet(out)
            out = F.normalize(out, p=2, dim=1)

            return out

    """
    Custom 3D convnet, nothing fancy
    """

    def setup_norm(self, norm_name : str, norm_kwargs: dict) -> Dict:
        norm = {}
        if norm_name == DNet16.NORM_BATCHNORM:
            norm["module"] = nn.BatchNorm3d
        if norm_name == DNet16.NORM_GROUPNORM:
            norm["module"] = nn.GroupNorm
        norm["kwargs"] = norm_kwargs

        return norm

    def __init__(self,
                 output_channels: int,
                 dropout: float,
                 norm_name: str,
                 norm_kwargs: Dict = {},
                 ):

        super().__init__()
        norm = self.setup_norm(norm_name, norm_kwargs)
        self.model = self.Model(output_channels, dropout, norm)

    def init_weights(self):
        def _init_weights(model):
            if isinstance(model, nn.Conv3d):
                torch.nn.init.kaiming_normal_(model.weight)

        self.model.apply(_init_weights)

    def get_model(self) -> nn.Module:
        return self.model
