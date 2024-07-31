from typing import Dict, Union
import torch
import torch.nn as nn
from tomotwin.modules.networks.torchmodel import TorchModel


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding='same'),
            nn.GroupNorm(num_groups, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels, num_groups)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, num_groups):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, num_groups)

    def forward(self, x1):
        x1 = self.up(x1)
        return self.conv(x1)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=3, padding='same'),
                                  nn.Sigmoid())

    def forward(self, x):
        return self.conv(x)
    
class UNet3D(nn.Module):
    def __init__(self, n_channels, out_channels, num_groups=8):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels

        self.inc = DoubleConv(n_channels, 32, num_groups)
        self.down1 = Down(32, 32, num_groups)
        self.down2 = Down(32, 64, num_groups)
        self.down3 = Down(64, 64, num_groups)
        #self.up1 = Up(64, 64, num_groups)
        #self.up2 = Up(64, 32, num_groups)
        #self.up3 = Up(32, 32, num_groups)
        #self.outc = OutConv(32, out_channels)
        self.Flatten = nn.Flatten()
        self._initialize_weights()
        self._load_checkpoint()
    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x_flat = self.Flatten(x4)
        x_flat = torch.nn.functional.normalize(x_flat, p=2.0, dim=1)
        #x = x_flat.view(x4.size())
        #x = self.up1(x)
        #x = self.up2(x)
        #x = self.up3(x)
        #logits = self.outc(x)
        return x_flat

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _load_checkpoint(self):
        chk_pth = '/home/yousef.metwally/projects/no32_64_4_sphere/weights/model_weights_epoch_206.pt'
        checkpoint = torch.load(chk_pth)
        state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
        model_state_dict = self.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_state_dict}
        self.load_state_dict(filtered_state_dict)




class UNet_GN(TorchModel):
    def __init__(self) -> None:
        super().__init__()
        self.model = UNet3D(1,1)

    def init_weights(self):
        self.model._initialize_weights()
        self.model._load_checkpoint()

    def get_model(self) -> nn.Module:
        return self.model
