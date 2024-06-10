#!/raven/u/ymetwally/miniforge3/envs/tomotwin/bin/python3.10
# -*- coding: utf-8 -*-


import argparse
from typing import Dict, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tomotwin.modules.networks.torchmodel import TorchModel
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tomotwin.modules.training.mrctriplethandler import MRCTripletHandler
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
            self.en_layer0 = self._make_conv_new(1, 64, norm=norm_func)

            norm_func = self.make_norm(norm, 128)
            self.en_layer1 = self._make_conv_new(64, 128, norm=norm_func)

            norm_func = self.make_norm(norm, 256)
            self.en_layer2 = self._make_conv_new(128, 256, norm=norm_func)

            norm_func = self.make_norm(norm, 512)
            self.en_layer3 = self._make_conv_new(256, 512, norm=norm_func)


            self.max_pooling = nn.MaxPool3d((2, 2, 2))
            if gem_pooling:
                self.adap_max_pool = gem_pooling
            else:
                self.adap_max_pool = nn.AdaptiveAvgPool3d((2, 2, 2))
            
            self.headnet = self._make_headnet(
                 512, 256, 64, 1, dropout=dropout
            )

            norm_func = self.make_norm(norm, 256)
            self.de_layer0 = self._make_conv_new(512, 256, norm=norm_func)

            norm_func = self.make_norm(norm, 128)
            self.de_layer1 = self._make_conv_new(256, 128, norm=norm_func)

            norm_func = self.make_norm(norm, 64)
            self.de_layer2 = self._make_conv_new(128, 64, norm=norm_func)

            #norm_func = self.make_norm(norm, 64)
            #self.de_layer3 = self._make_conv_layer(128, 64, norm=norm_func)

            #norm_func = self.make_norm(norm, 1)
            #self.de_layer4 = self._make_conv_layer(64, 1, norm=norm_func)
            self.de_layer4 = nn.Sequential(
                nn.Conv3d(64, 1, kernel_size=3, padding=1),
                nn.LeakyReLU(),
                nn.Conv3d(1, 1, kernel_size=1),
                nn.Identity() 
            )

            self.up_sampling = nn.Upsample(scale_factor =2, mode='trilinear', align_corners = True)

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
        def _make_conv_new(in_c: int, out_c: int, norm: nn.Module, padding: int = 1, kernel_size: int =3):
            conv_layer = nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=padding),
                nn.BatchNorm3d(out_c),
                nn.LeakyReLU(),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=padding),
                nn.BatchNorm3d(out_c),
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

            #out = self.headnet(out)
            #out = out.reshape(-1,512,5,5,5)
            out = self.de_layer0(out)
            out = self.up_sampling(out)
            out = self.de_layer1(out)
            out = self.up_sampling(out)
            out = self.de_layer2(out)
            out = self.up_sampling(out)
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
    
norm_name = "GroupNorm"
norm_kwargs = {"num_groups": 64,
        "num_channels": 1024}




class MRCVolumeDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_paths = self._get_file_paths()
        self.reader = MRCTripletHandler()

    def _get_file_paths(self):
        file_paths = []
        for round_dir in os.listdir(self.root_dir):
            round_path = os.path.join(self.root_dir, round_dir)
            if os.path.isdir(round_path):
                for tomo_dir in os.listdir(round_path):
                    tomo_path = os.path.join(round_path, tomo_dir)
                    if os.path.isdir(tomo_path):
                        mrc_files = [f for f in os.listdir(tomo_path) if f.endswith('.mrc')]
                        for mrc_file in mrc_files:
                            file_paths.append(os.path.join(tomo_path, mrc_file))
        return file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        mrc_path = self.file_paths[idx]
        volume = self.reader.read_mrc_and_norm(mrc_path)

        return {'input': volume, 'target': volume}
    

def loss_function(recon_x, x):
    mse_loss = F.mse_loss(recon_x, x)
    return mse_loss

def train_autoencoder(model, data_loader,val_loader, optimizer,scheduler, logging, num_epochs=10, device='cuda'):
    writer = SummaryWriter(logging)
    model.to(device)
    model.train()
    best_loss = float('inf')
    if not os.path.exists(f"{logging}/weights"):
        os.makedirs(f"{logging}/weights")

    for epoch in range(num_epochs):
        total_loss = 0.0
        with tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as progress_bar:
            for batch_idx, data in enumerate(progress_bar):
                input_data = data['input'].to(device)
                input_data = input_data.reshape(-1,1,37,37,37)
                target_data = data['target'].to(device)
                target_data = target_data.reshape(-1,1,37,37,37)
                target_data = F.pad(target_data, (1, 2, 1, 2, 1, 2))  
                optimizer.zero_grad()
                recon_data = model(input_data)
                loss = loss_function(recon_data, target_data)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                writer.add_scalar('Loss/train', avg_loss, epoch * len(data_loader) + batch_idx)
                progress_bar.set_postfix(loss=avg_loss)
                   

        val_loss = validate_autoencoder(model,val_loader,writer, epoch+1,num_epochs)
        scheduler.step(val_loss)
        if val_loss < best_loss:
            torch.save(model.state_dict(), f"{logging}/weights/model_weights_epoch_{epoch+1}.pt")
            best_loss = val_loss

        print(f"Epoch {epoch+1}/{num_epochs}, Avg. Loss: {avg_loss:.4f}, Val. Loss: {val_loss:.4f}")

    writer.close()

def validate_autoencoder(model, data_loader, writer, epoch,num_epochs, device='cuda'):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        with tqdm(data_loader, desc=f"Validation Epoch {epoch}/{num_epochs}", unit="batch") as progress_bar:
            for batch_idx, data in enumerate(progress_bar):
                input_data = data['input'].to(device)
                input_data = input_data.reshape(-1, 1, 37, 37, 37)
                target_data = data['target'].to(device)
                target_data = target_data.reshape(-1, 1, 37, 37, 37)
                target_data = F.pad(target_data, (1, 2, 1, 2, 1, 2))
                recon_data = model(input_data)
                loss = loss_function(recon_data, target_data)
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(data_loader)
    writer.add_scalar('Loss/val', avg_loss, epoch)
    return avg_loss

def load_model_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    #state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)

def main(args):
    dataset = MRCVolumeDataset(args.dataset_root)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_set = MRCVolumeDataset(args.val_root)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    M = AutoEncoder(norm_name,norm_kwargs,32)
    model = M.get_model()
    model = nn.DataParallel(model)
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    if args.checkpoint_path:
        load_model_weights(model, args.checkpoint_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    
    train_autoencoder(model, data_loader, val_loader, optimizer,scheduler,  args.logging, args.num_epochs, args.device)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Training script for autoencoder")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loader")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for training (cuda or cpu)")
    parser.add_argument("--dataset_root", type=str, default="",  help="Path for you dataset root directory")
    parser.add_argument("--val_root", type=str, default="",  help="Path for you validation dataset root directory")
    parser.add_argument("--logging", type=str, default="", help="Path for you logging directory including /run number")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to the saved model checkpoint")
    

    args = parser.parse_args()

    
    main(args)

