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
import random

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffZ = x2.size(2) - x1.size(2)
        diffY = x2.size(3) - x1.size(3)
        diffX = x2.size(4) - x1.size(4)

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet3D(nn.Module):
    def __init__(self, n_channels, out_channels, bilinear=True):
        super(UNet3D, self).__init__()
        self.n_channels = n_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

        self._initialize_weights()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
    

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)




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

def train_autoencoder(model, data_loader, val_loader, optimizer, scheduler, logging, num_epochs=10, device='cuda', patience=10):
    writer = SummaryWriter(logging)
    model.to(device)
    model.train()
    best_loss = float('inf')
    early_stopping_counter = 0

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
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Validation loss did not improve for {patience} epochs. Early stopping...")
                break

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
                recon_data = model(input_data)
                loss = loss_function(recon_data, target_data)
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(data_loader)
    writer.add_scalar('Loss/val', avg_loss, epoch)
    return avg_loss

def evaluate (model, data_loader):
    total_loss = 0.0
    model.eval()
    with torch.no_grad():
        with tqdm(data_loader, desc="Evaluating", unit="batch") as progress_bar:
            for batch_idx, data in enumerate(progress_bar):
                input_data = data['input'].to(device)
                input_data = input_data.reshape(-1, 1, 37, 37, 37)
                target_data = data['target'].to(device)
                target_data = target_data.reshape(-1, 1, 37, 37, 37)
                recon_data = model(input_data)
                loss = loss_function(recon_data, target_data)
                total_loss += loss.item()
                progress_bar.set_postfix(loss=loss.item())
    
    avg_loss = total_loss / len(data_loader)
    print(f"Validation Loss: {avg_loss}")

def predict(model, data_loader):
    model.eval()
    predictions = []
    targets = []
    with torch.no_grad():
        for batch in data_loader:
            input_data = batch['input'].unsqueeze(1)
            target_data = batch['target'].unsqueeze(1)
            output = model(input_data)
            predictions.append(output.squeeze(1).cpu().numpy())
            targets.append(target_data.squeeze(1).cpu().numpy())
    return predictions, targets

def save_as_mrc(data, filename):
    with mrcfile.new(filename, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))

def save_ouput (out_dir: str, predictions, targets):
    for i, (pred, target) in enumerate(zip(predictions, targets)):
        pred_filename = os.path.join(out_dir, f'prediction_{i:03d}.mrc')
        target_filename = os.path.join(out_dir, f'target_{i:03d}.mrc')
        save_as_mrc(pred, pred_filename)
        save_as_mrc(target, target_filename)
        print(f'Saved prediction to {pred_filename} and target to {target_filename}')

def load_model_weights(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    #state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
    model.load_state_dict(checkpoint)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(args):
    set_seed(1)
    dataset = MRCVolumeDataset(args.dataset_root)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_set = MRCVolumeDataset(args.val_root)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    model = UNet3D(1,1)
    model = nn.DataParallel(model)
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    if args.checkpoint_path:
        load_model_weights(model, args.checkpoint_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    
    train_autoencoder(model, data_loader, val_loader, optimizer,  scheduler,args.logging, args.num_epochs, args.device)

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

