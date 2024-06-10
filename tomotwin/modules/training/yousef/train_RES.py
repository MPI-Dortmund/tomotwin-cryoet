#!/raven/u/ymetwally/miniforge3/envs/tomotwin/bin/python3.10
# -*- coding: utf-8 -*-

import argparse
import random
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
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import mrcfile

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        if in_channels != out_channels:
            self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = None

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        
        if self.residual_conv:
            residual = self.residual_conv(residual)
        
        x += residual
        return F.relu(x)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)
        self.res1 = ResidualBlock(64, 64)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.res2 = ResidualBlock(128, 128)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.res3 = ResidualBlock(256, 256)
        self.conv4 = nn.Conv3d(256, 512, kernel_size=3, padding=1)
        self.res4 = ResidualBlock(512, 512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #print(f'After conv1: {x.shape}')
        x = F.max_pool3d(x, 2)
        #print(f'After max_pool3d 1: {x.shape}')
        x = self.res1(x)
        #print(f'After res1: {x.shape}')
        
        x = F.relu(self.conv2(x))
        #print(f'After conv2: {x.shape}')
        x = F.max_pool3d(x, 2)
        #print(f'After max_pool3d 2: {x.shape}')
        x = self.res2(x)
        #print(f'After res2: {x.shape}')
        
        x = F.relu(self.conv3(x))
        #print(f'After conv3: {x.shape}')
        x = F.max_pool3d(x, 2)
        #print(f'After max_pool3d 3: {x.shape}')
        x = self.res3(x)
        #print(f'After res3: {x.shape}')
        
        x = F.relu(self.conv4(x))
        #print(f'After conv4: {x.shape}')
        x = F.max_pool3d(x, 2)
        #print(f'After max_pool3d 4: {x.shape}')
        x = self.res4(x)
        #print(f'After res4: {x.shape}')
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.Conv3d(512, 512, kernel_size=3, padding=1)
        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv2 = nn.Conv3d(512, 256, kernel_size=3, padding=1)
        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv3 = nn.Conv3d(256, 256, kernel_size=3, padding=1)
        self.up3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv4 = nn.Conv3d(256, 128, kernel_size=3, padding=1)
        self.up4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv5 = nn.Conv3d(128, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        self.conv7 = nn.Conv3d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #print(f'After conv1: {x.shape}')
        x = self.up1(x)
        #print(f'After up1 and padding: {x.shape}')
        x = F.relu(self.conv2(x))
        #print(f'After conv2: {x.shape}')
        x = self.up2(x)
        x = F.pad(x, (1, 0, 1, 0, 1, 0))
        #print(f'After up2 and padding: {x.shape}')
        x = F.relu(self.conv3(x))
        #print(f'After conv3: {x.shape}')
        x = self.up3(x)
       # print(f'After up3 and padding: {x.shape}')
        x = F.relu(self.conv4(x))
       # print(f'After conv4: {x.shape}')
        x = self.up4(x)
        x = F.pad(x, (1, 0, 1, 0, 1, 0))
       # print(f'After up4 and padding: {x.shape}')
        x = F.relu(self.conv5(x))
       # print(f'After conv5: {x.shape}')
        x = F.relu(self.conv6(x))
       # print(f'After conv6: {x.shape}')
        x = torch.sigmoid(self.conv7(x))
       # print(f'After conv7: {x.shape}')
        return x

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


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

        return {'input': volume}
    

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
        model.train()
        with tqdm(data_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch") as progress_bar:
            for batch_idx, data in enumerate(progress_bar):
                input_data = data['input'].to(device)
                input_data = input_data.unsqueeze(1)
                optimizer.zero_grad()
                recon_data = model(input_data)
                loss = loss_function(recon_data, input_data)
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
                input_data = input_data.unsqueeze(1)
                recon_data = model(input_data)
                loss = loss_function(recon_data, input_data)
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
    model = Autoencoder()
    model = nn.DataParallel(model)
    #os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    if args.checkpoint_path:
        load_model_weights(model, args.checkpoint_path)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    
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

