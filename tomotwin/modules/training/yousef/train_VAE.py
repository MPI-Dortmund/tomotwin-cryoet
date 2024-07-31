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
import mrcfile



class Flatten(nn.Module):
    def forward(self, input):
        output = input.view(input.size(0), -1)
        return output

class UnFlatten(nn.Module):
    def forward(self, input, size=512):
        output = input.view(input.size(0), size, 8, 8, 8)
        return output

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding='same'),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding='same'),
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
    def __init__(self, in_channels, out_channels):
        super(Up, self).__init__()


        self.up = nn.ConvTranspose3d(in_channels , in_channels , kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1):
        x1 = self.up(x1)

        return self.conv(x1)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Sequential( nn.Conv3d(in_channels, out_channels, kernel_size=3, padding ='same'),
                                  nn.Sigmoid())


    def forward(self, x):
        return self.conv(x)
    
    
class CVAE_3D(nn.Module):
    def __init__(self):
        super(CVAE_3D, self).__init__()
        self.inc = DoubleConv(1, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256,512)
        
        self.mu = nn.Conv3d(512, 512, kernel_size=3, padding='same')
        self.logvar = nn.Conv3d(512, 512, kernel_size=3, padding='same')
        
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64,32)
        self.outc = OutConv(32, 1)
 

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        z = eps.mul(std).add_(mu)
        return z



    def forward(self, x):
        
        x = self.inc(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = self.down4(x)
        
        m = self.mu(x)
        v = self.logvar(x)
        z = self.reparameterize(m,v)
        
        
        x = self.up1(z)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        logits = self.outc(x)

        return logits, m, v, z


class MRCVolumeDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_paths = self._get_file_paths()

    def read_and_normalize_mrc(self, file_path):
        with mrcfile.open(file_path, permissive=True) as mrc:
            data = mrc.data.astype(np.float32)
            min_val = np.min(data)
            max_val = np.max(data)
            normalized_data = (data - min_val) / (max_val - min_val)
        return normalized_data

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
        volume = self.read_and_normalize_mrc(mrc_path)
        #volume = volume[2:-3,2:-3,2:-3]

        return {'input': volume}


def schedule_KL_annealing(start, stop, n_epochs, n_cycle=4, ratio=0.5):
    """
    Custom function for multiple annealing scheduling: Monotonic and cyclical_annealing
    Given number of epochs, it returns the value of the KL weight at each epoch as a list.

    Based on from: https://github.com/haofuml/cyclical_annealing/blob/master/plot/plot_schedules.ipynb
    """

    weights = np.ones(n_epochs)
    period = n_epochs/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v , i = start , 0
        while v <= stop and (int(i+c*period) < n_epochs):
            weights[int(i+c*period)] = v
            v += step
            i += 1

    return weights

def mse(recon_x, x):
    mse_loss = F.mse_loss(recon_x, x)
    return mse_loss

def loss_function(recon_x, x, mu, logvar, kl_weight):

    
    # reconstruction loss (MSE/BCE for image-like data)
    # CE = torch.nn.CrossEntropyLoss()(recon_x, x)
    # MSE = torch.nn.MSELoss(reduction='mean')(recon_x, x)
    MSE = mse(recon_x, x)
    # BCE = F.binary_cross_entropy(recon_x, x, reduction="mean") # only takes data in range [0, 1]
    # BCEL = torch.nn.BCEWithLogitsLoss(reduction="mean")(recon_x, x)

    # KL divergence loss (with annealing)
    # KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()) # sum or mean
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # KLD = KLD * kl_weight

    return MSE + KLD, MSE, KLD


def train(epoch, model, train_loader, kl_weight, optimizer, device, scheduler, args):
    """
    Mini-batch training.
    """

    model.train()
    train_total_loss = 0
    train_BCE_loss = 0
    train_KLD_loss = 0

    print("entered batch training")
    print("train device:", device)
    for batch_idx, data in tqdm(enumerate(train_loader), total=len(train_loader), desc='train'):

        # move data into GPU tensors
        data = data['input'].to(device, dtype=torch.float)
        data = data.unsqueeze(1)
        # reset gradients
        optimizer.zero_grad()

        # call CVAE model
        # feeding 3D volume to Conv3D: https://discuss.pytorch.org/t/feeding-3d-volumes-to-conv3d/32378/6
        recon_batch, mu, logvar, _ = model(data)

        # compute batch losses
        total_loss, BCE_loss, KLD_loss = loss_function(recon_batch, data, mu, logvar, kl_weight)

        train_total_loss += total_loss.item()
        train_BCE_loss += BCE_loss.item()
        train_KLD_loss += KLD_loss.item()

        # compute gradients and update weights
        total_loss.backward()
        optimizer.step()

        # schedule learning rate
        scheduler.step()

    train_total_loss /= len(train_loader.dataset)
    train_BCE_loss /= len(train_loader.dataset)
    train_KLD_loss /= len(train_loader.dataset)

    return train_total_loss, train_BCE_loss, train_KLD_loss


def test(epoch, model, test_loader, reference_batch, kl_weight, writer, device, args):
    """
    Evaluates reconstructions at every epoch (at batch idx 0) by loading test data
    and feeding it through the 3D CVAE.

    TODO: Evaluate generations at every epoch.
    """

    model.eval()
    test_total_loss = 0
    test_BCE_loss = 0
    test_KLD_loss = 0

    # print()
    print("[INFO] entered batch testing")
    print("test device:", device)
    with torch.no_grad():
        for batch_idx, data in tqdm(enumerate(test_loader), total=len(test_loader), desc='test'):
            # forward pass for random batch
            data = data['input'].to(device, dtype=torch.float)
            data=data.unsqueeze(1)
            recon_batch, mu, logvar, latent_batch = model(data)
            total_loss, BCE_loss, KLD_loss = loss_function(recon_batch, data, mu, logvar, kl_weight)

            test_total_loss += total_loss.item()
            test_BCE_loss += BCE_loss.item()
            test_KLD_loss += KLD_loss.item()

    test_total_loss /= len(test_loader.dataset)
    test_BCE_loss /= len(test_loader.dataset)
    test_KLD_loss /= len(test_loader.dataset)

    return test_total_loss, test_BCE_loss, test_KLD_loss

def init_weights(model):
    """
    Set weight initialization for Conv3D in network.
    Based on: https://discuss.pytorch.org/t/how-are-layer-weights-and-biases-initialized-by-default/13073/24
    """
    if isinstance(model, torch.nn.Conv3d):
        torch.nn.init.xavier_uniform_(model.weight)
        torch.nn.init.constant_(model.bias, 0)
        # torch.nn.init.zeros_(model.bias)

def main(args):
    dataset = MRCVolumeDataset(args.dataset_root)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_set = MRCVolumeDataset(args.val_root)
    test_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    device = args.device
    model = CVAE_3D().to(device=device, dtype=torch.float)
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader))
    writer = SummaryWriter(args.logging)
    kl_weights = schedule_KL_annealing(0.0, 1.0, args.num_epochs, 4)
    kl_weight = 0
    ar = None
    best_test_loss = float('inf')
    for epoch in range(0, args.num_epochs):
    #kl_weight = kl_weights[epoch]
        kl_weight = 1
        train_total_loss, train_BCE_loss, train_KLD_loss = train(epoch, model, train_loader, kl_weight, optimizer, device, scheduler, ar)
        writer.add_scalar("train/train_loss", train_total_loss, epoch) # save loss values with writer (dumped into runs/ dir)
        writer.add_scalar("train/BCE_loss", train_BCE_loss, epoch)
        writer.add_scalar("train/KLD_loss", train_KLD_loss, epoch)
        print("Epoch [%d/%d] train_total_loss: %.9f, train_REC_loss: %.9f, train_KLD_loss: %.9f" % (epoch, args.num_epochs, train_total_loss, train_BCE_loss, train_KLD_loss))
        test_total_loss, test_BCE_loss, test_KLD_loss = test(epoch, model, test_loader, args, kl_weight, writer, device, ar)
        writer.add_scalar("test/test_loss", test_total_loss, epoch)
        writer.add_scalar("test/BCE_loss", test_BCE_loss, epoch)
        writer.add_scalar("test/KLD_loss", test_KLD_loss, epoch)
        print("Epoch [%d/%d] test_total_loss: %.9f, test_REC_loss: %.9f, test_KLD_loss: %.9f" % (epoch, args.num_epochs, test_total_loss, test_BCE_loss, test_KLD_loss))
        if test_total_loss < best_test_loss:
            print (f'{test_total_loss} < {best_test_loss}, epoch: {epoch+1} ')
            torch.save(model.state_dict(), f"{args.logging}/weights/model_weights_epoch_{epoch+1}.pt")
            best_test_loss = test_total_loss
        scheduler.step()
    writer.close()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Training script for autoencoder")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--num_workers", type=int, default=18, help="Number of workers for data loader")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for training (cuda or cpu)")
    parser.add_argument("--dataset_root", type=str, default="",  help="Path for you dataset root directory")
    parser.add_argument("--val_root", type=str, default="",  help="Path for you validation dataset root directory")
    parser.add_argument("--logging", type=str, default="", help="Path for you logging directory including /run number")
    parser.add_argument("--checkpoint_path", type=str, default="", help="Path to the saved model checkpoint")
    

    args = parser.parse_args()

    
    main(args)