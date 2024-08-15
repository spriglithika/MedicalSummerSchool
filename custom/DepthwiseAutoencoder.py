import torch
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import matplotlib.pyplot as plt
import torch.utils
import torch.utils.data
from tqdm import tqdm
import numpy as np
import copy
import argparse
from dtu_spine_config import DTUConfig
import os

class DepthWiseDown(nn.Module):
    # Initialize the Down module for downsampling
    def __init__(self, in_c, depth_c, out_c):
        super().__init__()
        # Define a sequence of operations: max pooling followed by two DoubleConv modules
        self.X_depth = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c)
        self.X_point = nn.Conv2d(in_c, depth_c, kernel_size=1)
        self.Y_depth = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c)
        self.Y_point = nn.Conv2d(in_c, depth_c, kernel_size=1)
        self.Z_depth = nn.Conv2d(in_c, in_c, kernel_size=3, padding=1, groups=in_c)
        self.Z_point = nn.Conv2d(in_c, depth_c, kernel_size=1)
        self.total = nn.Conv2d(3*depth_c, out_c, kernel_size=1)

    # Forward pass through the module
    def forward(self, img):
        x = self.X_point(self.X_depth(img))
        y = self.Y_point(self.Y_depth(img.permute(0, 3, 1, 2)))
        z = self.Z_point(self.Z_depth(img.permute(0, 3, 1, 2)))
        t = torch.cat((x, y, z), 1)
        t = self.total(t)
        return t
    
class DepthWiseUp(nn.Module):
    def __init__(self, in_c, depth_c, out_c):
        super().__init__()
        # Define transposed convolutions for upsampling
        self.X_depth = nn.ConvTranspose2d(out_c, out_c, kernel_size=3, padding=1, groups=out_c)
        self.X_point = nn.ConvTranspose2d(depth_c, out_c, kernel_size=1)
        self.Y_depth = nn.ConvTranspose2d(out_c, out_c, kernel_size=3, padding=1, groups=out_c)
        self.Y_point = nn.ConvTranspose2d(depth_c, out_c, kernel_size=1)
        self.Z_depth = nn.ConvTranspose2d(out_c, out_c, kernel_size=3, padding=1, groups=out_c)
        self.Z_point = nn.ConvTranspose2d(depth_c, out_c, kernel_size=1)
        self.total = nn.ConvTranspose2d(in_c, depth_c*3, kernel_size=1)

    def forward(self, x):
        # Split the input tensor into three components
        x = self.total(x)
        x1, x2, x3 = torch.chunk(x, 3, dim=1)
        
        # Apply transposed convolutions to each component
        x1 = self.X_depth(self.X_point(x1))
        x2 = self.Y_depth(self.Y_point(x2))
        x3 = self.Z_depth(self.Z_point(x3))
        
        # Combine the components
        combined = x1 + x2 + x3
        #output = self.total(combined)
        output = F.sigmoid(combined)
        return output
    
class DoubleConv(nn.Module):
    # Initialize the DoubleConv module
    def __init__(self, in_c, out_c, mid_c=None, residual=False):
        super().__init__()
        self.residual = residual  # Whether to use residual connections
        if not mid_c:
            mid_c = out_c  # If mid_c is not provided, use out_c
        # Define a sequence of operations: two convolutional layers with normalization and activation
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_c, mid_c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_c),
            nn.GELU(),
            nn.Conv2d(mid_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_c)
        )

    # Forward pass through the module
    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))  # Apply residual connection if enabled
        else:
            return self.double_conv(x)  # Otherwise, just pass through the double convolution

class Down(nn.Module):
    # Initialize the Down module for downsampling
    def __init__(self, in_c, out_c,padding=0):
        super().__init__()
        # Define a sequence of operations: max pooling followed by two DoubleConv modules
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(3,padding=padding),
            DoubleConv(in_c, in_c, residual=True),
            DoubleConv(in_c, out_c)
        )

    # Forward pass through the module
    def forward(self, x):
        x = self.maxpool_conv(x)  # Apply max pooling and convolutions
        return x

class Up(nn.Module):
    # Initialize the Up module for upsampling
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Define a sequence of operations: upsampling followed by two DoubleConv modules
        self.upconv = nn.Sequential(
            nn.Upsample(scale_factor=3, mode='bilinear', align_corners=True),
            DoubleConv(in_channels, in_channels),
            DoubleConv(in_channels, out_channels, in_channels // 2)
        )

    # Forward pass through the module
    def forward(self, x):
        return self.upconv(x)  # Apply upsampling and convolutions

class DepthwiseAutoEncoder(nn.Module):
    # Initialize the AutoEncoder module
    def __init__(self):
        super().__init__()
        channel_list = [64, 128, 256, 512]  # Define the channel sizes
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Determine the device
        # Define the encoder part of the autoencoder
        flat = 81
        unflat = (1, 9, 9)
        self.encoder = nn.Sequential(
            DepthWiseDown(243, 81, channel_list[0]),
            Down(channel_list[0], channel_list[1]),
            Down(channel_list[1], channel_list[2]),
            Down(channel_list[2], channel_list[3]),
            DoubleConv(channel_list[3], 1), nn.Flatten(), nn.Linear(flat, 64)
        )
        # Define the decoder part of the autoencoder
        self.decoder = nn.Sequential(
            nn.Linear(64, flat), nn.Unflatten(-1, unflat),
            DoubleConv(1, channel_list[3]),
            Up(channel_list[3], channel_list[2]),
            Up(channel_list[2], channel_list[1]),
            Up(channel_list[1], channel_list[0]),
            DepthWiseUp(channel_list[0], 81, 243)
        )

    # Forward pass through the autoencoder
    #@torch.jit.script
    def forward(self, x):
        x = self.encoder(x)
        return (self.decoder(x))  # Encode, then decode the input

class DepthwiseVariationalAutoEncoder(nn.Module):
    # Initialize the AutoEncoder module
    def __init__(self, latent_dim=512):
        super().__init__()
        channel_list = [64, 128, 256, 512]  # Define the channel sizes
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Determine the device
        # Define the encoder part of the autoencoder
        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.to(self.device) 
        self.N.scale = self.N.scale.to(self.device)
        self.kl = 0
        flat = 81
        unflat = (1, 9, 9)
        self.linear_mu = nn.Linear(flat, latent_dim)
        self.linear_sigma = nn.Linear(flat, latent_dim)
        self.encoder = nn.Sequential(
            DepthWiseDown(243, 81, channel_list[0]),
            Down(channel_list[0], channel_list[1]),
            Down(channel_list[1], channel_list[2]),
            Down(channel_list[2], channel_list[3]),
            DoubleConv(channel_list[3], 1), nn.Flatten()
        )
        # Define the decoder part of the autoencoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, flat), nn.Unflatten(-1, unflat),
            DoubleConv(1, channel_list[3]),
            Up(channel_list[3], channel_list[2]),
            Up(channel_list[2], channel_list[1]),
            Up(channel_list[1], channel_list[0]),
            DepthWiseUp(channel_list[0], 81, 243)
        )
    # Forward pass through the autoencoder
    #@torch.jit.script
    def forward(self, x):
        x = self.encoder(x)
        mu = self.linear_mu(x)
        sigma = torch.exp(self.linear_sigma(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return self.decoder(z)  # Encode, then decode the input
    

class DepthCheck(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = DepthWiseDown(243, 81, 64)
        self.up = DepthWiseUp(64, 81, 243)
    def forward(self, x):
        x = self.up(self.down(x))
        return x

def select_spine_label(label):
    # Set all non-spine labels to 0 and spine labels to 1  
    return torch.where(label == 20, torch.ones_like(label).to(label.device), torch.zeros_like(label).to(label.device))

class CTSpineDataset(torch.utils.data.Dataset):
    def __init__(self, settings, split, anomolies=None, right_labels=False):
        self.imgs = []
        self.labels = []
        self.anomolies = []
        self.right_labels = right_labels
        data_dir = settings["data_dir"]
        test_list = settings["data_set"]
        result_dir = settings["result_dir"]
        scan_file_path = os.path.join(result_dir, test_list)
        self.crop_dir = os.path.join(data_dir, split, "crops")
        anomoly_list = ['','_sphere_outlier_mean_std_inpaint','_sphere_outlier_water','_warp_outlier']
        with open(scan_file_path, 'r') as f:
            all_scan_ids = f.readlines()
        for scan_id in all_scan_ids:
            scan_id = scan_id.strip()
            for anomoly in anomolies:
                img_path = os.path.join(self.crop_dir, f"{scan_id}_crop{anomoly_list[anomoly]}.nii.gz")
                if right_labels and anomolies != None:
                    label_path = os.path.join(self.crop_dir, f"{scan_id}_crop_label{anomoly_list[anomoly]}.nii.gz")
                elif anomolies != None:
                    label_path = os.path.join(self.crop_dir, f"{scan_id}_crop_label{anomoly_list[0]}.nii.gz")
                else:
                    label_path = None
                self.imgs.append(img_path)
                self.labels.append(label_path)
                self.anomolies.append(anomoly)
    def __len__(self):
        return len(self.imgs)
    def __getitem__(self, idx): 
        img = F.pad(torch.tensor(sitk.GetArrayFromImage(sitk.ReadImage(self.imgs[idx])).astype(float)), (1, 1, 1, 1, 1, 1))
        label = F.pad(torch.tensor(sitk.GetArrayFromImage(sitk.ReadImage(self.labels[idx])).astype(float)), (1, 1, 1, 1, 1, 1))
        anomaly = self.anomolies[idx]
        label = select_spine_label(label)
        return img, label,  anomaly
    

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='test-segmentation-outlier-detection')
    config = DTUConfig(args)
    ## SETUP
    anomolies = [0,1,2,3]
    train_on_normal_versions = False

    train_dataset = torch.utils.data.DataLoader(CTSpineDataset(config.settings, split='train', anomolies=anomolies, right_labels=train_on_normal_versions), batch_size=2, shuffle=True)
    #test_dataset = torch.utils.data.DataLoader(CTSpineDataset(config.settings, split='test', anomolies=anomolies, right_labels=not train_on_normal_versions), batch_size=2, shuffle=True)
    test_dataset = torch.utils.data.DataLoader(CTSpineDataset(config.settings, split='test', anomolies=[0], right_labels=train_on_normal_versions), batch_size=2, shuffle=True)
    model = DepthwiseVariationalAutoEncoder()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    ## Train
    loss_hist = []
    pbar = tqdm(range(10))
    for epoch in pbar:
        for img, label,  _ in train_dataset:
            img, label = img.float(), label.float()
            pred = model(img)
            opt.zero_grad()
            loss = F.mse_loss(pred, label) + .01 * model.kl
            loss.backward()
            opt.step()
            loss_hist.append(loss.item())
            pbar.set_postfix(Loss=loss.item())
    ## EVAL
    dists = []
    anoms = []

    for batch in test_dataset:

        imgs, labels, anomolies = batch

        img, label = img.float(), label.float()
        preds = model(imgs)
        for img, label, pred, anomoly in zip(imgs, labels, preds, anomolies):
            
            distance = F.mse_loss(pred, label)
            dists.append(distance.item())  # Convert tensor to scalar and append
            anoms.append(anomoly.item())  # Convert to integer and append

    dists = np.array(dists)
    anoms = np.array(anoms, dtype=np.bool)  # Ensure anoms is a boolean tensor

    anomolous = dists[anoms]
    normal = dists[~anoms]
    print(f'Training Anomolous: {anomolous.mean()}')
    print(f'Training Normal: {normal.mean()}')

    plt.hist(anomolous, bins=10, alpha=.5, label='Training Anomolous')
    plt.hist(normal, bins=10, alpha=.5, label='Training Normal')
    plt.legend()
    # plt.plot(list(range(len(loss_hist))), loss_hist)
    plt.show()