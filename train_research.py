"""
Comprehensive VAE Training Script for Telugu Character Generation
Supports multiple architectures and extensive evaluation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

#=============================================================================
# DATASET
#=============================================================================

class TeluguDataset(Dataset):
    """Telugu character dataset"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        # Load all images
        class_dirs = sorted([d for d in os.listdir(root_dir)
                           if os.path.isdir(os.path.join(root_dir, d))
                           and d.startswith('class_')])
        
        for class_dir in class_dirs:
            class_idx = int(class_dir.split('_')[1])
            self.class_to_idx[class_dir] = class_idx
            
            class_path = os.path.join(root_dir, class_dir)
            images = [f for f in os.listdir(class_path) if f.endswith('.png')]
            
            for img_name in images:
                img_path = os.path.join(class_path, img_name)
                self.samples.append((img_path, class_idx))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

#=============================================================================
# VAE ARCHITECTURES
#=============================================================================

class Encoder(nn.Module):
    """Encoder network"""
    def __init__(self, latent_dim=64, input_channels=1):
        super(Encoder, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, 4, stride=2, padding=1)  # 32 -> 16
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2, padding=1)              # 16 -> 8
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2, padding=1)             # 8 -> 4
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2, padding=1)            # 4 -> 2
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Latent layers
        self.fc_mu = nn.Linear(256 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(256 * 2 * 2, latent_dim)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar

class Decoder(nn.Module):
    """Decoder network"""
    def __init__(self, latent_dim=64, output_channels=1):
        super(Decoder, self).__init__()
        
        self.fc = nn.Linear(latent_dim, 256 * 2 * 2)
        
        # Deconvolutional layers
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)  # 2 -> 4
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)   # 4 -> 8
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)    # 8 -> 16
        self.deconv4 = nn.ConvTranspose2d(32, output_channels, 4, stride=2, padding=1)  # 16 -> 32
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)
    
    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 256, 2, 2)
        
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = torch.sigmoid(self.deconv4(x))
        
        return x

class VAE(nn.Module):
    """Variational Autoencoder"""
    def __init__(self, latent_dim=64, beta=1.0):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.beta = beta
        
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar
    
    def sample(self, num_samples, device):
        """Generate samples from prior distribution"""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decoder(z)
        return samples

class ConditionalVAE(nn.Module):
    """Conditional VAE with class information"""
    def __init__(self, latent_dim=64, num_classes=12, beta=1.0):
        super(ConditionalVAE, self).__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.beta = beta
        
        # Embedding for class labels
        self.label_embedding = nn.Embedding(num_classes, latent_dim)
        
        # Encoder (takes image input)
        self.encoder = Encoder(latent_dim)
        
        # Decoder (takes latent + class embedding)
        self.fc_decode = nn.Linear(latent_dim * 2, 256 * 2 * 2)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1)
        
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(32)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, labels):
        """Decode with class conditioning"""
        label_embed = self.label_embedding(labels)
        z_combined = torch.cat([z, label_embed], dim=1)
        
        x = self.fc_decode(z_combined)
        x = x.view(x.size(0), 256, 2, 2)
        
        x = F.relu(self.bn1(self.deconv1(x)))
        x = F.relu(self.bn2(self.deconv2(x)))
        x = F.relu(self.bn3(self.deconv3(x)))
        x = torch.sigmoid(self.deconv4(x))
        
        return x
    
    def forward(self, x, labels):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, labels)
        return recon, mu, logvar
    
    def sample(self, num_samples, labels, device):
        """Generate samples for specific classes"""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        samples = self.decode(z, labels)
        return samples

#=============================================================================
# LOSS FUNCTIONS
#=============================================================================

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """VAE loss = Reconstruction + Beta * KL Divergence"""
    # Reconstruction loss (BCE)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

#=============================================================================
# CONTINUE IN NEXT FILE...
#=============================================================================
