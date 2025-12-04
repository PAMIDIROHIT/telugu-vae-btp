"""
Complete Research Training Pipeline
Trains VAE models, generates samples, evaluates similarity, creates plots
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image, make_grid
import os
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
import shutil

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.vae import BetaVAE, ConditionalVAE

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Hyperparameters
CONFIG = {
    'batch_size': 16,
    'learning_rate': 1e-3,
    'epochs': 100,
    'latent_dim': 64,
    'beta': 4.0,
    'num_classes': 12,
    'image_size': 32,
    'save_interval': 10,
}

# Paths
DATA_TRAIN = 'data/train'
DATA_TEST = 'data/test'
RESULTS_DIR = 'results_research'
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, 'checkpoints')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
SAMPLES_DIR = os.path.join(RESULTS_DIR, 'generated_samples')
LOGS_DIR = os.path.join(RESULTS_DIR, 'logs')

# Create directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

#=============================================================================
# DATASET
#=============================================================================

class TeluguDataset(Dataset):
    """Telugu character dataset"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        
        class_dirs = sorted([d for d in os.listdir(root_dir)
                           if os.path.isdir(os.path.join(root_dir, d))
                           and d.startswith('class_')])
        
        for class_dir in class_dirs:
            class_idx = int(class_dir.split('_')[1])
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
# TRAINING
#=============================================================================

def train_epoch(model, dataloader, optimizer, epoch, is_conditional=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{CONFIG["epochs"]}')
    
    for images, labels in pbar:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        if is_conditional:
            recon, mu, logvar, _ = model(images, labels)
        else:
            recon, mu, logvar, _ = model(images)
        
        loss, recon_loss, kl_loss = model.loss(recon, images, mu, logvar)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item()/len(images):.4f}',
            'recon': f'{recon_loss.item()/len(images):.4f}',
            'kl': f'{kl_loss.item()/len(images):.4f}'
        })
    
    n_samples = len(dataloader.dataset)
    return total_loss/n_samples, total_recon/n_samples, total_kl/n_samples

def evaluate(model, dataloader, is_conditional=False):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            
            if is_conditional:
                recon, mu, logvar, _ = model(images, labels)
            else:
                recon, mu, logvar, _ = model(images)
            
            loss, recon_loss, kl_loss = model.loss(recon, images, mu, logvar)
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
    
    n_samples = len(dataloader.dataset)
    return total_loss/n_samples, total_recon/n_samples, total_kl/n_samples

#=============================================================================
# MAIN TRAINING LOOP
#=============================================================================

def main():
    print("=" * 70)
    print("TELUGU VAE RESEARCH TRAINING PIPELINE")
    print("=" * 70)
    print()
    
    # Save configuration
    with open(os.path.join(RESULTS_DIR, 'config.json'), 'w') as f:
        json.dump(CONFIG, f, indent=2)
    
    print("Configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    print()
    
    # Data transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = TeluguDataset(DATA_TRAIN, transform=transform)
    test_dataset = TeluguDataset(DATA_TEST, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'],
                            shuffle=False, num_workers=4)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print()
    
    #=========================================================================
    # EXPERIMENT  1: Beta-VAE
    #=========================================================================
    
    print("=" * 70)
    print("EXPERIMENT 1: Beta-VAE Training")
    print("=" * 70)
    print()
    
    model = BetaVAE(
        latent_dim=CONFIG['latent_dim'],
        beta=CONFIG['beta'],
        in_channels=1
    ).to(DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Training history
    history = {
        'train_loss': [],
        'train_recon': [],
        'train_kl': [],
        'test_loss': [],
        'test_recon': [],
        'test_kl': []
    }
    
    best_test_loss = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        # Train
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, epoch, is_conditional=False
        )
        
        # Evaluate
        test_loss, test_recon, test_kl = evaluate(model, test_loader, is_conditional=False)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_recon'].append(train_recon)
        history['train_kl'].append(train_kl)
        history['test_loss'].append(test_loss)
        history['test_recon'].append(test_recon)
        history['test_kl'].append(test_kl)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}")
        
        # Save best model
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_loss': test_loss,
                'config': CONFIG
            }, os.path.join(CHECKPOINT_DIR, 'best_beta_vae.pth'))
        
        # Save checkpoint periodically
        if (epoch + 1) % CONFIG['save_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'history': history
            }, os.path.join(CHECKPOINT_DIR, f'beta_vae_epoch_{epoch+1}.pth'))
    
    # Save final model and history
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'final_beta_vae.pth'))
    with open(os.path.join(LOGS_DIR, 'beta_vae_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    
    print()
    print("Beta-VAE training complete!")
    print(f"Best test loss: {best_test_loss:.4f}")
    print()
    
    # Plot training curves
    plot_training_curves(history, 'Beta-VAE')
    
    #=========================================================================
    # EXPERIMENT 2: Conditional VAE
    #=========================================================================
    
    print("=" * 70)
    print("EXPERIMENT 2: Conditional VAE Training")
    print("=" * 70)
    print()
    
    model_cvae = ConditionalVAE(
        latent_dim=CONFIG['latent_dim'],
        num_classes=CONFIG['num_classes'],
        in_channels=1
    ).to(DEVICE)
    
    optimizer_cvae = optim.Adam(model_cvae.parameters(), lr=CONFIG['learning_rate'])
    
    history_cvae = {
        'train_loss': [],
        'train_recon': [],
        'train_kl': [],
        'test_loss': [],
        'test_recon': [],
        'test_kl': []
    }
    
    best_test_loss_cvae = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        # Train
        train_loss, train_recon, train_kl = train_epoch(
            model_cvae, train_loader, optimizer_cvae, epoch, is_conditional=True
        )
        
        # Evaluate
        test_loss, test_recon, test_kl = evaluate(model_cvae, test_loader, is_conditional=True)
        
        # Save history
        history_cvae['train_loss'].append(train_loss)
        history_cvae['train_recon'].append(train_recon)
        history_cvae['train_kl'].append(train_kl)
        history_cvae['test_loss'].append(test_loss)
        history_cvae['test_recon'].append(test_recon)
        history_cvae['test_kl'].append(test_kl)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}")
        
        # Save best model
        if test_loss < best_test_loss_cvae:
            best_test_loss_cvae = test_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_cvae.state_dict(),
                'optimizer_state_dict': optimizer_cvae.state_dict(),
                'test_loss': test_loss,
                'config': CONFIG
            }, os.path.join(CHECKPOINT_DIR, 'best_cvae.pth'))
        
        # Save checkpoint periodically
        if (epoch + 1) % CONFIG['save_interval'] == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model_cvae.state_dict(),
                'optimizer_state_dict': optimizer_cvae.state_dict(),
                'history': history_cvae
            }, os.path.join(CHECKPOINT_DIR, f'cvae_epoch_{epoch+1}.pth'))
    
    # Save final model and history
    torch.save(model_cvae.state_dict(), os.path.join(CHECKPOINT_DIR, 'final_cvae.pth'))
    with open(os.path.join(LOGS_DIR, 'cvae_history.json'), 'w') as f:
        json.dump(history_cvae, f, indent=2)
    
    print()
    print("Conditional VAE training complete!")
    print(f"Best test loss: {best_test_loss_cvae:.4f}")
    print()
    
    # Plot training curves
    plot_training_curves(history_cvae, 'Conditional VAE')
    
    print("=" * 70)
    print("ALL TRAINING COMPLETE!")
    print("=" * 70)

def plot_training_curves(history, model_name):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Total loss
    axes[0].plot(epochs, history['train_loss'], label='Train')
    axes[0].plot(epochs, history['test_loss'], label='Test')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Total Loss')
    axes[0].set_title(f'{model_name}: Total Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[1].plot(epochs, history['train_recon'], label='Train')
    axes[1].plot(epochs, history['test_recon'], label='Test')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Reconstruction Loss')
    axes[1].set_title(f'{model_name}: Reconstruction Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # KL divergence
    axes[2].plot(epochs, history['train_kl'], label='Train')
    axes[2].plot(epochs, history['test_kl'], label='Test')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('KL Divergence')
    axes[2].set_title(f'{model_name}: KL Divergence')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR,  f'{model_name.lower().replace(" ", "_")}_training_curves.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    main()
