"""
Main training script for VAE models on Telugu glyphs.
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import create_model
from scripts.utils import (
    set_seed, load_config, save_config, create_experiment_dir,
    save_checkpoint, load_checkpoint, Logger, get_git_hash, 
    save_git_info, print_model_summary
)


class TeluguGlyphDataset(torch.utils.data.Dataset):
    """Dataset for Telugu glyphs."""
    
    def __init__(self, image_dir, metadata_path, split='train', split_ratio=0.8):
        """
        Args:
            image_dir: Directory containing glyph images
            metadata_path: Path to metadata CSV
            split: 'train', 'val', or 'test'
            split_ratio: Fraction for training set
        """
        import csv
        from PIL import Image
        
        self.image_dir = image_dir
        self.split = split
        
        # Load metadata
        self.samples = []
        with open(metadata_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row)
        
        # Split dataset
        n_total = len(self.samples)
        n_train = int(n_total * split_ratio)
        n_val = int(n_total * 0.1)
        
        if split == 'train':
            self.samples = self.samples[:n_train]
        elif split == 'val':
            self.samples = self.samples[n_train:n_train + n_val]
        else:  # test
            self.samples = self.samples[n_train + n_val:]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        from PIL import Image
        
        sample = self.samples[idx]
        img_path = sample['filename']
        
        # Load image
        img = Image.open(img_path).convert('L')
        img = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        img = torch.from_numpy(img).unsqueeze(0)  # Add channel dimension
        
        return img


def train_epoch(model, train_loader, optimizer, device, beta=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for batch_idx, x in enumerate(pbar):
        x = x.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        if hasattr(model, 'forward'):
            output = model(x)
            if len(output) == 4:  # recon, mu, logvar, z
                recon_x, mu, logvar, z = output
            else:
                recon_x, mu, logvar = output
        
        # Compute loss
        # BetaVAE uses self.beta internally, VanillaVAE can accept beta parameter
        if model.__class__.__name__ == 'BetaVAE':
            loss, recon_loss, kl_loss = model.loss(recon_x, x, mu, logvar)
        else:
            loss, recon_loss, kl_loss = model.loss(recon_x, x, mu, logvar, beta=beta)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'recon': f'{recon_loss.item():.4f}',
            'kl': f'{kl_loss.item():.4f}'
        })
    
    avg_loss = total_loss / len(train_loader)
    avg_recon = total_recon / len(train_loader)
    avg_kl = total_kl / len(train_loader)
    
    return avg_loss, avg_recon, avg_kl


@torch.no_grad()
def validate(model, val_loader, device, beta=1.0):
    """Validate model."""
    model.eval()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    for x in tqdm(val_loader, desc='Validation'):
        x = x.to(device)
        
        output = model(x)
        if len(output) == 4:
            recon_x, mu, logvar, z = output
        else:
            recon_x, mu, logvar = output
        
        if model.__class__.__name__ == 'BetaVAE':
            loss, recon_loss, kl_loss = model.loss(recon_x, x, mu, logvar)
        else:
            loss, recon_loss, kl_loss = model.loss(recon_x, x, mu, logvar, beta=beta)
        
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
    
    avg_loss = total_loss / len(val_loader)
    avg_recon = total_recon / len(val_loader)
    avg_kl = total_kl / len(val_loader)
    
    return avg_loss, avg_recon, avg_kl


def main():
    parser = argparse.ArgumentParser(description="Train VAE on Telugu glyphs")
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--experiment-name', type=str, required=True,
                       help='Name for this experiment')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    set_seed(config.get('seed', 42))
    
    # Setup
    device = torch.device(config['training']['device'])
    exp_dir = create_experiment_dir(config['logging']['experiment_dir'], 
                                    args.experiment_name)
    
    print(f"Experiment directory: {exp_dir}")
    
    # Save config and git info
    save_config(config, os.path.join(exp_dir, 'config.yaml'))
    git_hash = get_git_hash()
    save_git_info(git_hash, os.path.join(exp_dir, 'git_info.txt'))
    
    # Load dataset
    train_dataset = TeluguGlyphDataset(
        config['data']['dataset_path'],
        config['data']['metadata_path'],
        split='train',
        split_ratio=config['data']['train_split']
    )
    
    val_dataset = TeluguGlyphDataset(
        config['data']['dataset_path'],
        config['data']['metadata_path'],
        split='val',
        split_ratio=config['data']['train_split']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers']
    )
    
    # Create model
    model = create_model(
        config['model']['type'],
        latent_dim=config['model']['latent_dim'],
        beta=config['model']['beta'],
        in_channels=config['model']['in_channels']
    )
    model.to(device)
    print_model_summary(model)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    
    # Learning rate scheduler
    if config['training']['lr_scheduler'] == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['num_epochs']
        )
    else:
        scheduler = None
    
    # Logger
    logger = Logger(os.path.join(exp_dir, 'logs', 'metrics.csv'))
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['training']['num_epochs']):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
        print(f"{'='*50}")
        
        # Train
        train_loss, train_recon, train_kl = train_epoch(
            model, train_loader, optimizer, device,
            beta=config['model']['beta']
        )
        
        # Validate
        if (epoch + 1) % config['logging']['validation_interval'] == 0:
            val_loss, val_recon, val_kl = validate(
                model, val_loader, device,
                beta=config['model']['beta']
            )
            
            print(f"Val Loss: {val_loss:.4f} | Recon: {val_recon:.4f} | KL: {val_kl:.4f}")
            
            # Log metrics
            logger.log(
                epoch,
                train_loss=train_loss,
                train_recon=train_recon,
                train_kl=train_kl,
                val_loss=val_loss,
                val_recon=val_recon,
                val_kl=val_kl
            )
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save best checkpoint
                checkpoint_path = os.path.join(exp_dir, 'checkpoints', 'best.pth')
                save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= config['training']['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # Save checkpoint
        if (epoch + 1) % config['logging']['checkpoint_interval'] == 0:
            checkpoint_path = os.path.join(
                exp_dir, 'checkpoints', f'checkpoint_epoch{epoch+1}.pth'
            )
            save_checkpoint(model, optimizer, epoch, train_loss, checkpoint_path)
        
        # Learning rate step
        if scheduler is not None:
            scheduler.step()
    
    print(f"\nTraining complete! Results saved to {exp_dir}")


if __name__ == '__main__':
    main()
