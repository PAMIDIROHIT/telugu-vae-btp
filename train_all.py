#!/home/mohanganesh/vae_env/bin/python
"""
Standalone VAE Training Script for Telugu Glyphs
Installs PyTorch, trains 4 models
"""

import subprocess
import sys
import os

# Change to project directory
os.chdir('/home/mohanganesh/vae_project')

# Step 1: Install PyTorch
print("=" * 70)
print("Step 1: Installing PyTorch")
print("=" * 70)
try:
    subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '--no-cache-dir',
         'torch', 'torchvision', 'torchaudio',
         '--index-url', 'https://download.pytorch.org/whl/cu118', '-q'],
        check=True
    )
    print("✓ PyTorch installed\n")
except Exception as e:
    print(f"✗ Failed to install PyTorch: {e}")
    sys.exit(1)

# Step 2: Verify PyTorch
print("Step 2: Verifying PyTorch")
print("=" * 70)
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("✗ CUDA not available!")
        sys.exit(1)
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")
    sys.exit(1)

# Step 3: Verify dataset
print("\nStep 3: Verifying Dataset")
print("=" * 70)
from pathlib import Path
images = list(Path('data/raw').glob('**/*.png'))
print(f"✓ Found {len(images)} images")
if len(images) < 10800:
    print(f"✗ ERROR: Expected 10800 images!")
    sys.exit(1)

# Step 4: Start training
print("\nStep 4: Starting Model Training")
print("=" * 70)
print(f"Training 4 models with ~100 epochs each")
print(f"Estimated time: 8-12 hours\n")

# Import training dependencies
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import json
from datetime import datetime
from tqdm import tqdm

sys.path.insert(0, '/home/mohanganesh/vae_project')
from models.vae import VanillaVAE, BetaVAE, ConditionalVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TeluguDataset(Dataset):
    def __init__(self, split='train'):
        images = sorted(Path('data/raw').glob('**/*.png'))
        n = len(images)
        idx_train = int(n * 0.8)
        idx_val = idx_train + int(n * 0.1)
        
        if split == 'train':
            self.images = images[:idx_train]
        elif split == 'val':
            self.images = images[idx_train:idx_val]
        else:
            self.images = images[idx_val:]
        
        print(f"  {split.upper()}: {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('L')
        img = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)
        return img

def train_vae(model_name, model, beta=1.0, epochs=100):
    print(f"\n{'='*70}")
    print(f"Training: {model_name} (β={beta}, {epochs} epochs)")
    print(f"{'='*70}\n")
    
    model.to(device)
    
    tr_ds = TeluguDataset('train')
    va_ds = TeluguDataset('val')
    
    tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    exp_dir = Path(f'experiments/{model_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)
    
    history = {'epoch': [], 'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss_total = 0
        
        for x in tr_loader:
            x = x.to(device)
            recon, mu, logvar = model(x)
            
            mse = nn.MSELoss()(recon, x)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = mse + beta * kl
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss_total += loss.item()
        
        train_loss_avg = train_loss_total / len(tr_loader)
        scheduler.step()
        
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_loss_total = 0
            
            with torch.no_grad():
                for x in va_loader:
                    x = x.to(device)
                    recon, mu, logvar = model(x)
                    
                    mse = nn.MSELoss()(recon, x)
                    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = mse + beta * kl
                    
                    val_loss_total += loss.item()
            
            val_loss_avg = val_loss_total / len(va_loader)
            
            history['epoch'].append(epoch + 1)
            history['train_loss'].append(train_loss_avg)
            history['val_loss'].append(val_loss_avg)
            
            improved = "↓" if val_loss_avg < best_val_loss else "↑"
            print(f"E{epoch+1:3d}/{epochs} | Train: {train_loss_avg:.6f} | Val: {val_loss_avg:.6f} {improved}")
            
            if val_loss_avg < best_val_loss:
                best_val_loss = val_loss_avg
                patience = 0
                torch.save(model.state_dict(), exp_dir / 'checkpoints' / 'best.pth')
            else:
                patience += 1
                if patience >= 5:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        else:
            print(f"E{epoch+1:3d}/{epochs} | Train: {train_loss_avg:.6f}")
    
    torch.save(model.state_dict(), exp_dir / 'checkpoints' / 'final.pth')
    
    with open(exp_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\n✓ Saved to {exp_dir}")
    return str(exp_dir)

# Execute training
Path('experiments').mkdir(exist_ok=True)
results = {}

m1 = VanillaVAE(latent_dim=10, in_channels=1)
results['vanilla_vae'] = train_vae('vanilla_vae', m1, beta=1.0, epochs=100)

m2 = BetaVAE(beta=1.0, latent_dim=10, in_channels=1)
results['beta_vae_b1'] = train_vae('beta_vae_b1', m2, beta=1.0, epochs=100)

m3 = BetaVAE(beta=5.0, latent_dim=10, in_channels=1)
results['beta_vae_b5'] = train_vae('beta_vae_b5', m3, beta=5.0, epochs=100)

m4 = ConditionalVAE(num_classes=72, latent_dim=10, in_channels=1)
results['conditional_vae'] = train_vae('conditional_vae', m4, beta=1.0, epochs=100)

with open('experiments/training_summary.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*70}")
print("✓✓✓ All training complete! ✓✓✓")
print(f"{'='*70}\n")

for name, path in results.items():
    print(f"  {name}: {path}")

print(f"\nResults saved to: experiments/")
