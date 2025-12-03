#!/home/mohanganesh/vae_env/bin/python
"""Quick training test - 5 epochs to verify setup, then full training"""

import sys
sys.path.insert(0, '/home/mohanganesh/vae_project')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
import numpy as np

# Disable cuDNN to avoid initialization issues
torch.backends.cudnn.enabled = False

print("=" * 70)
print("CUDA Verification")
print("=" * 70)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    for i in range(min(2, torch.cuda.device_count())):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("ERROR: CUDA not available")
    sys.exit(1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nUsing device: {device}\n")

print("=" * 70)
print("Quick Dataset Test")
print("=" * 70)

class SimpleDataset(Dataset):
    def __init__(self, split='train', limit=100):
        images = sorted(Path('/home/mohanganesh/vae_project/data/raw').glob('**/*.png'))[:limit]
        n = len(images)
        idx_train = int(n * 0.8)
        if split == 'train':
            self.images = images[:idx_train]
        else:
            self.images = images[idx_train:]
        print(f"{split.upper()}: {len(self.images)} images")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert('L')
        img = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)
        return img

tr_ds = SimpleDataset('train', limit=100)
te_ds = SimpleDataset('test', limit=100)
tr_loader = DataLoader(tr_ds, batch_size=16, shuffle=True)
te_loader = DataLoader(te_ds, batch_size=16)

print(f"DataLoaders created: {len(tr_loader)} batches\n")

print("=" * 70)
print("Testing VAE Training (5 quick epochs)")
print("=" * 70)

from models.vae import VanillaVAE

model = VanillaVAE(latent_dim=10, in_channels=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(5):
    model.train()
    total_loss = 0
    for x in tr_loader:
        x = x.to(device)
        recon, mu, logvar = model(x)
        
        mse = nn.MSELoss()(recon, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        loss = mse + kl
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(tr_loader)
    print(f"Epoch {epoch+1}/5 | Loss: {avg_loss:.4f}")

print("\n✓ Test training successful!")
print("Ready to proceed with full training")
