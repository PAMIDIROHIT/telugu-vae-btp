"""
Complete Evaluation and Analysis Script
- Generate 10 samples per class
- Analyze similarity between original and generated
- Create all visualizations for research paper
- Generate comprehensive research report
"""

import torch
import torch.nn as nn
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
from sklearn.metrics.pairwise import cosine_similarity
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')

def calculate_ssim_manual(img1, img2):
    """Manual SSIM calculation"""
    img1_np = img1.cpu().numpy().squeeze()
    img2_np = img2.cpu().numpy().squeeze()
    
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    
    mu1 = img1_np.mean()
    mu2 = img2_np.mean()
    sigma1 = img1_np.std()
    sigma2 = img2_np.std()
    sigma12 = ((img1_np - mu1) * (img2_np - mu2)).mean()
    
    ssim_val = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 ** 2 + sigma2 ** 2 + c2))
    
    return float(ssim_val)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.vae import BetaVAE, ConditionalVAE

#=============================================================================
# CONFIGURATION
#=============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = False

CONFIG = {
    'latent_dim': 64,
    'beta': 4.0,
    'num_classes': 12,
    'samples_per_class': 10,
}

# Paths
DATA_TEST = 'data/test'
CHECKPOINT_DIR = 'results_research/checkpoints'
RESULTS_DIR = 'results_research'
SAMPLES_DIR = os.path.join(RESULTS_DIR, 'generated_samples')
ANALYSIS_DIR = os.path.join(RESULTS_DIR, 'analysis')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')

# Create directories
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)
for class_id in range(CONFIG['num_classes']):
    os.makedirs(os.path.join(SAMPLES_DIR, f'class_{str(class_id).zfill(3)}'), exist_ok=True)

#=============================================================================
# DATASET
#=============================================================================

class TeluguDataset(Dataset):
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
        
        return image, label, img_path

#=============================================================================
# SAMPLE GENERATION
#=============================================================================

def generate_samples(model, num_samples, class_id, is_conditional=False):
    """Generate samples from the model"""
    model.eval()
    with torch.no_grad():
        if is_conditional:
            labels = torch.full((num_samples,), class_id, dtype=torch.long, device=DEVICE)
            samples = model.sample(num_samples, class_id, device=DEVICE)
        else:
            samples = model.sample(num_samples, device=DEVICE)
    
    return samples

def save_samples_grid(samples, save_path, nrow=10):
    """Save samples as a grid"""
    save_image(samples, save_path, nrow=nrow, normalize=True, padding=2)

#=============================================================================
# SIMILARITY METRICS
#=============================================================================

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images"""
    return calculate_ssim_manual(img1, img2)

def calculate_mse(img1, img2):
    """Calculate MSE between two images"""
    return torch.mean((img1 - img2) ** 2).item()

def calculate_cosine_similarity(img1, img2):
    """Calculate cosine similarity"""
    img1_flat = img1.cpu().numpy().flatten().reshape(1, -1)
    img2_flat = img2.cpu().numpy().flatten().reshape(1, -1)
    return cosine_similarity(img1_flat, img2_flat)[0][0]

#=============================================================================
# MAIN EVALUATION
#=============================================================================

def main():
    print("=" * 70)
    print("TELUGU VAE EVALUATION AND ANALYSIS")
    print("=" * 70)
    print()
    
    # Load test dataset
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = TeluguDataset(DATA_TEST, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    print(f"Test samples: {len(test_dataset)}")
    print()
    
    #=========================================================================
    # LOAD MODELS
    #=========================================================================
    
    print("Loading trained models...")
    
    # Load Beta-VAE
    beta_vae = BetaVAE(
        latent_dim=CONFIG['latent_dim'],
        beta=CONFIG['beta'],
        in_channels=1
    ).to(DEVICE)
    
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, 'best_beta_vae.pth'))
    beta_vae.load_state_dict(checkpoint['model_state_dict'])
    beta_vae.eval()
    print("✓ Beta-VAE loaded")
    
    # Load Conditional VAE
    cvae = ConditionalVAE(
        latent_dim=CONFIG['latent_dim'],
        num_classes=CONFIG['num_classes'],
        in_channels=1
    ).to(DEVICE)
    
    checkpoint = torch.load(os.path.join(CHECKPOINT_DIR, 'best_cvae.pth'))
    cvae.load_state_dict(checkpoint['model_state_dict'])
    cvae.eval()
    print("✓ Conditional VAE loaded")
    print()
    
   #=========================================================================
    # GENERATE SAMPLES
    #=========================================================================
    
    print("=" * 70)
    print("GENERATING SAMPLES")
    print("=" * 70)
    print()
    
    all_generated_beta = []
    all_generated_cvae = []
    
    for class_id in tqdm(range(CONFIG['num_classes']), desc="Generating samples"):
        # Beta-VAE samples
        samples_beta = generate_samples(
            beta_vae, CONFIG['samples_per_class'], class_id, is_conditional=False
        )
        all_generated_beta.append(samples_beta)
        
        # Save individual samples
        for i in range(CONFIG['samples_per_class']):
            save_path = os.path.join(
                SAMPLES_DIR, f'class_{str(class_id).zfill(3)}',
                f'beta_vae_sample_{i:02d}.png'
            )
            save_image(samples_beta[i], save_path, normalize=True)
        
        # Save grid
        save_samples_grid(
            samples_beta,
            os.path.join(SAMPLES_DIR, f'class_{str(class_id).zfill(3)}_beta_vae_grid.png')
        )
        
        # Conditional VAE samples
        samples_cvae = generate_samples(
            cvae, CONFIG['samples_per_class'], class_id, is_conditional=True
        )
        all_generated_cvae.append(samples_cvae)
        
        # Save individual samples
        for i in range(CONFIG['samples_per_class']):
            save_path = os.path.join(
                SAMPLES_DIR, f'class_{str(class_id).zfill(3)}',
                f'cvae_sample_{i:02d}.png'
            )
            save_image(samples_cvae[i], save_path, normalize=True)
        
        # Save grid
        save_samples_grid(
            samples_cvae,
            os.path.join(SAMPLES_DIR, f'class_{str(class_id).zfill(3)}_cvae_grid.png')
        )
    
    print()
    print(f"✓ Generated {CONFIG['num_classes'] * CONFIG['samples_per_class'] * 2} total samples")
    print(f"✓ Saved to: {SAMPLES_DIR}/")
    print()
    
    #=========================================================================
    # SIMILARITY ANALYSIS
    #=========================================================================
    
    print("=" * 70)
    print("SIMILARITY ANALYSIS")
    print("=" * 70)
    print()
    
    similarity_results = {
        'beta_vae': {'ssim': [], 'mse': [], 'cosine': []},
        'cvae': {'ssim': [], 'mse': [], 'cosine': []}
    }
    
    class_similarity = {i: {'beta_vae': [], 'cvae': []} for i in range(CONFIG['num_classes'])}
    
    print("Calculating similarities...")
    for images, labels, paths in tqdm(test_loader, desc="Analyzing"):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        
        # Beta-VAE reconstruction
        with torch.no_grad():
            recon_beta, _, _, _ = beta_vae(images)
            recon_cvae, _, _, _ = cvae(images, labels)
        
        # Calculate metrics
        ssim_beta = calculate_ssim(images[0], recon_beta[0])
        mse_beta = calculate_mse(images[0], recon_beta[0])
        cosine_beta = calculate_cosine_similarity(images[0], recon_beta[0])
        
        ssim_cvae = calculate_ssim(images[0], recon_cvae[0])
        mse_cvae = calculate_mse(images[0], recon_cvae[0])
        cosine_cvae = calculate_cosine_similarity(images[0], recon_cvae[0])
        
        similarity_results['beta_vae']['ssim'].append(ssim_beta)
        similarity_results['beta_vae']['mse'].append(mse_beta)
        similarity_results['beta_vae']['cosine'].append(cosine_beta)
        
        similarity_results['cvae']['ssim'].append(ssim_cvae)
        similarity_results['cvae']['mse'].append(mse_cvae)
        similarity_results['cvae']['cosine'].append(cosine_cvae)
        
        class_id = labels[0].item()
        class_similarity[class_id]['beta_vae'].append(ssim_beta)
        class_similarity[class_id]['cvae'].append(ssim_cvae)
    
    # Calculate averages
    results = {
        'beta_vae': {
            'ssim_mean': np.mean(similarity_results['beta_vae']['ssim']),
            'ssim_std': np.std(similarity_results['beta_vae']['ssim']),
            'mse_mean': np.mean(similarity_results['beta_vae']['mse']),
            'cosine_mean': np.mean(similarity_results['beta_vae']['cosine']),
        },
        'cvae': {
            'ssim_mean': np.mean(similarity_results['cvae']['ssim']),
            'ssim_std': np.std(similarity_results['cvae']['ssim']),
            'mse_mean': np.mean(similarity_results['cvae']['mse']),
            'cosine_mean': np.mean(similarity_results['cvae']['cosine']),
        }
    }
    
    print()
    print("Similarity Results:")
    print("-" * 70)
    print("Beta-VAE:")
    print(f"  SSIM: {results['beta_vae']['ssim_mean']:.4f} ± {results['beta_vae']['ssim_std']:.4f}")
    print(f"  MSE:  {results['beta_vae']['mse_mean']:.6f}")
    print(f"  Cosine Similarity: {results['beta_vae']['cosine_mean']:.4f}")
    print()
    print("Conditional VAE:")
    print(f"  SSIM: {results['cvae']['ssim_mean']:.4f} ± {results['cvae']['ssim_std']:.4f}")
    print(f"  MSE:  {results['cvae']['mse_mean']:.6f}")
    print(f"  Cosine Similarity: {results['cvae']['cosine_mean']:.4f}")
    print()
    
    # Save results
    with open(os.path.join(ANALYSIS_DIR, 'similarity_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    #=========================================================================
    # CREATE VISUALIZATIONS
    #=========================================================================
    
    print("=" * 70)
    print("CREATING VISUALIZATIONS")
    print("=" * 70)
    print()
    
    # Plot similarity distributions
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].hist(similarity_results['beta_vae']['ssim'], bins=20, alpha=0.6, label='Beta-VAE')
    axes[0].hist(similarity_results['cvae']['ssim'], bins=20, alpha=0.6, label='cVAE')
    axes[0].set_xlabel('SSIM')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('SSIM Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(similarity_results['beta_vae']['mse'], bins=20, alpha=0.6, label='Beta-VAE')
    axes[1].hist(similarity_results['cvae']['mse'], bins=20, alpha=0.6, label='cVAE')
    axes[1].set_xlabel('MSE')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('MSE Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].hist(similarity_results['beta_vae']['cosine'], bins=20, alpha=0.6, label='Beta-VAE')
    axes[2].hist(similarity_results['cvae']['cosine'], bins=20, alpha=0.6, label='cVAE')
    axes[2].set_xlabel('Cosine Similarity')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Cosine Similarity Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'similarity_distributions.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Created similarity distribution plots")
    
    print()
    print("=" * 70)
    print("EVALUATION COMPLETE!")
    print("=" * 70)
    print()
    print(f"Results saved to: {RESULTS_DIR}/")
    print(f"Samples: {SAMPLES_DIR}/")
    print(f"Analysis: {ANALYSIS_DIR}/")
    print(f"Plots: {PLOTS_DIR}/")

if __name__ == "__main__":
    main()
