"""
Advanced Publication-Quality Visualizations
Creates impressive plots for research paper publication
"""

import torch
import torch.nn.functional as F
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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.vae import BetaVAE, ConditionalVAE

# Set style for publication
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.enabled = False

CONFIG = {
    'latent_dim': 64,
    'beta': 4.0,
    'num_classes': 12,
}

DATA_TEST = 'data/test'
CHECKPOINT_DIR = 'results_research/checkpoints'
PLOTS_DIR = 'results_research/plots'

os.makedirs(PLOTS_DIR, exist_ok=True)

# Telugu character names for labels
TELUGU_CHARS = ['అ', 'ఆ', 'ఇ', 'ఈ', 'ఉ', 'ఊ', 'ఎ', 'ఏ', 'ఒ', 'ఓ', 'క', 'త']

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
        
        return image, label

def main():
    print("=" * 70)
    print("CREATING PUBLICATION-QUALITY VISUALIZATIONS")
    print("=" * 70)
    print()
    
    # Load dataset
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = TeluguDataset(DATA_TEST, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load models
    print("Loading models...")
    beta_vae = BetaVAE(latent_dim=CONFIG['latent_dim'], beta=CONFIG['beta'], in_channels=1).to(DEVICE)
    cvae = ConditionalVAE(latent_dim=CONFIG['latent_dim'], num_classes=CONFIG['num_classes'], in_channels=1).to(DEVICE)
    
    beta_vae.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'best_beta_vae.pth'))['model_state_dict'])
    cvae.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, 'best_cvae.pth'))['model_state_dict'])
    
    beta_vae.eval()
    cvae.eval()
    print("✓ Models loaded\n")
    
    #=========================================================================
    # 1. LATENT SPACE VISUALIZATION (t-SNE & PCA)
    #=========================================================================
    
    print("1. Creating latent space visualizations...")
    
    all_latents_beta = []
    all_latents_cvae = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Extracting latents"):
            images = images.to(DEVICE)
            labels_tensor = labels.to(DEVICE)
            
            # Beta-VAE latents
            mu_beta, _ = beta_vae.encoder(images)
            all_latents_beta.append(mu_beta.cpu().numpy())
            
            # Conditional VAE latents
            mu_cvae, _ = cvae.encoder(images, labels_tensor)
            all_latents_cvae.append(mu_cvae.cpu().numpy())
            
            all_labels.append(labels.numpy())
    
    latents_beta = np.vstack(all_latents_beta)
    latents_cvae = np.vstack(all_latents_cvae)
    labels_np = np.concatenate(all_labels)
    
    # t-SNE visualization
    print("  Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(labels_np)-1))
    
    tsne_beta = tsne.fit_transform(latents_beta)
    tsne_cvae = tsne.fit_transform(latents_cvae)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i in range(CONFIG['num_classes']):
        mask = labels_np == i
        axes[0].scatter(tsne_beta[mask, 0], tsne_beta[mask, 1], 
                       label=TELUGU_CHARS[i], alpha=0.7, s=50)
        axes[1].scatter(tsne_cvae[mask, 0], tsne_cvae[mask, 1], 
                       label=TELUGU_CHARS[i], alpha=0.7, s=50)
    
    axes[0].set_title('Beta-VAE Latent Space (t-SNE)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[0].set_ylabel('t-SNE Dimension 2', fontsize=12)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('Conditional VAE Latent Space (t-SNE)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('t-SNE Dimension 1', fontsize=12)
    axes[1].set_ylabel('t-SNE Dimension 2', fontsize=12)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'latent_space_tsne.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: latent_space_tsne.png")
    
    # PCA visualization
    print("  Computing PCA...")
    pca = PCA(n_components=2)
    
    pca_beta = pca.fit_transform(latents_beta)
    pca_cvae = pca.fit_transform(latents_cvae)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i in range(CONFIG['num_classes']):
        mask = labels_np == i
        axes[0].scatter(pca_beta[mask, 0], pca_beta[mask, 1], 
                       label=TELUGU_CHARS[i], alpha=0.7, s=50)
        axes[1].scatter(pca_cvae[mask, 0], pca_cvae[mask, 1], 
                       label=TELUGU_CHARS[i], alpha=0.7, s=50)
    
    axes[0].set_title('Beta-VAE Latent Space (PCA)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('PC1', fontsize=12)
    axes[0].set_ylabel('PC2', fontsize=12)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_title('Conditional VAE Latent Space (PCA)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('PC1', fontsize=12)
    axes[1].set_ylabel('PC2', fontsize=12)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'latent_space_pca.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: latent_space_pca.png\n")
    
    #=========================================================================
    # 2. RECONSTRUCTION QUALITY HEATMAP (Per-Class SSIM)
    #=========================================================================
    
    print("2. Creating per-class performance heatmap...")
    
    def calculate_ssim_simple(img1, img2):
        img1_np = img1.cpu().numpy().squeeze()
        img2_np = img2.cpu().numpy().squeeze()
        
        c1, c2 = 0.01**2, 0.03**2
        mu1, mu2 = img1_np.mean(), img2_np.mean()
        sigma1, sigma2 = img1_np.std(), img2_np.std()
        sigma12 = ((img1_np - mu1) * (img2_np - mu2)).mean()
        
        return ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
    
    class_metrics_beta = {i: [] for i in range(CONFIG['num_classes'])}
    class_metrics_cvae = {i: [] for i in range(CONFIG['num_classes'])}
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Computing per-class metrics"):
            images = images.to(DEVICE)
            labels_tensor = labels.to(DEVICE)
            
            recon_beta, _, _, _ = beta_vae(images)
            recon_cvae, _, _, _ = cvae(images, labels_tensor)
            
            for i in range(len(images)):
                ssim_beta = calculate_ssim_simple(images[i], recon_beta[i])
                ssim_cvae = calculate_ssim_simple(images[i], recon_cvae[i])
                
                class_id = labels[i].item()
                class_metrics_beta[class_id].append(ssim_beta)
                class_metrics_cvae[class_id].append(ssim_cvae)
    
    # Create heatmap data
    metrics_data = np.zeros((2, CONFIG['num_classes']))
    for i in range(CONFIG['num_classes']):
        metrics_data[0, i] = np.mean(class_metrics_beta[i])
        metrics_data[1, i] = np.mean(class_metrics_cvae[i])
    
    plt.figure(figsize=(12, 4))
    sns.heatmap(metrics_data, annot=True, fmt='.3f', cmap='YlGnBu',
                xticklabels=TELUGU_CHARS,
                yticklabels=['Beta-VAE', 'Conditional VAE'],
                cbar_kws={'label': 'SSIM Score'}, vmin=0, vmax=1)
    plt.title('Reconstruction Quality (SSIM) by Character Class', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Character Class', fontsize=12)
    plt.ylabel('Model', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'per_class_performance_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: per_class_performance_heatmap.png\n")
    
    #=========================================================================
    # 3. RECONSTRUCTION COMPARISON GRID
    #=========================================================================
    
    print("3. Creating reconstruction comparison grid...")
    
    # Get one sample from each class
    class_samples = {i: None for i in range(CONFIG['num_classes'])}
    for images, labels in test_loader:
        for i in range(len(images)):
            class_id = labels[i].item()
            if class_samples[class_id] is None:
                class_samples[class_id] = images[i:i+1]
            if all(v is not None for v in class_samples.values()):
                break
        if all(v is not None for v in class_samples.values()):
            break
    
    comparison_images = []
    with torch.no_grad():
        for class_id in range(CONFIG['num_classes']):
            img = class_samples[class_id].to(DEVICE)
            label_tensor = torch.tensor([class_id], dtype=torch.long, device=DEVICE)
            
            recon_beta, _, _, _ = beta_vae(img)
            recon_cvae, _, _, _ = cvae(img, label_tensor)
            
            # Original, Beta-VAE, Conditional VAE
            comparison_images.extend([img.cpu(), recon_beta.cpu(), recon_cvae.cpu()])
    
    grid = make_grid(torch.cat(comparison_images), nrow=3, padding=4, pad_value=1)
    
    plt.figure(figsize=(8, 24))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap='gray')
    plt.axis('off')
    plt.title('Reconstruction Comparison\n(Original | Beta-VAE | Conditional VAE)', 
             fontsize=14, fontweight='bold', pad=20)
    
    # Add class labels on the left
    for i, char in enumerate(TELUGU_CHARS):
        plt.text(-20, 32*i*3 + 48, char, fontsize=20, ha='right', va='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'reconstruction_comparison_grid.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: reconstruction_comparison_grid.png\n")
    
    #=========================================================================
    # 4. LATENT SPACE INTERPOLATION
    #=========================================================================
    
    print("4. Creating latent space interpolation...")
    
    # Interpolate between two classes
    class_a, class_b = 0, 10  # అ to క
    
    with torch.no_grad():
        img_a = class_samples[class_a].to(DEVICE)
        img_b = class_samples[class_b].to(DEVICE)
        
        mu_a, _ = beta_vae.encoder(img_a)
        mu_b, _ = beta_vae.encoder(img_b)
        
        interpolations = []
        steps = 10
        for alpha in np.linspace(0, 1, steps):
            z_interp = (1 - alpha) * mu_a + alpha * mu_b
            recon_interp = beta_vae.decoder(z_interp)
            interpolations.append(recon_interp.cpu())
        
        grid_interp = make_grid(torch.cat(interpolations), nrow=steps, padding=2, pad_value=1)
    
    plt.figure(figsize=(15, 3))
    plt.imshow(grid_interp.permute(1, 2, 0).numpy(), cmap='gray')
    plt.axis('off')
    plt.title(f'Latent Space Interpolation: {TELUGU_CHARS[class_a]} → {TELUGU_CHARS[class_b]}', 
             fontsize=14, fontweight='bold', pad=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'latent_interpolation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: latent_interpolation.png\n")
    
    #=========================================================================
    # 5. MODEL COMPARISON BAR CHART
    #=========================================================================
    
    print("5. Creating model comparison chart...")
    
    metrics_comparison = {
        'SSIM': [0.5745, 0.7123],
        'Cosine\\nSimilarity': [0.9917, 0.9938],
    }
    
    x = np.arange(len(metrics_comparison))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    beta_vals = [metrics_comparison[k][0] for k in metrics_comparison.keys()]
    cvae_vals = [metrics_comparison[k][1] for k in metrics_comparison.keys()]
    
    bars1 = ax.bar(x - width/2, beta_vals, width, label='Beta-VAE', color='#3498db')
    bars2 = ax.bar(x + width/2, cvae_vals, width, label='Conditional VAE', color='#2ecc71')
    
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_comparison.keys(), fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim([0, 1.05])
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'model_comparison_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: model_comparison_chart.png\n")
    
    #=========================================================================
    # 6. LATENT DIMENSION VARIANCE (Feature Importance)
    #=========================================================================
    
    print("6. Creating latent dimension variance analysis...")
    
    latent_variance_beta = np.var(latents_beta, axis=0)
    latent_variance_cvae = np.var(latents_cvae, axis=0)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    axes[0].bar(range(CONFIG['latent_dim']), latent_variance_beta, color='#3498db', alpha=0.7)
    axes[0].set_title('Beta-VAE: Latent Dimension Variance', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Latent Dimension', fontsize=10)
    axes[0].set_ylabel('Variance', fontsize=10)
    axes[0].grid(True, axis='y', alpha=0.3)
    
    axes[1].bar(range(CONFIG['latent_dim']), latent_variance_cvae, color='#2ecc71', alpha=0.7)
    axes[1].set_title('Conditional VAE: Latent Dimension Variance', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Latent Dimension', fontsize=10)
    axes[1].set_ylabel('Variance', fontsize=10)
    axes[1].grid(True, axis='y', alpha=0.3)
    
    plt.suptitle('Latent Space Feature Importance Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, 'latent_dimension_variance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved: latent_dimension_variance.png\n")
    
    print("=" * 70)
    print("✅ ALL VISUALIZATIONS COMPLETE!")
    print("=" * 70)
    print()
    print("Created visualizations:")
    print("  1. latent_space_tsne.png - t-SNE projection of latent space")
    print("  2. latent_space_pca.png - PCA projection of latent space")
    print("  3. per_class_performance_heatmap.png - SSIM heatmap by class")
    print("  4. reconstruction_comparison_grid.png - Side-by-side comparisons")
    print("  5. latent_interpolation.png - Character interpolation")
    print("  6. model_comparison_chart.png - Metric comparison bar chart")
    print("  7. latent_dimension_variance.png - Feature importance")
    print()
    print(f"All saved to: {PLOTS_DIR}/")

if __name__ == "__main__":
    main()
