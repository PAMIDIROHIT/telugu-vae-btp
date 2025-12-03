"""
Latent Space Visualization for VAE Models

Analyze and visualize the learned latent space through:
- t-SNE and UMAP projections
- Latent traversals (walking along dimensions)
- Interpolations between glyphs

Usage:
    python latent_visualizer.py --model_path checkpoints/beta_vae_best.pth \\
                                 --model_type beta_vae \\
                                 --data_path data/raw/metadata.csv \\
                                 --num_samples 1000
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torchvision.utils import save_image
from tqdm import tqdm
from PIL import Image

# Try importing UMAP, but don't fail if not available
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Warning: UMAP not installed. Install with: pip install umap-learn")

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import create_model
from scripts.utils import load_config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Visualize VAE latent space')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['vanilla_vae', 'beta_vae', 'cvae'],
                        help='Type of VAE model')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to dataset metadata CSV')
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='Latent dimensionality')
    parser.add_argument('--num_samples', type=int, default=1000,
                        help='Number of samples to visualize')
    parser.add_argument('--num_classes', type=int, default=128,
                        help='Number of classes (for cVAE)')
    parser.add_argument('--output_dir', type=str, default='results/latent_space',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Traversal parameters
    parser.add_argument('--traversal_range', type=float, default=3.0,
                        help='Range for latent traversal (-range to +range)')
    parser.add_argument('--traversal_steps', type=int, default=11,
                        help='Number of steps in latent traversal')
    parser.add_argument('--max_traversal_dims', type=int, default=10,
                        help='Maximum number of dimensions to traverse')
    
    return parser.parse_args()


def load_model(args):
    """Load trained VAE model."""
    print(f"Loading model from {args.model_path}...")
    
    if args.model_type == 'cvae':
        model = create_model(args.model_type, args.latent_dim, args.num_classes)
    else:
        model = create_model(args.model_type, args.latent_dim)
    
    checkpoint = torch.load(args.model_path, map_location=args.device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(args.device)
    model.eval()
    
    return model


def encode_dataset(model, data_path, args):
    """
    Encode dataset to latent space.
    
    Returns:
        latents: Encoded latent vectors
        labels: Glyph class labels
        metadata: Additional metadata
    """
    print(f"\nEncoding dataset to latent space...")
    
    # Load metadata
    df = pd.read_csv(data_path)
    
    # Limit to num_samples
    if len(df) > args.num_samples:
        df = df.sample(n=args.num_samples, random_state=args.seed)
    
    latents = []
    labels = []
    
    from torchvision import transforms
    from PIL import Image
    
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Encoding images"):
            # Load image
            img_path = row['path']
            if not os.path.exists(img_path):
                continue
            
            img = Image.open(img_path)
            img_tensor = transform(img).unsqueeze(0).to(args.device)
            
            # Encode
            if args.model_type == 'cvae':
                # For cVAE, need class label
                class_id = row.get('unicode', 0)
                c = torch.tensor([class_id], dtype=torch.long, device=args.device)
                mu, logvar = model.encode(img_tensor, c)
            else:
                mu, logvar = model.encode(img_tensor)
            
            latents.append(mu.cpu().numpy())
            labels.append(row.get('glyph', 'unknown'))
    
    latents = np.concatenate(latents, axis=0)
    
    print(f"Encoded {len(latents)} samples to {latents.shape[1]}-D latent space")
    
    return latents, labels, df


def plot_tsne(latents, labels, output_dir):
    """Create t-SNE projection of latent space."""
    print("\nComputing t-SNE projection...")
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latents_2d = tsne.fit_transform(latents)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color by glyph class
    unique_labels = list(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, min(len(unique_labels), 20)))
    
    for i, label in enumerate(unique_labels[:20]):  # Limit to 20 classes for clarity
        mask = np.array(labels) == label
        ax.scatter(latents_2d[mask, 0], latents_2d[mask, 1], 
                   c=[colors[i]], label=label, alpha=0.6, s=30)
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('t-SNE Projection of Latent Space', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    tsne_path = output_dir / 'tsne_projection.png'
    plt.savefig(tsne_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved t-SNE plot: {tsne_path}")
    
    return latents_2d


def plot_umap(latents, labels, output_dir):
    """Create UMAP projection of latent space."""
    if not HAS_UMAP:
        print("\nSkipping UMAP (not installed)")
        return None
    
    print("\nComputing UMAP projection...")
    
    umap = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    latents_2d = umap.fit_transform(latents)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    unique_labels = list(set(labels))
    colors = plt.cm.tab20(np.linspace(0, 1, min(len(unique_labels), 20)))
    
    for i, label in enumerate(unique_labels[:20]):
        mask = np.array(labels) == label
        ax.scatter(latents_2d[mask, 0], latents_2d[mask, 1],
                   c=[colors[i]], label=label, alpha=0.6, s=30)
    
    ax.set_xlabel('UMAP Dimension 1', fontsize=12)
    ax.set_ylabel('UMAP Dimension 2', fontsize=12)
    ax.set_title('UMAP Projection of Latent Space', fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    umap_path = output_dir / 'umap_projection.png'
    plt.savefig(umap_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved UMAP plot: {umap_path}")
    
    return latents_2d


def latent_traversal(model, args, output_dir):
    """
    Generate latent traversal images.
    
    Walk along each latent dimension while keeping others at 0.
    """
    print(f"\nGenerating latent traversals...")
    
    traversal_dir = output_dir / 'latent_traversals'
    traversal_dir.mkdir(exist_ok=True)
    
    # Number of dimensions to traverse
    num_dims = min(args.latent_dim, args.max_traversal_dims)
    
    # Values to traverse
    values = np.linspace(-args.traversal_range, args.traversal_range, args.traversal_steps)
    
    with torch.no_grad():
        for dim in tqdm(range(num_dims), desc="Traversing dimensions"):
            images = []
            
            for val in values:
                # Create latent vector with only this dimension varying
                z = torch.zeros(1, args.latent_dim, device=args.device)
                z[0, dim] = val
                
                # Decode
                if args.model_type == 'cvae':
                    # Use a fixed class for cVAE
                    c = torch.tensor([0], dtype=torch.long, device=args.device)
                    img = model.decode(z, c)
                else:
                    img = model.decode(z)
                
                images.append(img.cpu())
            
            # Create grid
            images = torch.cat(images, dim=0)
            grid_path = traversal_dir / f'dim_{dim:02d}.png'
            save_image(images, grid_path, nrow=args.traversal_steps, normalize=True, padding=2)
    
    print(f"Saved {num_dims} traversal images to {traversal_dir}/")
    
    # Create summary grid showing all traversals
    create_traversal_summary(traversal_dir, num_dims, output_dir)


def create_traversal_summary(traversal_dir, num_dims, output_dir):
    """Create a summary image showing all traversals."""
    print("Creating traversal summary...")
    
    rows = (num_dims + 1) // 2
    cols = 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 5 * rows))
    axes = axes.flatten()
    
    for dim in range(num_dims):
        img_path = traversal_dir / f'dim_{dim:02d}.png'
        if img_path.exists():
            img = plt.imread(str(img_path))
            axes[dim].imshow(img)
            axes[dim].set_title(f'Latent Dimension {dim}', fontsize=12, fontweight='bold')
            axes[dim].axis('off')
    
    # Hide unused subplots
    for i in range(num_dims, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    summary_path = output_dir / 'traversal_summary.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved traversal summary: {summary_path}")


def interpolate_glyphs(model, latent1, latent2, args, output_dir, name='interpolation'):
    """
    Interpolate between two latent vectors.
    
    Args:
        latent1: First latent vector
        latent2: Second latent vector
        name: Name for output file
    """
    steps = 10
    alphas = np.linspace(0, 1, steps)
    
    images = []
    
    with torch.no_grad():
        for alpha in alphas:
            # Linear interpolation
            z = (1 - alpha) * latent1 + alpha * latent2
            z = torch.tensor(z, dtype=torch.float32, device=args.device).unsqueeze(0)
            
            if args.model_type == 'cvae':
                c = torch.tensor([0], dtype=torch.long, device=args.device)
                img = model.decode(z, c)
            else:
                img = model.decode(z)
            
            images.append(img.cpu())
    
    images = torch.cat(images, dim=0)
    interp_path = output_dir / f'{name}.png'
    save_image(images, interp_path, nrow=steps, normalize=True, padding=2)
    
    return images


def analyze_latent_statistics(latents, output_dir):
    """Analyze and visualize latent space statistics."""
    print("\nAnalyzing latent statistics...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Mean and std per dimension
    ax1 = axes[0, 0]
    means = latents.mean(axis=0)
    stds = latents.std(axis=0)
    dims = np.arange(len(means))
    
    ax1.errorbar(dims, means, yerr=stds, fmt='o-', capsize=3, alpha=0.7)
    ax1.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax1.set_xlabel('Latent Dimension')
    ax1.set_ylabel('Mean ± Std')
    ax1.set_title('Latent Dimension Statistics')
    ax1.grid(alpha=0.3)
    
    # 2. Variance per dimension
    ax2 = axes[0, 1]
    variances = latents.var(axis=0)
    ax2.bar(dims, variances, color='coral', edgecolor='black', alpha=0.7)
    ax2.axhline(1, color='r', linestyle='--', alpha=0.5, label='Unit Variance')
    ax2.set_xlabel('Latent Dimension')
    ax2.set_ylabel('Variance')
    ax2.set_title('Latent Dimension Variance')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Distribution of latent values (histogram)
    ax3 = axes[1, 0]
    ax3.hist(latents.flatten(), bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax3.axvline(0, color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Latent Value')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of All Latent Values')
    ax3.grid(alpha=0.3)
    
    # 4. Correlation heatmap (if not too many dims)
    ax4 = axes[1, 1]
    if latents.shape[1] <= 32:
        corr = np.corrcoef(latents.T)
        im = ax4.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax4.set_title('Latent Dimension Correlations')
        ax4.set_xlabel('Dimension')
        ax4.set_ylabel('Dimension')
        plt.colorbar(im, ax=ax4, fraction=0.046)
    else:
        ax4.text(0.5, 0.5, f'Too many dimensions ({latents.shape[1]}) for correlation plot',
                 ha='center', va='center', transform=ax4.transAxes)
        ax4.axis('off')
    
    plt.tight_layout()
    stats_path = output_dir / 'latent_statistics.png'
    plt.savefig(stats_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved latent statistics: {stats_path}")


def main():
    """Main execution function."""
    args = parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("Latent Space Visualization")
    print("=" * 70)
    print(f"Model: {args.model_type}")
    print(f"Latent Dim: {args.latent_dim}")
    print(f"Num Samples: {args.num_samples}")
    print(f"Output: {output_dir}")
    print("=" * 70)
    
    # Load model
    model = load_model(args)
    
    # Encode dataset
    latents, labels, metadata = encode_dataset(model, args.data_path, args)
    
    # Save latent encodings
    latent_df = pd.DataFrame(latents, columns=[f'z_{i}' for i in range(latents.shape[1])])
    latent_df['label'] = labels
    latent_df.to_csv(output_dir / 'latent_encodings.csv', index=False)
    print(f"\nSaved latent encodings to {output_dir / 'latent_encodings.csv'}")
    
    # Analyze statistics
    analyze_latent_statistics(latents, output_dir)
    
    # t-SNE projection
    plot_tsne(latents, labels, output_dir)
    
    # UMAP projection
    plot_umap(latents, labels, output_dir)
    
    # Latent traversals
    latent_traversal(model, args, output_dir)
    
    # Random interpolations
    print("\nGenerating interpolations...")
    interp_dir = output_dir / 'interpolations'
    interp_dir.mkdir(exist_ok=True)
    
    for i in range(5):
        idx1, idx2 = np.random.choice(len(latents), 2, replace=False)
        interpolate_glyphs(model, latents[idx1], latents[idx2], args, 
                           interp_dir, f'interpolation_{i}')
    
    print(f"Saved interpolations to {interp_dir}/")
    
    print("\n✅ Latent space visualization complete!")
    print(f"📁 Check {output_dir}/ for all visualizations")


if __name__ == '__main__':
    main()
