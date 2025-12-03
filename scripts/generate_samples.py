"""
Sample Generation Script for VAE Models

Generate synthetic Telugu glyphs from trained VAE models by sampling
from the learned latent distribution.

Usage:
    # Vanilla VAE or Beta-VAE (unconditional)
    python generate_samples.py --model_path checkpoints/beta_vae_best.pth \\
                                --model_type beta_vae \\
                                --num_samples 100 \\
                                --latent_dim 32
    
    # Conditional VAE (generate specific glyph class)
    python generate_samples.py --model_path checkpoints/cvae_best.pth \\
                                --model_type cvae \\
                                --conditional \\
                                --condition_class 15 \\
                                --num_samples 50
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import create_model


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Generate samples from trained VAE models')
    
    # Model configuration
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint (.pth file)')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['vanilla_vae', 'beta_vae', 'cvae'],
                        help='Type of VAE model')
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='Latent dimensionality (must match trained model)')
    
    # Sampling configuration
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of samples to generate')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size for generation (memory constraint)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Conditional generation (cVAE only)
    parser.add_argument('--conditional', action='store_true',
                        help='Enable conditional generation (for cVAE)')
    parser.add_argument('--condition_class', type=int, default=None,
                        help='Glyph class ID for conditional generation')
    parser.add_argument('--num_classes', type=int, default=128,
                        help='Total number of classes (for cVAE)')
    
    # Output configuration
    parser.add_argument('--output_dir', type=str, default='results/generated_samples',
                        help='Output directory for generated samples')
    parser.add_argument('--save_individual', action='store_true', default=True,
                        help='Save individual PNG files for each sample')
    parser.add_argument('--grid_cols', type=int, default=10,
                        help='Number of columns in sample grid')
    
    # Device configuration
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (cuda or cpu)')
    
    return parser.parse_args()


def load_model(args):
    """
    Load trained VAE model from checkpoint.
    
    Args:
        args: Command-line arguments
    
    Returns:
        model: Loaded model in eval mode
    """
    print(f"Loading {args.model_type} model from {args.model_path}...")
    
    # Create model architecture
    if args.model_type == 'cvae':
        model = create_model(
            model_type=args.model_type,
            latent_dim=args.latent_dim,
            num_classes=args.num_classes,
            in_channels=1
        )
    else:
        model = create_model(
            model_type=args.model_type,
            latent_dim=args.latent_dim,
            in_channels=1
        )
    
    # Load checkpoint
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.model_path}")
    
    checkpoint = torch.load(args.model_path, map_location=args.device)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Training loss: {checkpoint.get('loss', 'N/A')}")
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(args.device)
    model.eval()
    
    print(f"Model loaded successfully on {args.device}")
    return model


def generate_samples(model, args):
    """
    Generate samples from the latent distribution.
    
    Args:
        model: Trained VAE model
        args: Command-line arguments
    
    Returns:
        samples: Generated images (num_samples, 1, H, W)
        latent_vectors: Sampled latent vectors (num_samples, latent_dim)
        conditions: Condition labels if conditional (num_samples,)
    """
    print(f"\nGenerating {args.num_samples} samples...")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    all_samples = []
    all_latents = []
    all_conditions = []
    
    # Generate in batches to avoid memory issues
    num_batches = (args.num_samples + args.batch_size - 1) // args.batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Generating batches"):
            # Determine batch size for this iteration
            start_idx = batch_idx * args.batch_size
            end_idx = min(start_idx + args.batch_size, args.num_samples)
            current_batch_size = end_idx - start_idx
            
            # Sample from standard normal latent distribution N(0, I)
            z = torch.randn(current_batch_size, args.latent_dim, device=args.device)
            
            if args.conditional and args.model_type == 'cvae':
                # Conditional generation
                if args.condition_class is not None:
                    # Generate all samples for specific class
                    c = torch.full((current_batch_size,), args.condition_class, 
                                   dtype=torch.long, device=args.device)
                else:
                    # Random class for each sample
                    c = torch.randint(0, args.num_classes, (current_batch_size,),
                                      device=args.device)
                
                samples = model.decode(z, c)
                all_conditions.append(c.cpu().numpy())
            else:
                # Unconditional generation
                samples = model.decode(z)
                c = None
            
            all_samples.append(samples.cpu())
            all_latents.append(z.cpu().numpy())
    
    # Concatenate all batches
    all_samples = torch.cat(all_samples, dim=0)
    all_latents = np.concatenate(all_latents, axis=0)
    
    if all_conditions:
        all_conditions = np.concatenate(all_conditions, axis=0)
    else:
        all_conditions = None
    
    print(f"Generated {all_samples.shape[0]} samples of shape {all_samples.shape}")
    
    return all_samples, all_latents, all_conditions


def save_samples(samples, latents, conditions, args):
    """
    Save generated samples to disk.
    
    Args:
        samples: Generated images
        latents: Latent vectors
        conditions: Condition labels (None if unconditional)
        args: Command-line arguments
    """
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving samples to {output_dir}...")
    
    # 1. Save grid visualization
    grid_path = output_dir / 'grid.png'
    nrow = min(args.grid_cols, samples.shape[0])
    save_image(samples, grid_path, nrow=nrow, normalize=True, padding=2)
    print(f"Saved grid visualization: {grid_path}")
    
    # 2. Save individual images
    if args.save_individual:
        individual_dir = output_dir / 'individual'
        individual_dir.mkdir(exist_ok=True)
        
        for i, img in enumerate(tqdm(samples, desc="Saving individual images")):
            img_path = individual_dir / f'sample_{i:04d}.png'
            save_image(img, img_path, normalize=True)
        
        print(f"Saved {len(samples)} individual images to {individual_dir}/")
    
    # 3. Save latent vectors and metadata
    metadata = {
        f'z_{j}': latents[:, j] for j in range(latents.shape[1])
    }
    
    if conditions is not None:
        metadata['condition_class'] = conditions
    
    metadata['sample_id'] = np.arange(len(samples))
    metadata['model_type'] = args.model_type
    metadata['latent_dim'] = args.latent_dim
    metadata['seed'] = args.seed
    
    df = pd.DataFrame(metadata)
    csv_path = output_dir / 'latent_vectors.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved latent vectors and metadata: {csv_path}")
    
    # 4. Create summary plot with statistics
    create_summary_plot(samples, latents, conditions, output_dir, args)
    
    print(f"\n✓ All outputs saved to {output_dir}/")


def create_summary_plot(samples, latents, conditions, output_dir, args):
    """
    Create summary visualization with sample statistics.
    
    Args:
        samples: Generated images
        latents: Latent vectors
        conditions: Condition labels
        output_dir: Output directory
        args: Command-line arguments
    """
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Show first 16 samples
    ax1 = plt.subplot(2, 3, 1)
    num_display = min(16, len(samples))
    grid = torch.cat([samples[i] for i in range(num_display)], dim=0)
    grid = grid.view(4, 4, samples.shape[2], samples.shape[3])
    grid = grid.permute(0, 2, 1, 3).contiguous()
    grid = grid.view(4 * samples.shape[2], 4 * samples.shape[3])
    
    ax1.imshow(grid.numpy(), cmap='gray')
    ax1.set_title('Sample Grid (first 16)', fontsize=12, fontweight='bold')
    ax1.axis('off')
    
    # 2. Pixel intensity histogram
    ax2 = plt.subplot(2, 3, 2)
    pixel_values = samples.numpy().flatten()
    ax2.hist(pixel_values, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.set_xlabel('Pixel Intensity')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Pixel Intensity Distribution', fontsize=12, fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # 3. Latent dimension variance
    ax3 = plt.subplot(2, 3, 3)
    latent_var = np.var(latents, axis=0)
    ax3.bar(range(len(latent_var)), latent_var, color='coral', edgecolor='black')
    ax3.set_xlabel('Latent Dimension')
    ax3.set_ylabel('Variance')
    ax3.set_title('Latent Dimension Variance', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3)
    
    # 4. Latent correlation matrix
    ax4 = plt.subplot(2, 3, 4)
    if latents.shape[1] <= 64:  # Only plot if not too many dimensions
        corr_matrix = np.corrcoef(latents.T)
        im = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax4.set_title('Latent Correlation Matrix', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Latent Dimension')
        ax4.set_ylabel('Latent Dimension')
        plt.colorbar(im, ax=ax4, fraction=0.046)
    else:
        ax4.text(0.5, 0.5, f'Too many dimensions ({latents.shape[1]}) to visualize',
                 ha='center', va='center', transform=ax4.transAxes)
        ax4.axis('off')
    
    # 5. Class distribution (if conditional)
    ax5 = plt.subplot(2, 3, 5)
    if conditions is not None:
        unique, counts = np.unique(conditions, return_counts=True)
        ax5.bar(unique, counts, color='mediumseagreen', edgecolor='black')
        ax5.set_xlabel('Condition Class')
        ax5.set_ylabel('Count')
        ax5.set_title('Condition Class Distribution', fontsize=12, fontweight='bold')
        ax5.grid(alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Unconditional Generation', 
                 ha='center', va='center', transform=ax5.transAxes, fontsize=14)
        ax5.axis('off')
    
    # 6. Statistics summary (text)
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = f"""
    Generation Summary
    ==================
    Model Type: {args.model_type}
    Latent Dim: {args.latent_dim}
    Num Samples: {len(samples)}
    Conditional: {args.conditional}
    Seed: {args.seed}
    
    Image Statistics
    ----------------
    Mean: {pixel_values.mean():.4f}
    Std: {pixel_values.std():.4f}
    Min: {pixel_values.min():.4f}
    Max: {pixel_values.max():.4f}
    
    Latent Statistics
    -----------------
    Mean: {latents.mean():.4f}
    Std: {latents.std():.4f}
    """
    
    ax6.text(0.1, 0.5, stats_text, transform=ax6.transAxes, 
             fontsize=10, verticalalignment='center', family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    summary_path = output_dir / 'summary.png'
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary plot: {summary_path}")


def main():
    """Main execution function."""
    args = parse_args()
    
    # Print configuration
    print("=" * 70)
    print("VAE Sample Generation")
    print("=" * 70)
    print(f"Model Type: {args.model_type}")
    print(f"Model Path: {args.model_path}")
    print(f"Latent Dimension: {args.latent_dim}")
    print(f"Number of Samples: {args.num_samples}")
    print(f"Conditional: {args.conditional}")
    if args.conditional:
        print(f"Condition Class: {args.condition_class if args.condition_class is not None else 'Random'}")
    print(f"Device: {args.device}")
    print(f"Random Seed: {args.seed}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 70)
    
    # Validate conditional arguments
    if args.conditional and args.model_type != 'cvae':
        raise ValueError("Conditional generation requires model_type='cvae'")
    
    # Load model
    model = load_model(args)
    
    # Generate samples
    samples, latents, conditions = generate_samples(model, args)
    
    # Save outputs
    save_samples(samples, latents, conditions, args)
    
    print("\n✅ Sample generation complete!")
    print(f"📁 Check {args.output_dir}/ for results")


if __name__ == '__main__':
    main()
