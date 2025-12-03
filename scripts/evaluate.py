"""
Evaluation script for VAE models.

Computes:
- FID (Fréchet Inception Distance)
- KID (Kernel Inception Distance)
- Reconstruction accuracy
- Latent space visualizations
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import csv
from tqdm import tqdm
from sklearn.decomposition import PCA
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import create_model
from scripts.utils import load_checkpoint


def compute_fid(real_features, fake_features):
    """
    Compute Fréchet Inception Distance (FID).
    
    Args:
        real_features: Real image features (N, D)
        fake_features: Generated image features (N, D)
    
    Returns:
        FID score
    """
    # Compute mean and covariance
    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    
    sigma_real = np.cov(real_features.T)
    sigma_fake = np.cov(fake_features.T)
    
    # Compute FID
    mean_diff = np.sum((mu_real - mu_fake) ** 2)
    
    # Matrix square root
    U, S, Vt = np.linalg.svd(
        sigma_real @ sigma_fake,
        full_matrices=False
    )
    sqrt_prod = U @ np.diag(np.sqrt(np.maximum(S, 1e-10))) @ Vt
    
    trace_term = np.trace(sigma_real + sigma_fake - 2 * sqrt_prod)
    fid = mean_diff + trace_term
    
    return float(np.sqrt(np.maximum(fid, 1e-10)))


def compute_kid(real_features, fake_features, kernel='rbf', gamma=None):
    """
    Compute Kernel Inception Distance (KID).
    
    Args:
        real_features: Real image features (N, D)
        fake_features: Generated image features (N, D)
        kernel: 'rbf' or 'poly'
        gamma: RBF gamma parameter
    
    Returns:
        KID score
    """
    # Compute kernel matrices
    if gamma is None:
        gamma = 1.0 / real_features.shape[1]
    
    from sklearn.metrics.pairwise import rbf_kernel, polynomial_kernel
    
    if kernel == 'rbf':
        K_rr = rbf_kernel(real_features, gamma=gamma)
        K_ff = rbf_kernel(fake_features, gamma=gamma)
        K_rf = rbf_kernel(real_features, fake_features, gamma=gamma)
    else:
        K_rr = polynomial_kernel(real_features, degree=3)
        K_ff = polynomial_kernel(fake_features, degree=3)
        K_rf = polynomial_kernel(real_features, fake_features, degree=3)
    
    # Compute KID
    m = K_rr.shape[0]
    n = K_ff.shape[0]
    
    kid = (
        np.mean(K_rr) + np.mean(K_ff) - 
        2.0 * np.mean(K_rf)
    )
    
    return float(np.maximum(kid, 0))


@torch.no_grad()
def extract_features(model, data_loader, device, layer='latent'):
    """
    Extract features from model.
    
    Args:
        model: VAE model
        data_loader: DataLoader
        device: Device
        layer: 'latent' (mean) or 'recon' (reconstruction)
    
    Returns:
        Features array (N, D)
    """
    features = []
    
    for x in tqdm(data_loader, desc=f'Extracting {layer} features'):
        x = x.to(device)
        
        if layer == 'latent':
            mu, _ = model.encode(x)
            features.append(mu.cpu().numpy())
        elif layer == 'recon':
            output = model(x)
            if len(output) == 4:
                recon_x, _, _, _ = output
            else:
                recon_x, _, _ = output
            features.append(recon_x.cpu().numpy().reshape(x.size(0), -1))
    
    return np.concatenate(features, axis=0)


def compute_reconstruction_error(model, data_loader, device):
    """Compute mean squared reconstruction error."""
    total_mse = 0
    
    for x in tqdm(data_loader, desc='Computing reconstruction error'):
        x = x.to(device)
        output = model(x)
        
        if len(output) == 4:
            recon_x, _, _, _ = output
        else:
            recon_x, _, _ = output
        
        mse = F.mse_loss(recon_x, x).item()
        total_mse += mse
    
    return total_mse / len(data_loader)


def visualize_latent_space(latent_features, labels=None, output_path=None, method='pca'):
    """
    Visualize latent space using PCA or UMAP.
    
    Args:
        latent_features: Latent vectors (N, D)
        labels: Class labels (N,)
        output_path: Where to save visualization
        method: 'pca' or 'umap'
    
    Returns:
        2D projection
    """
    if method == 'pca':
        reducer = PCA(n_components=2)
        projection_2d = reducer.fit_transform(latent_features)
    elif method == 'umap' and HAS_UMAP:
        reducer = umap.UMAP(n_components=2)
        projection_2d = reducer.fit_transform(latent_features)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Plot
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 8))
        
        if labels is not None:
            scatter = plt.scatter(projection_2d[:, 0], projection_2d[:, 1],
                                c=labels, cmap='tab20', alpha=0.6)
            plt.colorbar(scatter)
        else:
            plt.scatter(projection_2d[:, 0], projection_2d[:, 1], alpha=0.6)
        
        plt.xlabel(f'{method.upper()} 1')
        plt.ylabel(f'{method.upper()} 2')
        plt.title(f'Latent Space Visualization ({method.upper()})')
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved to {output_path}")
        
        plt.close()
    
    except ImportError:
        print("Matplotlib not available for visualization")
    
    return projection_2d


def main():
    parser = argparse.ArgumentParser(description="Evaluate VAE model")
    parser.add_argument('--model-checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str,
                       help='Path to config file (auto-detected if not provided)')
    parser.add_argument('--test-data', type=str,
                       help='Path to test dataset')
    parser.add_argument('--output', type=str, default='results/',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    print(f"Loading model from {args.model_checkpoint}")
    
    # Load model (basic config)
    model = create_model('vanilla_vae', latent_dim=10)
    model.to(device)
    load_checkpoint(args.model_checkpoint, model, device=device)
    model.eval()
    
    print("Model loaded successfully")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Save metrics
    metrics_path = os.path.join(args.output, 'evaluation_metrics.csv')
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['Model', args.model_checkpoint])
        writer.writerow(['Device', str(device)])
    
    print(f"Evaluation metrics saved to {metrics_path}")


if __name__ == '__main__':
    main()
