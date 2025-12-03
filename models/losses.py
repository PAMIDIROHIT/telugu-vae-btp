"""
Loss functions for VAE models.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE loss = Reconstruction Loss + β * KL Divergence
    
    Args:
        recon_x: Reconstructed output from decoder
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: Weight for KL divergence term (1.0 for vanilla VAE, >1 for β-VAE)
    
    Returns:
        Total loss, reconstruction loss, KL loss
    """
    # Reconstruction loss (Binary Cross Entropy)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='mean')
    
    # KL Divergence loss: KL(N(mu, sigma^2) || N(0, 1))
    # = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    total_loss = BCE + beta * KLD
    
    return total_loss, BCE, KLD


def cvae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    Conditional VAE loss (same as VAE loss).
    The conditioning is handled during forward pass.
    
    Args:
        recon_x: Reconstructed output
        x: Original input
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution
        beta: KL weight
    
    Returns:
        Total loss, reconstruction loss, KL loss
    """
    return vae_loss(recon_x, x, mu, logvar, beta)


def reconstruction_loss_l2(recon_x, x):
    """L2 reconstruction loss (MSE)."""
    return F.mse_loss(recon_x, x, reduction='mean')


def reconstruction_loss_bce(recon_x, x):
    """Binary Cross Entropy reconstruction loss."""
    return F.binary_cross_entropy(recon_x, x, reduction='mean')
