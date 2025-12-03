"""
Variational Autoencoder implementations:
- Vanilla VAE
- β-VAE (beta-VAE)
- Conditional VAE (cVAE)
"""
import torch
import torch.nn as nn
from .networks import ConvEncoder, ConvDecoder, ConditionalEncoder, ConditionalDecoder
from .losses import vae_loss, cvae_loss


class VanillaVAE(nn.Module):
    """
    Standard Variational Autoencoder.
    
    Architecture:
    - Encoder: Conv layers → Latent (μ, σ²)
    - Sampling: z ~ N(μ, σ²) using reparameterization trick
    - Decoder: Latent → Conv layers → Reconstructed image
    - Loss: Reconstruction + KL divergence
    """
    
    def __init__(self, latent_dim=10, in_channels=1, hidden_dims=None):
        super(VanillaVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.encoder = ConvEncoder(in_channels, latent_dim, hidden_dims)
        self.decoder = ConvDecoder(latent_dim, in_channels, hidden_dims)
    
    def encode(self, x):
        """Encode image to latent distribution parameters."""
        return self.encoder(x)
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = μ + σ * ε
        where ε ~ N(0, 1)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """Decode latent vector to reconstructed image."""
        return self.decoder(z)
    
    def forward(self, x):
        """
        Args:
            x: Input image (batch_size, 1, 64, 64)
        
        Returns:
            recon_x: Reconstructed image
            mu: Latent mean
            logvar: Latent log-variance
            z: Sampled latent vector
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar, z
    
    def loss(self, recon_x, x, mu, logvar, beta=1.0):
        """Compute VAE loss."""
        return vae_loss(recon_x, x, mu, logvar, beta)
    
    def sample(self, num_samples, device='cpu'):
        """Generate new samples from standard normal prior."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)


class BetaVAE(VanillaVAE):
    """
    β-VAE: Weighted Variational Autoencoder.
    
    Encourages disentanglement by weighting KL divergence term with β > 1.
    Loss: Reconstruction + β * KL divergence
    
    References:
    - Higgins et al. (2016): β-VAE: Learning Basic Visual Concepts with 
      a Constrained Variational Framework
    """
    
    def __init__(self, latent_dim=10, beta=1.0, in_channels=1, hidden_dims=None):
        super(BetaVAE, self).__init__(latent_dim, in_channels, hidden_dims)
        self.beta = beta
    
    def loss(self, recon_x, x, mu, logvar):
        """Compute β-VAE loss with weighted KL term."""
        return vae_loss(recon_x, x, mu, logvar, beta=self.beta)
    
    def set_beta(self, beta):
        """Update β parameter (useful for annealing)."""
        self.beta = beta


class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder (cVAE).
    
    Extends VAE to be conditioned on external labels (e.g., glyph class, font).
    Both encoder and decoder are conditioned on the label.
    
    Useful for learning style-separated representations where:
    - Latent factors capture style variations
    - Conditioning captures class-specific information
    """
    
    def __init__(self, latent_dim=10, num_classes=128, in_channels=1, hidden_dims=None):
        super(ConditionalVAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        self.encoder = ConditionalEncoder(in_channels, latent_dim, num_classes, hidden_dims)
        self.decoder = ConditionalDecoder(latent_dim, in_channels, num_classes, hidden_dims)
    
    def encode(self, x, c):
        """Encode image and condition to latent distribution."""
        return self.encoder(x, c)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z, c):
        """Decode latent vector and condition to image."""
        return self.decoder(z, c)
    
    def forward(self, x, c):
        """
        Args:
            x: Input image (batch_size, 1, 64, 64)
            c: Class condition (batch_size,)
        
        Returns:
            recon_x: Reconstructed image
            mu: Latent mean
            logvar: Latent log-variance
            z: Sampled latent vector
        """
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, c)
        return recon_x, mu, logvar, z
    
    def loss(self, recon_x, x, mu, logvar, beta=1.0):
        """Compute cVAE loss."""
        return cvae_loss(recon_x, x, mu, logvar, beta)
    
    def sample(self, num_samples, class_id, device='cpu'):
        """Generate samples for a specific class."""
        z = torch.randn(num_samples, self.latent_dim, device=device)
        c = torch.full((num_samples,), class_id, dtype=torch.long, device=device)
        return self.decode(z, c)


def create_model(model_type, latent_dim=10, num_classes=None, beta=1.0, in_channels=1):
    """
    Factory function to create VAE models.
    
    Args:
        model_type: 'vanilla_vae', 'beta_vae', or 'cvae'
        latent_dim: Dimensionality of latent space
        num_classes: Number of classes (required for cVAE)
        beta: Beta parameter (for β-VAE)
        in_channels: Number of input channels (usually 1 for grayscale)
    
    Returns:
        Instantiated model
    """
    if model_type == 'vanilla_vae':
        return VanillaVAE(latent_dim, in_channels)
    elif model_type == 'beta_vae':
        return BetaVAE(latent_dim, beta, in_channels)
    elif model_type == 'cvae':
        if num_classes is None:
            raise ValueError("num_classes must be specified for cVAE")
        return ConditionalVAE(latent_dim, num_classes, in_channels)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
