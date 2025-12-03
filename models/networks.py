"""
Network modules for encoder and decoder.
"""
import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    """
    Convolutional encoder for glyph images.
    Compresses images to latent mean and log-variance.
    """
    
    def __init__(self, in_channels=1, latent_dim=10, hidden_dims=None):
        super(ConvEncoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]
        
        self.latent_dim = latent_dim
        
        # Build encoder blocks
        modules = []
        current_dim = in_channels
        
        for hidden_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Conv2d(current_dim, hidden_dim, 4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU()
            ))
            current_dim = hidden_dim
        
        self.encoder = nn.Sequential(*modules)
        
        # Calculate flattened size (assuming 64x64 input)
        # After each stride-2 conv: 64 -> 32 -> 16 -> 8
        self.flatten_size = hidden_dims[-1] * 8 * 8
        
        # Latent space projection
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
    
    def forward(self, x):
        """
        Args:
            x: Input image (batch_size, 1, 64, 64)
        
        Returns:
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
        """
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class ConvDecoder(nn.Module):
    """
    Convolutional decoder for glyph images.
    Reconstructs images from latent vectors.
    """
    
    def __init__(self, latent_dim=10, out_channels=1, hidden_dims=None):
        super(ConvDecoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        self.latent_dim = latent_dim
        self.fc_hidden_size = hidden_dims[0] * 8 * 8
        
        # Linear layer to expand latent vector
        self.fc = nn.Linear(latent_dim, self.fc_hidden_size)
        
        # Build decoder blocks
        modules = []
        current_dim = hidden_dims[0]
        
        for i, hidden_dim in enumerate(hidden_dims[1:], 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(current_dim, hidden_dim, 4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU()
            ))
            current_dim = hidden_dim
        
        # Final layer to reconstruct image
        modules.append(nn.Sequential(
            nn.ConvTranspose2d(current_dim, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid()  # Sigmoid for pixel values in [0, 1]
        ))
        
        self.decoder = nn.Sequential(*modules)
    
    def forward(self, z):
        """
        Args:
            z: Latent vector (batch_size, latent_dim)
        
        Returns:
            Reconstructed image (batch_size, 1, 64, 64)
        """
        x = self.fc(z)
        x = x.view(x.size(0), -1, 8, 8)
        x = self.decoder(x)
        return x


class ConditionalEncoder(nn.Module):
    """
    Conditional encoder that takes both image and condition.
    """
    
    def __init__(self, in_channels=1, latent_dim=10, num_classes=128, hidden_dims=None):
        super(ConditionalEncoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Embedding for class condition
        self.class_embedding = nn.Embedding(num_classes, 64)
        
        # Image encoder
        modules = []
        current_dim = in_channels
        
        for hidden_dim in hidden_dims:
            modules.append(nn.Sequential(
                nn.Conv2d(current_dim, hidden_dim, 4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU()
            ))
            current_dim = hidden_dim
        
        self.encoder = nn.Sequential(*modules)
        self.flatten_size = hidden_dims[-1] * 8 * 8
        
        # Concatenate embedded condition before projection
        self.fc_input_size = self.flatten_size + 64
        
        self.fc_mu = nn.Linear(self.fc_input_size, latent_dim)
        self.fc_logvar = nn.Linear(self.fc_input_size, latent_dim)
    
    def forward(self, x, c):
        """
        Args:
            x: Input image (batch_size, 1, 64, 64)
            c: Class condition (batch_size,)
        
        Returns:
            mu: Mean of latent distribution
            logvar: Log-variance of latent distribution
        """
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        
        # Embed and concatenate condition
        c_embed = self.class_embedding(c)
        x = torch.cat([x, c_embed], dim=1)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class ConditionalDecoder(nn.Module):
    """
    Conditional decoder that takes both latent vector and condition.
    """
    
    def __init__(self, latent_dim=10, out_channels=1, num_classes=128, hidden_dims=None):
        super(ConditionalDecoder, self).__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        # Embedding for class condition
        self.class_embedding = nn.Embedding(num_classes, 64)
        
        # Concatenate latent with condition
        self.fc_input_size = latent_dim + 64
        self.fc_hidden_size = hidden_dims[0] * 8 * 8
        
        self.fc = nn.Linear(self.fc_input_size, self.fc_hidden_size)
        
        # Decoder blocks
        modules = []
        current_dim = hidden_dims[0]
        
        for hidden_dim in hidden_dims[1:]:
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(current_dim, hidden_dim, 4, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU()
            ))
            current_dim = hidden_dim
        
        modules.append(nn.Sequential(
            nn.ConvTranspose2d(current_dim, out_channels, 4, stride=2, padding=1),
            nn.Sigmoid()
        ))
        
        self.decoder = nn.Sequential(*modules)
    
    def forward(self, z, c):
        """
        Args:
            z: Latent vector (batch_size, latent_dim)
            c: Class condition (batch_size,)
        
        Returns:
            Reconstructed image (batch_size, 1, 64, 64)
        """
        c_embed = self.class_embedding(c)
        z = torch.cat([z, c_embed], dim=1)
        
        x = self.fc(z)
        x = x.view(x.size(0), -1, 8, 8)
        x = self.decoder(x)
        return x
