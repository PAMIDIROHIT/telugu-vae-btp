"""
Utility functions for the project.
"""
import os
import json
import yaml
import random
import numpy as np
import torch


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load configuration from YAML or JSON file."""
    if config_path.endswith('.yaml') or config_path.endswith('.yml'):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        raise ValueError("Config must be .yaml or .json")
    return config


def save_config(config, save_path):
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def create_experiment_dir(base_dir, experiment_name):
    """Create experiment directory with timestamped folder."""
    from datetime import datetime
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(exp_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "samples"), exist_ok=True)
    
    return exp_dir


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, device='cpu'):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint.get('epoch', 0), checkpoint.get('loss', None)


def save_git_info(git_hash, save_path):
    """Save git commit hash to file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(f"Git Commit Hash: {git_hash}\n")


def get_git_hash():
    """Get current git commit hash."""
    try:
        import subprocess
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            capture_output=True,
            text=True
        )
        return result.stdout.strip() if result.returncode == 0 else "Unknown"
    except:
        return "Unknown"


class Logger:
    """Simple CSV logger for training metrics."""
    
    def __init__(self, log_path):
        self.log_path = log_path
        self.metrics = {}
    
    def log(self, epoch, **kwargs):
        """Log metrics for an epoch."""
        if epoch not in self.metrics:
            self.metrics[epoch] = {'epoch': epoch}
        
        self.metrics[epoch].update(kwargs)
        
        # Save to CSV
        self._save_csv()
    
    def _save_csv(self):
        """Save metrics to CSV file."""
        import csv
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
        if not self.metrics:
            return
        
        # Get all unique keys
        all_keys = set()
        for epoch_data in self.metrics.values():
            all_keys.update(epoch_data.keys())
        
        all_keys = sorted(list(all_keys))
        
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_keys)
            writer.writeheader()
            
            for epoch in sorted(self.metrics.keys()):
                writer.writerow(self.metrics[epoch])


def denormalize_image(img, mean=0.5, std=0.5):
    """Denormalize image from [-1, 1] or [0, 1] range."""
    if isinstance(img, torch.Tensor):
        img = img.clone()
        if img.dim() == 4:  # Batch
            img = img.cpu()
        img = img * std + mean
        return torch.clamp(img, 0, 1)
    else:
        img = np.array(img, dtype=np.float32)
        img = img * std + mean
        return np.clip(img, 0, 1)


def normalize_image(img, mean=0.5, std=0.5):
    """Normalize image to [-1, 1] or [0, 1] range."""
    if isinstance(img, torch.Tensor):
        img = img.clone().float()
        img = (img - mean) / std
        return img
    else:
        img = np.array(img, dtype=np.float32)
        img = (img - mean) / std
        return img


def print_model_summary(model):
    """Print model summary."""
    print("\n" + "="*50)
    print(f"Model: {model.__class__.__name__}")
    print("="*50)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("="*50 + "\n")
