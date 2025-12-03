#!/bin/bash
# Quick Training Script for VAE Baseline Models
# 
# This script trains baseline VAE models for the Telugu glyph generation project.
# Modify configurations in configs/ directory as needed.

set -e  # Exit on error

PROJECT_DIR="/home/mohanganesh/vae_project"
cd "$PROJECT_DIR"

# Activate virtual environment
source ~/vae_env/bin/activate

echo "========================================="
echo "Training VAE Baseline Models"
echo "========================================="
echo ""

# Check if dataset exists
if [ ! -f "data/raw/metadata.csv" ]; then
    echo "ERROR: Dataset not found!"
    echo "Please run: python scripts/render_glyphs.py first"
    exit 1
fi

# Count dataset samples
NUM_SAMPLES=$(wc -l < data/raw/metadata.csv)
echo "✓ Dataset found: $NUM_SAMPLES samples"
echo""

# Create necessary directories
mkdir -p checkpoints logs experiments results configs

# Function to train a model
train_model() {
    CONFIG=$1
    MODEL_NAME=$2
    
    echo "----------------------------------------"
    echo "Training: $MODEL_NAME"
    echo "Config: $CONFIG"
    echo "----------------------------------------"
    
    if [ ! -f "$CONFIG" ]; then
        echo "⚠ Config file not found: $CONFIG"
        echo "Skipping..."
        return
    fi
    
    python scripts/train.py --config "$CONFIG" 2>&1 | tee "logs/${MODEL_NAME}_training.log"
    
    if [ $? -eq 0 ]; then
        echo "✅ $MODEL_NAME training complete!"
    else
        echo "❌ $MODEL_NAME training failed!"
    fi
    echo ""
}

# Default: Train Beta-VAE baseline
echo "Starting baseline model training..."
echo ""

# Beta-VAE with latent_dim=32, beta=2.0
if [ ! -f "configs/beta_vae_baseline.yaml" ]; then
    echo "Creating default Beta-VAE config..."
    cat > configs/beta_vae_baseline.yaml << 'EOF'
model:
  type: "beta_vae"
  latent_dim: 32
  beta: 2.0
  in_channels: 1
  hidden_dims: [32, 64, 128]

training:
  batch_size: 64
  num_epochs: 100
  learning_rate: 0.001
  lr_scheduler: "cosine"
  device: "cuda"
  num_workers: 4

data:
  dataset_path: "data/raw/"
  metadata_path: "data/raw/metadata.csv"
  train_split: 0.8

logging:
  experiment_dir: "experiments/beta_vae_baseline"
  checkpoint_dir: "checkpoints/"
  
seed: 42
EOF
fi

# Uncomment to train models:
# train_model "configs/beta_vae_baseline.yaml" "beta_vae_baseline"

echo "========================================="
echo "Training script ready!"
echo "========================================="
echo ""
echo "To train a model, uncomment the train_model line in this script"
echo "or run directly:"
echo "  python scripts/train.py --config configs/beta_vae_baseline.yaml"
echo ""
echo "To train multiple configurations for ablation studies,"
echo "create additional YAML files in configs/ and add train_model calls."
echo ""
