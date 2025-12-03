# Telugu Glyph Generation using VAEs

**BTP Research Project**: Variational Autoencoders for Printed Telugu Character Synthesis

**Goal**: Train VAE models to generate high-quality printed Telugu glyphs and publish findings in a top-tier vision conference (ICDAR/CVPR-W/ICFHR).

---

## 📋 Project Overview

This project implements and compares three Variational Autoencoder (VAE) architectures for learning and generating Telugu script characters:

1. **Vanilla VAE**: Standard VAE with reconstruction + KL divergence loss
2. **β-VAE**: Weighted VAE encouraging disentangled latent representations
3. **Conditional VAE (cVAE)**: Class-conditioned VAE for controlled generation

### Dataset

- **Fonts**: Pothana, Akshara (Telugu fonts)
- **Size**: 10,801 glyph samples
- **Variations**: Multiple font sizes (12pt-20pt), rotations, blur, pixel shifts
- **Format**: 64×64 grayscale PNG images

---

## 🗂️ Project Structure

```
vae_project/
├── models/                     # VAE model implementations
│   ├── __init__.py
│   ├── vae.py                  # VanillaVAE, BetaVAE, ConditionalVAE
│   ├── networks.py             # Encoder/Decoder architectures
│   └── losses.py               # Loss functions
├── scripts/                    # Training and evaluation scripts
│   ├── train.py                # Main training script
│   ├── generate_samples.py     # Sample generation from trained models
│   ├── latent_visualizer.py    # Latent space analysis (t-SNE, UMAP, traversals)
│   ├── evaluate.py             # Model evaluation
│   ├── render_glyphs.py        # Dataset generation
│   └── utils.py                # Utility functions
├── configs/                    # Training configurations (YAML)
│   └── beta_vae_baseline.yaml
├── data/                       # Dataset
│   ├── raw/                    # Raw glyph images
│   │   └── metadata.csv        # Dataset metadata
│   └── fonts/                  # Telugu font files
├── checkpoints/                # Model checkpoints (.pth files)
├── experiments/                # Experiment logs and configs
├── results/                    # Generated samples and visualizations
├── logs/                       # Training logs
└── paper/                      # LaTeX paper and figures
```

---

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Activate virtual environment
source ~/vae_env/bin/activate

# Ensure all dependencies are installed
pip install torch torchvision matplotlib pandas scikit-learn \
    opencv-python pillow tqdm umap-learn seaborn
```

### 2. Dataset Status

✅ Dataset already generated: **10,801 Telugu glyph samples**

To regenerate or add more fonts:
```bash
python scripts/render_glyphs.py
```

### 3. Train a Model

```bash
# Train Beta-VAE baseline
python scripts/train.py --config configs/beta_vae_baseline.yaml

# Monitor training
tail -f logs/beta_vae_baseline_training.log
```

### 4. Generate Samples

Once a model is trained:

```bash
# Generate 100 samples from trained Beta-VAE
python scripts/generate_samples.py \
    --model_path checkpoints/beta_vae_best.pth \
    --model_type beta_vae \
    --latent_dim 32 \
    --num_samples 100 \
    --output_dir results/generated_samples
```

### 5. Visualize Latent Space

```bash
# Analyze latent space with t-SNE, UMAP, and traversals
python scripts/latent_visualizer.py \
    --model_path checkpoints/beta_vae_best.pth \
    --model_type beta_vae \
    --data_path data/raw/metadata.csv \
    --latent_dim 32 \
    --num_samples 1000
```

---

## 📊 Experiments & Ablation Studies

### Planned Ablation Grid

| Model Type   | Latent Dim | β Value | Status |
|--------------|-----------|---------|--------|
| Vanilla VAE  | 32        | 1.0     | ⏳ Pending |
| Beta-VAE     | 16        | 2.0     | ⏳ Pending |
| Beta-VAE     | 32        | 2.0     | ⏳ Pending |
| Beta-VAE     | 32        | 5.0     | ⏳ Pending |
| Beta-VAE     | 64        | 2.0     | ⏳ Pending |
| cVAE         | 32        | 1.0     | ⏳ Pending |

### Evaluation Metrics

- **FID Score**: Fréchet Inception Distance for generation quality
- **OCR Accuracy**: Can generated glyphs fool a classifier trained on real data?
- **Latent Disentanglement**: t-SNE/UMAP clustering, traversal smoothness
- **Reconstruction Loss**: MSE/BCE on validation set

---

## 📝 Research Paper

Target conferences:
- **ICDAR** (International Conference on Document Analysis and Recognition)
- **CVPR Workshops** (Computer Vision and Pattern Recognition)
- **ICFHR** (International Conference on Frontiers in Handwriting Recognition)

Paper sections:
1. Abstract & Introduction
2. Related Work (VAEs, Script Generation, Low-Resource Languages)
3. Dataset Creation Methodology
4. Model Architectures
5. Experimental Setup & Ablation Studies
6. Results & Analysis
7. Conclusion & Future Work

---

## 🔧 Training Configuration Example

```yaml
# configs/beta_vae_baseline.yaml
model:
  type: "beta_vae"
  latent_dim: 32
  beta: 2.0
  
training:
  batch_size: 64
  num_epochs: 100
  learning_rate: 0.001
  device: "cuda"
  
data:
  dataset_path: "data/raw/"
  metadata_path: "data/raw/metadata.csv"
  train_split: 0.8
```

---

## 📈 Current Progress

- [x] Dataset generation (10,801 samples)
- [x] Model implementations (VAE, β-VAE, cVAE)
- [x] Training pipeline
- [x] Sample generation script
- [x] Latent visualization tools
- [ ] Train baseline models
- [ ] Run ablation studies
- [ ] Compute FID scores
- [ ] OCR evaluation
- [ ] Write research paper
- [ ] Prepare for submission

---

## 🤝 Git Workflow

```bash
# Check status
git status

# Add all changes
git add .

# Commit with descriptive message
git commit -m "Description of changes"

# Push to remote (after setting up GitHub repo)
git push origin main
```

---

## 📚 Key References

1. Kingma & Welling (2014) - Auto-Encoding Variational Bayes
2. Higgins et al. (2016) - β-VAE: Learning Basic Visual Concepts
3. Sohn et al. (2015) - Learning Structured Output Representation using Deep Conditional Generative Models

---

## 👤 Author

**Student**: Rohit (BTP Project)  
**Institution**: [Your Institution]  
**Year**: 2024-2025

---

## 📄 License

Academic research project. Please cite if using this code or methodology.

---

## 🎯 Next Steps

1. **Train Baseline**: Run `train_baseline.sh` to start training
2. **Monitor Progress**: Check `logs/` and `experiments/` directories
3. **Generate & Evaluate**: Once trained, generate samples and compute metrics
4. **Iterate**: Adjust hyperparameters based on results
5. **Document**: Keep updating results in paper drafts

**For questions or issues, consult your advisor or create an issue in the repository.**
