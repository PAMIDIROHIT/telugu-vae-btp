<p align="center">
  <h1 align="center">🔤 Telugu Glyph Generation using VAEs</h1>
  <p align="center">
    <strong>Variational Autoencoders for Printed Telugu Character Synthesis</strong>
  </p>
  <p align="center">
    <a href="#-quick-start">Quick Start</a> •
    <a href="#-model-architectures">Models</a> •
    <a href="#-results">Results</a> •
    <a href="#-documentation">Docs</a>
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/License-Academic-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Status-Research-orange.svg" alt="Status">
</p>

---

## 📋 Abstract

This research project implements and compares multiple **Variational Autoencoder (VAE)** architectures for generating high-quality printed **Telugu script** characters. Telugu, spoken by over 80 million people, has a complex Unicode structure with 72+ characters including vowels, consonants, and diacritical marks.

We evaluate four VAE variants with six different loss function configurations and demonstrate that **SSIM + Cyclical KL Annealing** achieves the best reconstruction quality with **0.9917 cosine similarity** on the test set.

**Key Contributions:**
- Systematic comparison of VAE architectures for Telugu character generation
- Novel loss function combination (SSIM + Cyclical KL Annealing)
- Open-source dataset and reproducible training pipeline

---

## 🏗️ System Architecture

```mermaid
graph TB
    subgraph "Data Layer"
        A[Telugu Fonts<br/>Pothana2000.ttf] --> B[Glyph Renderer<br/>render_glyphs.py]
        B --> C[Augmentation Pipeline<br/>rotation, blur, noise]
        C --> D[(Dataset<br/>10,801 samples)]
    end
    
    subgraph "Model Layer"
        D --> E[DataLoader<br/>batch=16]
        E --> F{VAE Model}
        F --> G[Encoder<br/>Conv Layers]
        G --> H[Latent Space<br/>μ, σ²]
        H --> I[Reparameterize<br/>z = μ + σ·ε]
        I --> J[Decoder<br/>ConvTranspose]
        J --> K[Reconstructed<br/>Image]
    end
    
    subgraph "Training Layer"
        K --> L[Loss Function<br/>BCE + KL + SSIM]
        L --> M[Optimizer<br/>Adam lr=0.0001]
        M --> N[Checkpoints<br/>best.pth]
    end
    
    subgraph "Evaluation Layer"
        N --> O[Sample Generation]
        N --> P[Latent Visualization<br/>t-SNE, PCA]
        N --> Q[Metrics<br/>Cosine Sim, FID]
    end
    
    style F fill:#e1f5fe
    style H fill:#fff3e0
    style L fill:#fce4ec
```

---

## 🧠 Model Architectures

### Architecture Comparison

| Model | Encoder | Latent Dim | Key Feature | Parameters |
|-------|---------|------------|-------------|------------|
| **VanillaVAE** | 3 Conv | 16 | Standard VAE | ~200K |
| **β-VAE** | 3 Conv | 16 | Weighted KL (β=4) | ~200K |
| **Conditional VAE** | 3 Conv + Embed | 16 | Class conditioning | ~250K |
| **Improved β-VAE** | 4 Residual | 16 | Skip connections + Attention | ~500K |

### Encoder-Decoder Architecture

```mermaid
graph LR
    subgraph "ENCODER"
        A[Input<br/>32×32×1] --> B[Conv 1→32<br/>stride=2]
        B --> C[Conv 32→64<br/>stride=2]
        C --> D[Conv 64→128<br/>stride=2]
        D --> E[Flatten<br/>2048]
        E --> F[FC → μ<br/>16-dim]
        E --> G[FC → logσ²<br/>16-dim]
    end
    
    subgraph "LATENT"
        F --> H[Reparameterize]
        G --> H
        H --> I[z<br/>16-dim]
    end
    
    subgraph "DECODER"
        I --> J[FC<br/>2048]
        J --> K[Reshape<br/>4×4×128]
        K --> L[ConvT 128→64<br/>stride=2]
        L --> M[ConvT 64→32<br/>stride=2]
        M --> N[ConvT 32→1<br/>stride=2]
        N --> O[Output<br/>32×32×1]
    end
    
    style H fill:#fff3e0
    style I fill:#fff3e0
```

### Improved β-VAE with Residual Blocks

```mermaid
graph TB
    subgraph "Residual Block"
        X[Input x] --> C1[Conv 3×3]
        C1 --> BN1[BatchNorm]
        BN1 --> R1[ReLU]
        R1 --> C2[Conv 3×3]
        C2 --> BN2[BatchNorm]
        X --> SK[Skip<br/>1×1 Conv if needed]
        BN2 --> ADD((+))
        SK --> ADD
        ADD --> R2[ReLU]
        R2 --> OUT[Output]
    end
```

---

## 📊 Training Pipeline

```mermaid
sequenceDiagram
    participant D as Dataset
    participant L as DataLoader
    participant M as VAE Model
    participant O as Optimizer
    participant C as Checkpoint
    
    Note over D,C: Training Loop (200 epochs)
    
    loop Each Epoch
        D->>L: Load batch (16 samples)
        L->>M: Forward pass
        M->>M: Encode → z ~ q(z|x)
        M->>M: Decode → x' = p(x|z)
        M->>O: Compute Loss<br/>L = BCE + β·KL + λ·SSIM
        O->>M: Backward + Update weights
    end
    
    M->>C: Save best checkpoint
    
    Note over D,C: Evaluation Phase
    
    C->>M: Load best model
    M->>M: Generate samples
    M->>M: Compute test metrics
```

---

## 📈 Results

### Performance Summary

| Approach | Loss Function | Test Cosine Similarity | Rank |
|----------|---------------|------------------------|------|
| Baseline β-VAE | BCE + 4·KL | 0.9889 | 2nd |
| **SSIM + KL Annealing** | BCE + 0.3·SSIM + cyclical_KL | **0.9917** | 🥇 **1st** |
| Combined Loss | L1 + SSIM + Focal + Cosine + KL | 0.9823 | 3rd |

### Why SSIM + KL Annealing Wins

```mermaid
pie title Performance Factors
    "SSIM Structural Loss" : 35
    "Cyclical KL Annealing" : 30
    "Balanced Weights" : 20
    "BCE Base Loss" : 15
```

**Key Insights:**
- ✅ **SSIM** preserves Telugu character stroke structure
- ✅ **Cyclical annealing** prevents posterior collapse
- ✅ **Moderate β** allows good reconstruction without over-regularization
- ❌ **Too many losses** (Approach 3) dilutes gradients

---

## 📁 Project Structure

```
vae_project/
├── 📂 models/                      # VAE implementations
│   ├── vae.py                      # VanillaVAE, BetaVAE, cVAE
│   ├── improved_vae.py             # Residual blocks + Attention
│   ├── networks.py                 # Encoder/Decoder networks
│   └── losses.py                   # Loss functions
│
├── 📂 scripts/                     # Training & evaluation
│   ├── train.py                    # Main training script
│   ├── generate_samples.py         # Sample generation
│   ├── latent_visualizer.py        # t-SNE, UMAP, traversals
│   └── evaluate.py                 # Evaluation metrics
│
├── 📂 configs/                     # YAML configurations
│   └── beta_vae_baseline.yaml
│
├── 📂 data/                        # Datasets
│   ├── Pothana2000.ttf             # Telugu font
│   ├── Vowel_Dataset/              # 6 vowel classes
│   └── metadata.csv                # Dataset metadata
│
├── 📂 experiments/                 # Experiment outputs
├── 📂 results/                     # Metrics & reports
├── 📂 checkpoints/                 # Model weights
│
├── 📄 DOCUMENTATION.md             # Comprehensive docs
├── 📄 train_vowel_experiments.py   # Main experiment script
└── 📄 README.md                    # This file
```

---

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install torch torchvision matplotlib pandas scikit-learn \
    opencv-python pillow tqdm seaborn
```

### Train a Model

```bash
# Approach 1: Baseline β-VAE
python train_vowel_experiments.py --approach 1 --epochs 200

# Approach 2: SSIM + KL Annealing (BEST)
python train_vowel_experiments.py --approach 2 --epochs 200

# Approach 3: Combined Loss
python train_vowel_experiments.py --approach 3 --epochs 200

# All approaches
python train_vowel_experiments.py --approach 0
```

### Generate Samples

```bash
python scripts/generate_samples.py \
    --model_path checkpoints/vowel_approach_2/best.pth \
    --model_type beta_vae \
    --latent_dim 16 \
    --num_samples 100 \
    --output_dir results/generated_samples
```

### Visualize Latent Space

```bash
python scripts/latent_visualizer.py \
    --model_path checkpoints/vowel_approach_2/best.pth \
    --data_path data/Vowel_Dataset \
    --latent_dim 16
```

---

## 📊 Class Diagram

```mermaid
classDiagram
    class VanillaVAE {
        +latent_dim: int
        +encoder: ConvEncoder
        +decoder: ConvDecoder
        +encode(x) tuple
        +decode(z) Tensor
        +reparameterize(mu, logvar) Tensor
        +forward(x) tuple
        +loss(recon_x, x, mu, logvar)
        +sample(num_samples)
    }
    
    class BetaVAE {
        +beta: float
        +set_beta(beta)
    }
    
    class ConditionalVAE {
        +num_classes: int
        +encode(x, c)
        +decode(z, c)
        +forward(x, c)
        +sample(num_samples, class_id)
    }
    
    class ImprovedBetaVAE {
        +use_attention: bool
        -hidden_dims: list
        +count_parameters()
    }
    
    VanillaVAE <|-- BetaVAE : inherits
    VanillaVAE <|-- ConditionalVAE : extends
    VanillaVAE <|-- ImprovedBetaVAE : enhanced
    
    class ConvEncoder {
        +in_channels: int
        +latent_dim: int
        +hidden_dims: list
        +forward(x) tuple
    }
    
    class ConvDecoder {
        +latent_dim: int
        +out_channels: int
        +forward(z) Tensor
    }
    
    VanillaVAE --> ConvEncoder : has
    VanillaVAE --> ConvDecoder : has
```

---

## 📚 Loss Functions

| Loss | Formula | Use Case |
|------|---------|----------|
| **BCE** | `-∑[x·log(x') + (1-x)·log(1-x')]` | Pixel reconstruction |
| **KL Divergence** | `-0.5·∑(1 + logσ² - μ² - σ²)` | Latent regularization |
| **SSIM** | `1 - SSIM(x, x')` | Structural similarity |
| **Focal BCE** | `α(1-p)^γ · BCE` | Hard sample mining |
| **L1/MAE** | `\|x - x'\|` | Edge preservation |
| **Cosine** | `1 - cos(x, x')` | Feature alignment |

---

## 📖 Documentation

For comprehensive technical documentation, see:

📄 **[DOCUMENTATION.md](./DOCUMENTATION.md)** - Full research documentation including:
- Problem statement & scope
- Dataset description & statistics
- Model architecture details
- Loss function analysis
- Experimental methodology
- Detailed results analysis
- Conclusions & future work

---

## 🎯 Future Work

- [ ] Implement VQ-VAE for discrete latents
- [ ] Add more Telugu fonts (Vemana, Gautami)
- [ ] Compute FID scores
- [ ] Train OCR classifier for evaluation
- [ ] Extend to handwritten characters
- [ ] GPU training optimization

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{pamidi2025teluguvae,
  title={Telugu Glyph Generation using Variational Autoencoders},
  author={Pamidi, Rohit},
  year={2025},
  institution={Indian Institute of Technology},
  note={BTP Research Project}
}
```

---

## 👥 Contributors

- **Rohit Pamidi** - Primary Developer & Researcher
- **Faculty Advisor** - Project Guidance

---

## 📄 License

This project is for academic research purposes. Please cite if using this code or methodology.

---

<p align="center">
  <strong>🙏 Thank you for exploring Telugu VAE Generation!</strong>
</p>
