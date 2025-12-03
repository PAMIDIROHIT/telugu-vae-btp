# Paper Writing Guide

## Overview

This folder contains all materials for writing the academic paper on Variational Autoencoders for Indic Scripts (Telugu Fonts).

## Directory Structure

```
paper/
├── README.md              # This file
├── latex/                 # LaTeX source files
│   ├── main.tex          # Main paper document
│   ├── sections/         # Section files (imported in main.tex)
│   │   ├── abstract.tex
│   │   ├── introduction.tex
│   │   ├── related_work.tex
│   │   ├── methodology.tex
│   │   ├── experiments.tex
│   │   ├── results.tex
│   │   ├── conclusion.tex
│   │   └── appendix.tex
│   ├── references.bib    # BibTeX bibliography
│   └── preamble.tex      # LaTeX packages and macros
├── figures/              # High-quality figures for paper
│   ├── vae_architecture.pdf
│   ├── latent_traversals.pdf
│   ├── reconstruction_grid.pdf
│   ├── fid_comparison.pdf
│   └── ...
├── tables/               # Tables (CSV → LaTeX)
│   ├── ablation_study.csv
│   ├── metrics_comparison.csv
│   └── ...
└── supplement/           # Supplementary materials
    ├── additional_results.pdf
    └── sample_generations.pdf
```

## Compilation

### Option 1: Using pdflatex locally

```bash
cd paper/latex
pdflatex main.tex
bibtex main.aux
pdflatex main.tex
pdflatex main.tex
```

The compiled PDF will be `main.pdf`.

### Option 2: Using Overleaf

1. Create new project on Overleaf
2. Upload all files from `latex/` directory
3. Set main file to `main.tex`
4. Compile online

### Option 3: Using Docker

```bash
docker run --rm -v $(pwd):/workspace -w /workspace/paper/latex \
  texlive/texlive:latest \
  bash -c "pdflatex main.tex && bibtex main.aux && pdflatex main.tex && pdflatex main.tex"
```

## Paper Sections

### 1. Abstract (150-250 words)
- Summarize: Problem, Approach, Contribution, Results
- File: `latex/sections/abstract.tex`

### 2. Introduction (2-3 pages)
- Motivate the problem (Indic script recognition/generation)
- Related challenges (font variation, data scarcity)
- Project goals and contributions
- File: `latex/sections/introduction.tex`

### 3. Related Work (1-2 pages)
- VAE literature
- Generative models for fonts
- Indic script processing
- Data augmentation techniques

### 4. Methodology (2-3 pages)
- Data generation pipeline
- VAE architectures (Vanilla, β-VAE, cVAE)
- Training procedure
- Evaluation metrics

### 5. Experiments & Results (2-3 pages)
- Dataset statistics
- Training details
- Quantitative results (FID, KID, OCR accuracy)
- Qualitative analysis (latent traversals, reconstructions)
- Ablation studies

### 6. Conclusion (1 page)
- Summarize contributions
- Limitations
- Future work

### 7. Appendix (if needed)
- Additional figures
- Detailed proofs
- Hyperparameter tables

## Figures & Tables

### Creating Figures
1. Generate figures from `results/` directory
2. Export as PDF (preferred) or high-res PNG (300 dpi)
3. Save in `paper/figures/`
4. Reference in LaTeX: `\includegraphics[width=0.8\textwidth]{figures/figure_name.pdf}`

### Creating Tables
1. Export evaluation metrics to CSV (e.g., `results/metrics.csv`)
2. Convert to LaTeX table using:
   ```bash
   python scripts/csv_to_latex.py results/metrics.csv
   ```
3. Place in `paper/tables/` and import: `\input{tables/metrics_table.tex}`

### Figure Captions & References
```latex
\begin{figure}
  \centering
  \includegraphics[width=0.8\textwidth]{figures/vae_architecture.pdf}
  \caption{Schematic of Variational Autoencoder architecture. 
           Encoder (left) maps input $x$ to latent distribution $(\mu, \sigma^2)$, 
           from which $z$ is sampled. Decoder (right) reconstructs $\hat{x}$ from $z$.}
  \label{fig:vae_architecture}
\end{figure}

See Figure \ref{fig:vae_architecture} for the architecture diagram.
```

## Bibliography Management

### Adding References
1. Edit `latex/references.bib`
2. Add entries in BibTeX format:
   ```bibtex
   @inproceedings{kingma2013,
     title={Auto-Encoding Variational Bayes},
     author={Kingma, D. P. and Welling, M.},
     booktitle={ICLR},
     year={2013}
   }
   ```
3. Cite in text: `\cite{kingma2013}` or `\citep{kingma2013}`

### Key Papers to Cite
- Kingma & Welling (2013): VAE - https://arxiv.org/abs/1312.6114
- Higgins et al. (2016): β-VAE - https://openreview.net/forum?id=Sy2fzU9gl
- Heusel et al. (2017): FID - https://arxiv.org/abs/1706.08500
- Srivastava et al. (2014): KID - https://arxiv.org/abs/1401.4054

## LaTeX Tips & Tricks

### Equations
```latex
% Inline: $z \sim \mathcal{N}(\mu, \sigma^2)$
% Display:
\begin{equation}
  L = \mathbb{E}_q[\log p(x|z)] - D_{KL}(q(z|x) \| p(z))
  \label{eq:vae_loss}
\end{equation}

Reference: See Equation \ref{eq:vae_loss}.
```

### Code Listings
```latex
\begin{lstlisting}[language=Python, caption=Rendering glyphs]
font = ImageFont.truetype('Pothana.ttf', size)
draw.text((x, y), glyph, font=font, fill=255)
\end{lstlisting}
```

### Tables
```latex
\begin{table}[h]
  \centering
  \begin{tabular}{|c|c|c|c|}
    \hline
    Model & FID & KID & OCR Acc. \\
    \hline
    Vanilla VAE & 45.2 & 0.12 & 87.3\% \\
    β-VAE (β=1) & 42.1 & 0.10 & 88.1\% \\
    \hline
  \end{tabular}
  \caption{Comparison of VAE models}
  \label{tab:model_comparison}
\end{table}
```

## Submission Checklist

### For Conference Submission (e.g., ICDAR, CVPR)

- [ ] PDF compiles without errors
- [ ] All figures referenced and high-resolution
- [ ] All tables included with proper captions
- [ ] All citations in BibTeX
- [ ] Page limit respected (e.g., 8-10 pages)
- [ ] Double-blind review requirements met (no author names in text)
- [ ] Figures use colorblind-friendly palettes
- [ ] References follow conference format

### Camera-Ready Submission

Create a ZIP file with:
```
submission.zip
├── main.tex
├── main.pdf
├── sections/
│   ├── abstract.tex
│   ├── introduction.tex
│   └── ...
├── figures/
│   ├── *.pdf
│   └── *.png
└── references.bib
```

Upload to conference system with signed copyright form.

### arXiv Submission

1. Prepare arXiv-compatible files:
   ```bash
   cd paper/latex
   mkdir arXiv_submission
   cp main.tex arXiv_submission/
   cp -r figures arXiv_submission/
   cp references.bib arXiv_submission/
   ```

2. Upload to arXiv (https://arxiv.org)
3. Reference in paper as: "arXiv preprint arXiv:YYMM.NNNNN"

## Timeline

- **Week 1-2**: Draft Abstract, Introduction, Related Work
- **Week 3-4**: Finalize Methodology section, generate figures
- **Week 5-6**: Complete Experiments & Results, ablation analysis
- **Week 7**: First draft review, figures/tables finalization
- **Week 8**: Revisions, Conclusion, camera-ready prep

## Resources

- **ICDAR Guidelines**: https://icdar.org/
- **Overleaf LaTeX Tutorials**: https://www.overleaf.com/learn
- **IEEE/Springer LaTeX Templates**: Available from conference websites
- **BibTeX Reference**: http://www.ctan.org/tex-archive/biblio/bibtex/

---

**Last Updated**: December 2024

For questions or template updates, contact your advisor or thesis committee.
