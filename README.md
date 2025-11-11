ΨQRH with EinOps: Physically-Grounded Transformer Optimization

Author: Klenio Araujo Padilha
Affiliation: Independent Researcher
Email: klenioaraujo@gmail.com
Date: September 2025 License: GNU GPLv3

This repository presents an optimized implementation of the ΨQRH (Quaternionic Recursive Harmonic) framework—a physically-grounded Transformer architecture for Large Language Models (LLMs). Building on the original ΨQRH project, this version integrates EinOps for tensor manipulation, achieving significant improvements in code clarity, runtime efficiency, and shape safety as detailed in the original paper zenodo.org.

ΨQRH reinterprets Transformers as dynamical physical systems, incorporating quaternion algebra, spectral regularization, fractal modulation, Leech lattice error correction, and energy-conserving processing. This EinOps-optimized branch eliminates fragile reshaping operations (e.g., .view(), .permute(), .unsqueeze()), reduces boilerplate by >60%, and enhances GPU performance while preserving all physical and mathematical grounding.

Key highlights from the paper (Section 4: Enhancing ΨQRH with EinOps):

    Code Refactoring Gains: Reshaping lines reduced from 214 to 82 (-62%).
    Shape Safety: Zero shape-related bugs; runtime-checked operations.
    Performance: Forward pass 9% faster (28.9ms → 26.3ms); lower memory fragmentation.
    Overall Benefits: Empirical accuracy of 93.1% on GLUE-SST2, 25% memory reduction, 2.1× inference speedup.

This implementation maintains the project's philosophy: bridging symbolic mathematics, spectral physics, and deep learning for genuine physical grounding in AI.
Features

    EinOps Integration: Safe, expressive tensor operations with explicit shapes (e.g., rearrange(x, 'b t (h d) -> b t h d', h=n_heads)).
    Quaternionic Representations: Compact, rotation-equivariant 4D encodings with SO(4) isomorphisms.
    Spectral Attention: FFT-based O(n log n) attention with fractal-adaptive filtering.
    Energy Conservation: Explicit norm preservation at every layer for stable training.
    Leech Lattice Encoding: Built-in error correction and parameter compression.
    Padilha Wave Equation: Physical signal evolution with fractal-modulated chirp.
    Fractal Analysis: Differentiable box-counting for geometric priors.
    Hilbert Space Distillation: Genuine transforms for analytic signal processing.
    Production-Ready: GPU/CPU compatible; comprehensive benchmarks and tests.
    Model-Agnostic: Works with external LLMs (e.g., GPT-2) converted to semantic format.

Installation
Requirements

    Python 3.8+
    PyTorch 2.0+
    EinOps (pip install einops)
    Other dependencies: torch, numpy, datasets (for GLUE tasks)

Setup

Clone the repository:
bash

git clone https://github.com/klenioaraujo/ΨQRH-EinOps.git
cd ΨQRH-EinOps

Install dependencies:
bash

pip install -r requirements.txt

For GPU support, ensure CUDA is installed and compatible with PyTorch.
Usage
Quick Start

Run the benchmark for EinOps gains:
bash

python genuine_trained_distillation_transformer.py

This will output:
text

🚀 ΨQRH EINOPS BENCHMARK GAINS (Paper p.4)
==================================================
📊 reshaping_lines             214 → 82 (62%)
📊 shape_bugs               7/week → 0 (100%)
📊 forward_time              28.9ms → 26.3ms (+9%)
📊 memory_fragmentation       High → Low (Significant)
📊 code_safety               Fragile → Runtime-checked (Robust)
==================================================
✅ EinOps integration: COMPLETE AND VALIDATED

Training Example

Load the model and train on GLUE-SST2:
python

import torch
from genuine_trained_distillation_transformer import GenuineTrainedDistillationTransformer
from einops_optimized_training_system import EinOpsOptimizedTrainingSystem
from real_glue_dataset import RealGLUEDataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GenuineTrainedDistillationTransformer(vocab_size=10000, d_model=256, n_layers=3).to(device)

training_system = EinOpsOptimizedTrainingSystem(model, task='sst2')
train_dataset = RealGLUEDataset('sst2', 'train', max_samples=500)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=training_system._real_collate_fn)

train_loss = training_system.train_epoch(train_loader)
print(f"Training Loss: {train_loss:.4f}")

Inference
python

input_ids = torch.tensor([[...]])  # Tokenized input
logits = model(input_ids.to(device))
predictions = torch.argmax(logits, dim=1)

For full pipeline, refer to genuine_trained_distillation_transformer.py.
EinOps Cheat Sheet for ΨQRH
text

from einops import rearrange, reduce, repeat, parse_shape

# Multi-Head Attention
x = rearrange(q_proj, 'b t (h d) -> b t h d', h=8)
x = rearrange(x, 'b t h d -> b t (h d)')

# Batch Flattening (Leech Lattice)
flat = rearrange(x, 'b t d -> (b t) d')
x = rearrange(flat, '(b t) d -> b t d', b=B, t=T)

# Positional Broadcasting
pos_emb = repeat(self.pos_emb[:T], 't d -> b t d', b=B)

# Masked Pooling
mask = rearrange(padding_mask, 'b t -> b t 1')
seq_rep = reduce(x * mask, 'b t d -> b d', 'sum') / reduce(mask, 'b t 1 -> b 1', 'sum').clamp(min=1)

# Spectral Filter Broadcasting
filter = rearrange(filter, 't -> 1 t 1 1')

# Energy Conservation (L2 norm per token)
energy = reduce(x, 'b t d -> b t 1', 'norm')
x = x * (input_energy / (energy + 1e-8))

# Quaternion Stacking
q = rearrange([w, x, y, z], 'c ... -> ... c')

Benchmarks

From the original ΨQRH paper (Section 3: Empirical Results):
Model	Parameters	Accuracy (SST-2)	Memory (GB)	Inference Speed (tokens/s)
Transformer (Vaswani+)	86M	92.7%	12.3	1,240
ΨQRH (Ours)	82M	93.1%	7.3 (-25%)	2,680 (+116%)

EinOps-specific gains (Section 4.3):

    Reshaping code: -62%
    Forward pass: +9% faster
    Memory caching: Improved

Citation

If you use this work, please cite the original ΨQRH paper:
text

@software{Padilha_2025,
  author = {Padilha, Klenio Araujo},
  title = {Reformulating Transformers for LLMs: A Quaternionic-Harmonic Framework with Empirical Validation ΨQRH},
  month = sep,
  year = 2025,
  publisher = {Zenodo},
  version = {1.1},
  doi = {10.5281/zenodo.17171112},
  url = {https://doi.org/10.5281/zenodo.17171112}
}

License

This project is licensed under the GNU General Public License v3.0 (GPLv3). See the LICENSE file for details.
Contact

For questions or collaborations, contact Klenio Araujo Padilha:

    Email: klenioaraujo@gmail.com
    LinkedIn: kleniopadilha
    GitHub: @klenioaraujo

Acknowledgments

This is an extension of the original ΨQRH repository. Special thanks to the EinOps library for enabling production-grade tensor operations.