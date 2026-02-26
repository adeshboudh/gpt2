# GPT-2 From Scratch 🧠

A clean reimplementation of GPT-2 (124M) trained from scratch in PyTorch,
following [Andrej Karpathy's](https://github.com/karpathy/build-nanogpt)
"Let's reproduce GPT-2" tutorial.

> 🚧 **Status:** Training script is complete and ready.
> Waiting on A100 GPU access — full training run scheduled in ~2 days.

---

## Overview

This project replicates the GPT-2 (124M) pretraining pipeline end-to-end:

- Custom GPT-2 model architecture built from scratch in PyTorch
- Pretraining on the **FineWeb-Edu 10B** token dataset
- Evaluation on **HellaSwag** commonsense reasoning benchmark
- Multi-GPU training via **DDP (DistributedDataParallel)**

---

# GPT-2 124M — Pretrained from Scratch

Trained on FineWeb-Edu 10B tokens using Andrej Karpathy's build-nanogpt.

## Results

| Step | Val Loss | HellaSwag |
| ---- | -------- | --------- | --------------------- |
| 0    | 10.9517  | 0.2482    |
| 1000 | 4.1443   | 0.2573    |
| 3000 | 3.4432   | 0.2717    |
| 5000 | ~3.35    | ~0.279    |
| 5449 | —        | —         | ← run ended (credits) |

GPT-2 124M official baseline: HellaSwag = 0.2955

## Hardware

- Lightning AI H100 (1x)
- ~2.6 hrs to step 5449

---

## Project Structure

```

.
├── train_gpt2.py       \# Model architecture + full training loop
├── fineweb.py          \# Dataset download, tokenization \& sharding
├── index.txt           \# tiny_shakespeare dataset (1.1M chars ~ 338k tokens)
├── hellaswag.py        \# HellaSwag benchmark evaluation
├── playground.ipynb    \# Playground notebook
├── log/                \# Training logs and checkpoints
├── gpt2_training.ipynb \# Training notebook for Kaggle GPU
├── .env.example        \# Template for environment variables
├── .gitignore
└── README.md

```

---

## Model Architecture

| Hyperparameter                | Value                                          |
| ----------------------------- | ---------------------------------------------- |
| Layers (`n_layer`)            | 12                                             |
| Attention heads (`n_head`)    | 12                                             |
| Embedding dim (`n_embd`)      | 768                                            |
| Context length (`block_size`) | 1024                                           |
| Vocab size                    | 50,304 (padded from 50,257 for GPU efficiency) |
| Total parameters              | ~124M                                          |

### Key Design Choices

- **Pre-LayerNorm** architecture (LN before attention & MLP)
- **Flash Attention** via `F.scaled_dot_product_attention(..., is_causal=True)`
- **Weight tying** between token embedding (`wte`) and `lm_head`
- **Scaled residual init**: projection layers use `std *= (2 * n_layer)^-0.5`
- **GELU (tanh approx)** activation in MLP

---

## Training Configuration

| Setting                     | Value                                      |
| --------------------------- | ------------------------------------------ |
| Dataset                     | FineWeb-Edu (`sample-10BT`)                |
| Total batch size            | 524,288 tokens (2¹⁹)                       |
| Micro-batch size            | B=64, T=1024                               |
| Gradient accumulation steps | 8 (single GPU)                             |
| Optimizer                   | AdamW (fused), β=(0.9, 0.95), ε=1e-8       |
| Weight decay                | 0.1 (matrices/embeddings), 0.0 (biases/LN) |
| Max LR                      | 6e-4                                       |
| Min LR                      | 6e-5                                       |
| LR schedule                 | Linear warmup (715 steps) → Cosine decay   |
| Total steps                 | 19,073 (~1 epoch over 10B tokens)          |
| Precision                   | BF16 (via `torch.autocast`)                |
| Gradient clipping           | 1.0                                        |

---

## Setup

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/your-username/build-gpt2.git
cd build-gpt2

pip install torch tiktoken numpy datasets tqdm transformers huggingface_hub python-dotenv
```

### 2. Configure HuggingFace Token

```bash
cp .env.example .env
# Edit .env and add your HF token:
# HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx
```

Get your token at: https://huggingface.co/settings/tokens (Read access is sufficient)

### 3. Prepare the Dataset

```bash
python fineweb.py
# Downloads and tokenizes FineWeb-Edu into ./edu_fineweb10B/ ~23-25GB
# ~100 shards × 100M tokens each (uint16 .npy files)
```

---

## Training

### Single GPU

```bash
python train_gpt2.py
```

### Multi-GPU (DDP)

```bash
torchrun --standalone --nproc_per_node=8 train_gpt2.py
```

Checkpoints are saved every 5,000 steps to `log/model_XXXXX.pt`.
Logs (train loss, val loss, HellaSwag accuracy) are written to `log/log.txt`.

---

## Evaluation

### HellaSwag Benchmark

```bash
# Evaluate a HuggingFace GPT-2 checkpoint
python hellaswag.py --model_type gpt2 --device cuda
```

| Model                        | acc    | acc_norm |
| :--------------------------- | :----- | :------- |
| GPT-2 124M (OpenAI)          | 28.92% | 31.14%   |
| This implementation (target) | ~28.6% | ~29.6%   |

---

## What Gets Logged (Every 250 Steps)

- ✅ Validation loss (averaged over 20 batches)
- ✅ HellaSwag accuracy (`acc_norm`)
- ✅ Sample text generation (top-k=50, 4 sequences of 32 tokens)

---

## Hardware Target

|                          | Spec                         |
| :----------------------- | :--------------------------- |
| GPU                      | NVIDIA A100 (80GB)           |
| Expected training time   | ~1 day for full 19,073 steps |
| Estimated final val loss | ~3.28 (matching GPT-2 paper) |

---

## References

- [Andrej Karpathy — build-nanogpt](https://github.com/karpathy/build-nanogpt)
- [Original GPT-2 Paper — Radford et al., 2019](https://openai.com/research/language-unsupervised)
- [FineWeb-Edu Dataset — HuggingFaceFW](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- [HellaSwag Benchmark — Zellers et al., 2019](https://arxiv.org/abs/1905.07830)

---

## License

MIT
