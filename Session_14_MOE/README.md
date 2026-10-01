# Dense-to-Sparse MoE Transformer Upcycling

An end-to-end PyTorch implementation demonstrating **Sparse Upcycling**: transitioning a **20M-parameter dense Transformer** into a **4-expert Mixture-of-Experts (MoE) model** midway through a 50M-token training budget. Designed for execution on Google Colab using a T4 GPU.

---

## 📌 Project Strategy & Training Overview

The project splits a **50 Million Token budget** across three core execution phases:

1. **Phase 1: Dense Pre-training (Tokens 0 – 25M)**
   * Trains a standard ~20M parameter Transformer baseline utilizing SwiGLU Feed-Forward Networks (FFNs).
2. **Phase 2: Upcycling Event (25M Token Mark / Step 762)**
   * Converts all dense `SwiGLUFFN` layers into 4-expert `MoEFFN` layers.
   * Copies dense FFN weights across all 4 experts and applies small Gaussian noise ($\sigma = 0.01$) to break initial symmetry.
   * Initializes a top-1 router and updates the optimizer to track newly instantiated parameters.
3. **Phase 3: Sparse MoE Fine-Tuning (Tokens 25M – 50M)**
   * Continues training the sparse MoE architecture over the remaining 25M tokens, allowing the router to learn specialized expert routing and further reducing loss.

---

## 🏗️ Model Architecture

| Feature | Dense Phase (Phase 1) | Upcycled MoE Phase (Phase 3) |
| :--- | :--- | :--- |
| **Total Model Parameters** | ~20 Million | ~80 Million |
| **Active Parameters / Token** | ~20 Million | ~20 Million (Top-1 Routing) |
| **Hidden Dimension ($d_{\text{model}}$)** | 384 | 384 |
| **Attention Heads** | 6 | 6 |
| **FFN Expansion Ratio** | SwiGLU ($d_{\text{ffn}} = 1536$) | 4 x SwiGLU Experts ($d_{\text{ffn}} = 1536$) |
| **Routing Mechanism** | N/A | Top-1 Softmax Router |

---

## ⚡ Infrastructure & Memory Optimizations

Running MoE architectures on standard hardware can easily trigger CUDA Out-Of-Memory (OOM) errors. This script includes two key memory optimizations tailored for Colab's 16GB T4 GPU:

* **Gradient Accumulation:** Uses a micro-batch size of `16` with `4` accumulation steps to achieve an effective batch size of `64` ($64 \times 512 = 32,768$ tokens/step) while staying well under VRAM thresholds.
* **3D Logit Loss Transposition:** Avoids allocating large flattened 2D logit matrices during cross-entropy evaluation by using `logits.transpose(1, 2)` directly in standard tensor dimensions.
* **Atomic Checkpointing:** Saves state dictionaries atomically (`.tmp` $\rightarrow$ final overwrite) to Google Drive every 200 steps to support seamless resuming if the Colab runtime disconnects.

---

## 📁 Repository Structure & Storage

```text
├── dense_moe_upcycling.py        # Complete standalone PyTorch training script
├── README.md                     # Project documentation
└── Google Drive Output Directory (/MyDrive/MoE_Assignment_Checkpoints/)
    ├── dense_moe_upcycle.pt      # Latest training checkpoint state dictionary
    └── upcycling_loss_plot.png   # Generated training loss curve

