# Assignment 9: Transformer Loss Harness & Multi-Head Token Prediction

This repository contains the full implementation and benchmark analysis for **Assignment 9**, covering memory-optimized loss computation and multi-token prediction heads for Causal Language Models (**ToyGPT**).

---

## 📂 Repository Structure

```text
assignment-9-loss-harness/
│
├── submission_artifacts/
│   └── shards/                 # Exported evaluation and training shards
│
├── src/
│   ├── models/
│   │   └── toygpt.py           # ToyGPT architecture with dual-head prediction
│   │
│   ├── losses/
│   │   └── chunked_loss.py     # Memory-optimized Chunked Cross-Entropy implementation
│   │
│   └── utils/
│       └── metrics.py          # CUDA peak memory and profiling hooks
│
├── benchmark.py                # Profiling script to compare Standard vs. Chunked loss
├── train_simulation.py         # 50-step simulation running multi-head optimization
├── requirements.txt            # Project dependencies
└── README.md                   # Project documentation
```

---

## 📌 Part 1: Causal Self-Attention & Chunked Cross-Entropy

### Overview
In standard language model training, projecting hidden states of shape `(B, L, D)` to full vocabulary logits `(B, L, V)` requires significant VRAM allocation when sequence length ($L$) and vocabulary size ($V$) scale up. 

To resolve this, we implemented **Chunked Cross-Entropy**, which slices hidden representations along the temporal sequence dimension $L$ into micro-chunks before computing logit projections and loss values. This reduces peak VRAM consumption without altering the exact mathematical output of cross-entropy loss.

### PyTorch Code Implementation
The optimized loss loop chunks tensors sequentially to bypass multi-gigabyte logit instantiations:

```python
import torch
import torch.nn as nn

def chunked_cross_entropy(hidden_states, lm_head, targets, chunk_size=16):
    """
    Computes Cross Entropy Loss by chunking hidden_states along the sequence dimension
    to minimize peak memory usage during large-vocabulary projections.
    """
    b, l, d = hidden_states.size()
    total_loss = 0.0
    
    # Process the sequence dimension in sequential micro-chunks
    for i in range(0, l, chunk_size):
        # 1. Slice current micro-chunk along sequence dimension (L)
        hidden_chunk = hidden_states[:, i:i+chunk_size, :] # Shape: (B, Chunk, D)
        target_chunk = targets[:, i:i+chunk_size]          # Shape: (B, Chunk)
        
        # 2. Project chunk to vocabulary size
        logits_chunk = lm_head(hidden_chunk)               # Shape: (B, Chunk, V)
        
        # 3. Enforce layout contiguity before flattening for CrossEntropyLoss
        logits_flat = logits_chunk.contiguous().view(-1, logits_chunk.size(-1))
        targets_flat = target_chunk.contiguous().view(-1)
        
        # 4. Compute partial loss scaled by chunk size proportion
        loss = nn.functional.cross_entropy(logits_flat, targets_flat)
        scaled_loss = loss * (hidden_chunk.size(1) / l)
        
        # 5. Accumulate loss
        total_loss += scaled_loss
        
    return total_loss
```

### Code Highlights & Bug Fixes
* **Tensor Strides & Contiguity:** Fixed PyTorch memory layout exceptions during non-contiguous slice reshaping by replacing `.view()` with `.contiguous().view()` across sequence targets and sliced hidden projections.
* **CUDA Overhead & Cache Handling:** Utilized `torch.cuda.empty_cache()` and `torch.cuda.reset_peak_memory_stats()` between evaluation passes to accurately isolate peak VRAM usage under CUDA hardware acceleration.

### Memory Optimization Benchmark Results (Deliverable 7)
*Executed on NVIDIA T4 GPU via Google Colab runtime:*

| Method | Peak GPU Memory | Reduction Ratio |
| :--- | :---: | :---: |
| *Standard Cross-Entropy* | *2102.88 MB* (~2.10 GB) | Baseline (1.00x) |
| *Chunked Cross-Entropy* | *1282.92 MB* (~1.28 GB) | **1.64x** |

> 💡 **Key Takeaway:** Chunked cross-entropy reduced peak VRAM consumption by **1.64x** on identical sequence inputs by avoiding the instantiation of the full $(B 	imes L 	imes V)$ logit matrix in GPU memory during backpropagation.

---

## 🧠 Part 2: Dual-Head Prediction ($t+1$ vs $t+2$)

### Overview
We extended the ToyGPT architecture with two separate linear output heads to analyze multi-step forward token prediction:
1. **Head 1 ($	ext{Head}_{t+1}$):** Predicts the immediate next token ($t+1$).
2. **Head 2 ($	ext{Head}_{t+2}$):** Predicts the skip-step token ($t+2$) directly from the current hidden state $h_t$.

### Training Metrics Log (50 Steps Simulation)

| Step | Loss 1 ($t+1$) | Loss 2 ($t+2$) | Total Loss |
| :---: | :---: | :---: | :---: |
| *0* | 4.2644 | 4.3646 | 8.6290 |
| *10* | 4.4037 | 4.4195 | 8.8232 |
| *20* | 4.2305 | 4.2754 | 8.5059 |
| *30* | 4.2381 | 4.2735 | 8.5116 |
| *40* | 4.2027 | 4.2127 | 8.4154 |
| *50* | **4.1508** | **4.1700** | **8.3207** |

### Theoretical Analysis
Throughout optimization, **$	ext{Loss}_2$ ($t+2$) remained consistently higher than $	ext{Loss}_1$ ($t+1$)**. 

* **Reason:** Predicting a token two steps ahead ($t+2$) without seeing the intermediate token ($t+1$) introduces higher conditional entropy (skip-gram uncertainty). The model cannot resolve step $t+2$ with the same certainty because the immediate contextual transition at $t+1$ is hidden.

---

## ⚡ How to Run & Reproduce

### 1. Installation
Ensure you have PyTorch installed with CUDA capabilities:
```bash
pip install torch triton transformers
```

### 2. Run Memory Benchmark
To verify the **1.64x memory savings** and evaluate the maximum peak GPU statistics on your device run:
```bash
python benchmark.py --batch_size 4 --seq_len 1024 --chunk_size 16
```

### 3. Run Multi-Head Optimization
To replicate the 50-step simulation log documenting loss patterns for intermediate vs skip tokens:
```bash
python train_simulation.py --steps 50 --lr 0.001
```
