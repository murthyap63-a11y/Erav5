# ZeRO Memory Partitioning & Communication Primitives Simulation

This repository contains a first-principles simulation of **ZeRO (Zero Redundancy Optimizer) Stages 1, 2, and 3** across 32 virtual process ranks. Built using PyTorch, it calculates and validates the static and transient memory footprints of Large Language Models (LLMs) when scaling parameters across distributed topologies.

---

## Executive Summary

When scaling LLMs to 30B–40B+ parameters, standard Data Parallelism (DDP) hits a hard hardware memory ceiling because parameters, gradients, and optimizer states are duplicated on every GPU.

This project simulates how memory consumption transitions across ZeRO stages:
* **ZeRO-1 ($P_{os}$):** Shards Adam optimizer states across ranks.
* **ZeRO-2 ($P_{os+g}$):** Shards optimizer states and gradients (`Reduce-Scatter`).
* **ZeRO-3 ($P_{pos+g}$):** Shards optimizer states, gradients, and model parameters (`All-Gather` on-the-fly per layer).

---

## Empirical Simulation Results

The simulation was executed on a demo Transformer layer ($8,388,608$ parameters) across **32 virtual ranks** in FP32 precision.

### Console Output
```text
--- Model Parameters: 8,388,608 | Full State Size: 134.22 MB ---

[ZeRO-1 Static Memory per Rank]: 69.21 MB
[ZeRO-2 Static Memory per Rank]: 36.70 MB
[ZeRO-3 Static Memory per Rank (at rest)]: 4.19 MB
[ZeRO-3 Peak Memory during layer forward/backward]: 36.70 MB
```

## Mathematical Memory Proofs ($N_d = 32$ Ranks)

Assuming FP32 precision ($4 \text{ bytes/parameter}$) for a model parameter count of $\Psi = 8,388,608$:

* **Parameters ($P$):** $\Psi \times 4 \text{ bytes} = 33,554,432 \text{ bytes} \approx \mathbf{33.55 \text{ MB}}$
* **Gradients ($G$):** $\Psi \times 4 \text{ bytes} = 33,554,432 \text{ bytes} \approx \mathbf{33.55 \text{ MB}}$
* **Adam Optimizer States ($O_{state}$):** $\Psi \times 8 \text{ bytes} \text{ (Momentum + Variance)} = 67,108,864 \text{ bytes} \approx \mathbf{67.11 \text{ MB}}$
* **Full Unsharded State Baseline:** $P + G + O_{state} = 33.55 + 33.55 + 67.11 = \mathbf{134.22 \text{ MB}}$

---

### Stage-by-Stage Breakdown

1. **ZeRO-1 ($P_{os}$): Optimizer State Partitioning**
   * **Formula:** $\text{Memory}_{Z1} = P + G + \frac{O_{state}}{N_d}$
   * **Calculation:** $33.55 \text{ MB} + 33.55 \text{ MB} + \frac{67.11 \text{ MB}}{32} = 33.55 + 33.55 + 2.10 = \mathbf{69.21 \text{ MB}}$
   * **Reduction:** Sharding the Adam optimizer state reduces total memory footprint by $\sim 48.4\%$ compared to standard DDP.

2. **ZeRO-2 ($P_{os+g}$): Optimizer State + Gradient Partitioning**
   * **Formula:** $\text{Memory}_{Z2} = P + \frac{G}{N_d} + \frac{O_{state}}{N_d}$
   * **Calculation:** $33.55 \text{ MB} + \frac{33.55 \text{ MB}}{32} + \frac{67.11 \text{ MB}}{32} = 33.55 + 1.05 + 2.10 = \mathbf{36.70 \text{ MB}}$
   * **Reduction:** Gradient partitioning via `Reduce-Scatter` removes redundant gradient storage across workers, achieving a $\sim 72.7\%$ total reduction.

3. **ZeRO-3 ($P_{pos+g}$): Parameter + Gradient + Optimizer State Partitioning**
   * **Formula (At Rest):** $\text{Memory}_{Z3\_rest} = \frac{P}{N_d} + \frac{G}{N_d} + \frac{O_{state}}{N_d} = \frac{\text{Full State}}{N_d}$
   * **Calculation:** $\frac{33.55 \text{ MB}}{32} + \frac{33.55 \text{ MB}}{32} + \frac{67.11 \text{ MB}}{32} = 1.05 + 1.05 + 2.10 = \mathbf{4.19 \text{ MB}}$
   * **Formula (Transient Peak):** $\text{Memory}_{Z3\_peak} = P_{\text{layer}} + \frac{G}{N_d} + \frac{O_{state}}{N_d}$
   * **Calculation:** During execution of a layer, `All-Gather` fetches full layer parameters ($33.55 \text{ MB}$), driving peak memory momentarily to $33.55 + 1.05 + 2.10 = \mathbf{36.70 \text{ MB}}$.

---

## Memory Comparison Matrix

| Stage | Parameter Storage | Gradient Storage | Optimizer Storage | Theoretical Exact Expression | Simulated Memory ($N_d=32$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard DDP** | $33.55 \text{ MB}$ | $33.55 \text{ MB}$ | $67.11 \text{ MB}$ | $16\Psi$ | **$134.22 \text{ MB}$** |
| **ZeRO-1** | $33.55 \text{ MB}$ | $33.55 \text{ MB}$ | $2.10 \text{ MB}$ | $4\Psi + 4\Psi + \frac{8\Psi}{N_d}$ | **$69.21 \text{ MB}$** |
| **ZeRO-2** | $33.55 \text{ MB}$ | $1.05 \text{ MB}$ | $2.10 \text{ MB}$ | $4\Psi + \frac{4\Psi}{N_d} + \frac{8\Psi}{N_d}$ | **$36.70 \text{ MB}$** |
| **ZeRO-3 (At Rest)** | $1.05 \text{ MB}$ | $1.05 \text{ MB}$ | $2.10 \text{ MB}$ | $\frac{16\Psi}{N_d}$ | **$4.19 \text{ MB}$** |
| **ZeRO-3 (Peak Layer Step)** | $33.55 \text{ MB}$ | $1.05 \text{ MB}$ | $2.10 \text{ MB}$ | $\Psi_{\text{layer}} + \frac{12\Psi}{N_d}$ | **$36.70 \text{ MB}$** |

---

## Core Technical Insights

1. **Transient Parameter Gathering:** While ZeRO-3 drastically reduces baseline memory at rest to $4.19 \text{ MB}$ ($96.8\%$ savings), execution requires transient memory allocations. Each worker must invoke `dist.all_gather` to reconstruct full parameters before the forward/backward pass of a layer, causing memory to temporarily spike to $36.70 \text{ MB}$ before the weight tensors are discarded.
2. **Communication Primitives Trade-Off:**
   * **ZeRO-1 / ZeRO-2:** Perform gradient reduction (`All-Reduce` or `Reduce-Scatter`) once per backward pass. Parameter weights remain constant in worker memory.
   * **ZeRO-3:** Trades network bandwidth for memory by performing $2 \times$ `All-Gather` ops per layer (1 forward, 1 backward) alongside gradient `Reduce-Scatter`. This setup scales well when network interconnects (e.g., NVLink) are fast enough to keep up with compute throughput.
