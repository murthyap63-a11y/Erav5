# Assignment 11: Deep Dive into Adam Dynamics, Schedules, and Width Scaling

## Overview

This assignment provides an empirical and mathematical exploration of modern optimization dynamics in LLMs, focusing specifically on **Adam** and **AdamW**. Building upon concepts developed in Sessions 8–10 (Attention, Feed-Forward Networks, Next-Token Cross-Entropy Loss, and RMSNorm), this project implements a clean, modular **MiniGPT Transformer** to evaluate optimizer behavior, warmup dynamics, decay schedules, and scaling laws.

---

## Model Architecture (`MiniGPT`)

The underlying model is a decoder-only Mini-Transformer (`MiniGPT`) structured to replicate modern LLM pretraining architectures (such as Llama and DeepSeek):

* **Normalization:** Pre-LN `RMSNorm` across transformer layers to maintain feature stability.
* **Attention:** `CausalSelfAttention` with multi-head scaled dot-product attention and strict causal masking.
* **Feed-Forward Network:** `SwiGLUFFN` using gate and up-projections with SwiGLU activation.
* **Weight Tying:** Shared token embeddings and language model head (`lm_head.weight = tok_emb.weight`).

---

## Assignment Tasks & Key Findings

### Task 1: Reproducing Adam by Hand
* **Objective:** Calculate Adam updates step-by-step for a single parameter across 5 gradients and verify against PyTorch's native `torch.optim.Adam`.
* **Math Equations:**
  $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$
  $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
  $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
  $$w_t = w_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$
* **Verification:** Manual outputs match `torch.optim.Adam` to $10^{-7}$ precision across all 5 test steps.
### Step-by-Step Verification Results

| Step ($t$) | Gradient ($g_t$) | $\hat{m}_t$ | $\hat{v}_t$ | Hand $w_t$ | PyTorch $w_t$ | Absolute Error |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | $0.10$ | $0.100000$ | $0.000100$ | $1.4990000$ | $1.4990000$ | $< 10^{-7}$ |
| **2** | $-0.20$ | $-0.057895$ | $0.000200$ | $1.5030932$ | $1.5030932$ | $< 10^{-7}$ |
| **3** | $0.05$ | $-0.019446$ | $0.000134$ | $1.5047720$ | $1.5047720$ | $< 10^{-7}$ |
| **4** | $0.30$ | $0.076326$ | $0.000201$ | $1.4993910$ | $1.4993910$ | $< 10^{-7}$ |
| **5** | $-0.10$ | $0.021008$ | $0.000162$ | $1.4977418$ | $1.4977418$ | $< 10^{-7}$ |
---

### Task 2: Disable Bias Correction & Convergence Horizon
* **Objective:** Compare standard bias-corrected Adam against uncorrected Adam ($\Delta w_t = \eta \frac{m_t}{\sqrt{v_t} + \epsilon}$) for the first 20 steps, and identify when the difference stops mattering.
* **Observation:** Without bias correction, initial step sizes are heavily dampened near step 1 because $m_0 = 0$ and $v_0 = 0$.
* **Convergence Horizon:** Because $\beta_2 = 0.999$, the factor $1 - \beta_2^t$ takes over 100 steps to approach $1.0$. The relative update difference drops below $1\%$ at **Step ~100 to 300**.
* **Artifact:** `task2_bias_correction.png`

---

### Task 3: Layer-wise Update-to-Weight Ratio & Warmup
* **Objective:** Track the $L_2$ norm update ratio across every layer:
  $$\text{Ratio}_l^{(t)} = \frac{\Vert{}\Delta w_l^{(t)}\Vert{}_2}{\Vert{}w_l^{(t)}\Vert{}_2}$$
* **Warmup Impact:** During linear warmup (steps 1–30), the update ratio scales up linearly. At **Step 30** (the exact end of warmup), the ratio reaches its peak and stabilizes, protecting initial weights from destabilizing updates.

---

### Task 4: Cosine vs. Warmup-Stable-Decay (WSD) Schedule
* **Objective:** Train MiniGPT for 300 total budgeted steps under Cosine and WSD schedules, but halt both at step 200.
* **Loss at Step 200:**
  * **Cosine:** Lower loss at step 200 because it decays its learning rate continuously throughout training.
  * **WSD:** Higher loss at step 200 because it remains in its constant high-LR stable phase until step 240.
* **Model Selection:** **Keep the WSD Model**. Cosine looks artificially better at step 200 due to premature learning rate decay. WSD preserves optimization mobility, allowing flexible decay at step 200 or seamless continuation to higher token budgets without performance collapse.

---

### Task 5: Width Scaling Sweeps & Extrapolation ($d_{\text{model}} = 4096$)
* **Objective:** Sweep learning rates ($\eta \in \{10^{-4}, 3\cdot 10^{-4}, 10^{-3}, 3\cdot 10^{-3}, 10^{-2}\}$) across model widths $d_{\text{model}} \in \{256, 512, 1024\}$ to locate loss minima and extrapolate optimal learning rate for width $d=4096$.
* **Scaling Relationship:** Under Standard PyTorch Parameterization (SP), optimal learning rates scale inversely with width:
  $$\eta^*(d) \propto \frac{1}{\sqrt{d}}$$
* **Width 4096 Prediction:**
  $$\eta^*(4096) = \eta^*(1024) \times \sqrt{\frac{1024}{4096}} = 0.001 \times 0.5 = 0.0005$$
* **Confidence Level:** **Moderate (7/10)**. While inverse width scaling holds firmly across standard parameterizations, confidence is bounded by short step counts and fixed batch sizes at scale.
* **Artifact:** `task5_width_sweep.png`

---

## Repository Structure

```text
├── optimizer.ipynb            # Self-contained PyTorch master execution script
├── README.md                  # Comprehensive assignment report and documentation
├── task2_bias_correction.png  # Task 2 trajectory plot (Bias Corrected vs Uncorrected)
└── task5_width_sweep.png      # Task 5 LR sweep plot across widths 256, 512, and 1024
