# Transformer Internals: Gradients, Accumulation & Numerical Stability

This repository contains a comprehensive technical investigation into the mechanics of Transformer architectures, autograd execution, gradient behavior, and hardware performance metrics.

---

## Task 1: Numerical Gradient Verification (Gradient Checking)

### Methodology & Mathematical Foundation

To verify PyTorch's `autograd` symbolic differentiation engine, we compute the analytical gradient \(\nabla W\) and compare it against the empirical numerical gradient using central finite differences:

$$
\frac{\partial L}{\partial W_{i,j}} \approx \frac{L(W_{i,j} + \epsilon) - L(W_{i,j} - \epsilon)}{2\epsilon}
$$

### Empirical Verification Output

Testing the linear projection layer weight tensor `linear_layer.weight[0, 0]` in single-precision floating-point arithmetic (`FP32`) with perturbation step \(\epsilon = 10^{-4}\):

```text
Target Parameter: linear_layer.weight[0, 0]
Analytical Gradient (autograd): 0.07582111
Numerical Gradient (finite diff): 0.07629395
Absolute Difference:             4.72836196e-04
Relative Difference:             6.19755778e-03 (~0.6%)
```

### Technical Explanation of the Discrepancy

The relative discrepancy at the 3rd decimal place (\(\approx 0.6\%\)) is expected behavior caused by **Taylor Series Truncation Error**:

$$
\frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon} = f'(x) + \frac{\epsilon^{2}}{6}\, f'''(x) + O(\epsilon^{4})
$$

1. **Finite Step Truncation:** Central finite differences introduce an \(O(\epsilon^{2})\) approximation error. Higher-order derivatives (\(f'''(x)\)) of non-linear activations (`CrossEntropyLoss` and `Softmax`) introduce truncation errors in the \(10^{-3}\) to \(10^{-4}\) range when using \(\epsilon = 10^{-4}\).

2. **Floating-Point Precision Limits:** In single precision (`FP32`), numerical catastrophic cancellation occurs when subtracting nearly identical values \(L(W + \epsilon) - L(W - \epsilon)\). Re-running the evaluation in 64-bit precision (`FP64`) with \(\epsilon = 10^{-6}\) collapses the relative difference below \(10^{-6}\), proving that PyTorch's symbolic autograd engine is exact.

---

## Task 2: Gradient Accumulation with Variable Micro-Batches

### Methodology

When scaling effective batch sizes via gradient accumulation over micro-batches of variable sequence lengths, simple unweighted loss averaging introduces severe mathematical bias into parameter updates.

- **Broken Method (Average of Averages):**

$$
L_{\text{step}} = \frac{1}{M} \sum_{m=1}^{M} L_{m}
$$

  *(Treats all micro-batches equally regardless of token volume).*

- **Correct Method (Token-Weighted Loss):**

$$
L_{\text{step}} = \frac{\sum_{m=1}^{M} L_{m} \cdot N_{m}}{\sum_{m=1}^{M} N_{m}}
$$

  *(Scales micro-batch losses proportional to total valid target tokens \(N_{m}\)).*

### Empirical Plot & Divergence Analysis

When evaluating Micro-Batch A (\(N_{A} = 8\) tokens) against Micro-Batch B (\(N_{B} = 64\) tokens), unweighted averaging assigns 50% influence to \(N_{A}\), despite \(N_{A}\) representing only \(\frac{8}{72} \approx 11.1\%\) of total tokens.

```text
Loss
  ^
  |      /\--/\  <-- Broken: Average of Averages (Red Dashed)
  |  /\ /  \/  \
  | /  V  /\    \
  |/____\/__\____\ <-- Correct: Token-Weighted Loss (Green Solid)
  +--------------------------------------------------> Step
```

**Key Takeaway:** Unweighted averaging over-emphasizes small micro-batches, distorting gradients and causing optimization instability. Token-weighted scaling ensures gradient magnitudes match an equivalent monolithic batch execution.

---

## Task 3: Global Gradient Norm vs. Loss Trajectory

### Methodology

During Transformer optimization, we track both the scalar cross-entropy loss \(L\) and the total Euclidean norm of all parameter gradients:

$$
\|\nabla W\|_{2} = \sqrt{\sum_{p \in \Theta} \sum_{i} \left( \frac{\partial L}{\partial p_{i}} \right)^{2}}
$$

### Step Log Observations

Evaluating 30 training steps reveals critical disconnects between loss movement and gradient magnitude:

```text
Step  8 | Loss: 4.0538 | Grad Norm: 0.5228
Step  9 | Loss: 4.0346 (Δ 0.0192) | Grad Norm: 0.5661 (Δ 0.0433) <-- Grad Norm spikes 8.3% while Loss drops 0.4%

Step 28 | Loss: 3.9695 | Grad Norm: 0.4978
Step 29 | Loss: 3.9748 (Δ 0.0052) | Grad Norm: 0.5558 (Δ 0.0580) <-- Grad Norm jumps 11.6% while Loss is flat
```

### Curvature & Landscape Dynamics

- **Loss Scalar (\(L\)):** Measures the absolute height on the current loss surface.
- **Gradient Norm (\(\|\nabla W\|_{2}\)):** Measures local surface steepness/curvature.
- **Insight:** At **Step 9** and **Step 29**, the optimizer enters high-curvature "cliffs" or steep valleys. The gradient norm increases significantly *before* the scalar loss changes, demonstrating that gradient norm acts as an early indicator of landscape shifts.

---

## Task 4: Model FLOPs Utilization (MFU) Analysis

### Mathematical Formulation

For a Transformer model with \(N\) non-embedding parameters, \(L\) layers, \(H\) heads, head dimension \(Q\), and sequence length \(S\), the total compute cost per token (forward + backward pass) is:

$$
\text{FLOPs per Token} \approx 6N + 12LHQS
$$

$$
\text{Total Step FLOPs} = \text{FLOPs per Token} \times (\text{Batch Size} \times S)
$$

$$
\text{MFU} = \frac{\text{Achieved TFLOPS}}{\text{Theoretical Peak GPU TFLOPS}} = \frac{\text{Total Step FLOPs}/\text{Step Time}}{\text{Peak GPU TFLOPS}}
$$

### Benchmarking Results

- **Non-Embedding Parameters (\(N\)):** 3,672,040
- **Tokens per Step:** 1,024 (\(B = 8\), \(S = 128\))
- **Total Step FLOPs:** 24,196,481,024 FLOPs \(\approx\) 24.2 GFLOPs
- **Measured CPU Step Time:** 748.07 ms
- **Achieved Compute Performance:** 0.0323 TFLOPS (32.3 GFLOPS)
- **Theoretical Benchmark (NVIDIA T4 FP16 Peak):** 65.0 TFLOPS
- **Reported Baseline MFU:** 0.05%

### "Costing the Distance to 40% MFU"

Achieving state-of-the-art 30%–40% MFU requires addressing key system bottlenecks:

1. **Hardware Execution Bounds:** Standard CPU execution is memory-bandwidth bound (0.0323 TFLOPS). Transitioning to CUDA Tensor Cores provides massive parallel execution capabilities.

2. **Memory Bandwidth Bottlenecks (Non-MatMul Ops):** `LayerNorm`, `GELU`, and element-wise additions spend more time fetching bytes from high-bandwidth memory (HBM) than performing operations.

3. **Kernel Launch Overhead:** Small batch sizes (1,024 tokens) fail to saturate thousands of CUDA cores. CPU thread scheduling overhead dominates execution time.

4. **Intermediate Activation IO:** Standard attention materializes \(S \times S\) attention matrices in VRAM. Fusing kernels via FlashAttention-2 and `torch.compile()` keeps intermediate tensors inside fast GPU SRAM/L1 caches, dramatically accelerating step time toward target MFU levels.

---

## Task 5: Floating-Point Bit Representations & Precision Selection

### Bit-Level Representation of Decimal 0.1

Decimal \(0.1_{10}\) is an infinite repeating fraction in binary:

$$
0.1_{10} = 0.0001100110011\ldots_{2} = 1.1001100110011\ldots_{2} \times 2^{-4}
$$

### 1. IEEE 754 Single Precision (FP32) — 32 bits

- **Structure:** `1 Sign Bit | 8 Exponent Bits (Bias 127) | 23 Mantissa Bits`
- **Sign Bit:** `0`
- **Exponent:** \(-4 + 127 = 123_{10} = 01111011_{2}\)
- **Mantissa:** `10011001100110011001101`
- **Bit Sequence:**

```text
0 | 01111011 | 10011001100110011001101
```

### 2. Brain Floating Point (BF16) — 16 bits

- **Structure:** `1 Sign Bit | 8 Exponent Bits (Bias 127) | 7 Mantissa Bits`
- **Sign Bit:** `0`
- **Exponent:** \(-4 + 127 = 123_{10} = 01111011_{2}\)
- **Mantissa:** `1001101`
- **Bit Sequence:**

```text
0 | 01111011 | 1001101
```

### 3. FP8 E4M3 (OCP Standard) — 8 bits

- **Structure:** `1 Sign Bit | 4 Exponent Bits (Bias 7) | 3 Mantissa Bits`
- **Sign Bit:** `0`
- **Exponent:** \(-4 + 7 = 3_{10} = 0011_{2}\)
- **Mantissa:** `101`
- **Bit Sequence:**

```text
0 | 0011 | 101
```

### Training Precision Recommendation

**Selected Format: BF16 Mixed Precision (with FP32 Master Weights)**

**Technical Rationale:**

1. **Dynamic Range Preservation:** BF16 maintains the exact same 8-bit exponent width as FP32, providing a dynamic range (\(\approx 10^{-38}\) to \(10^{38}\)). This prevents gradient underflow and overflow during backpropagation without requiring complex dynamic loss scaling.

2. **Memory Throughput:** BF16 cuts memory bandwidth usage in half relative to FP32, doubling computational throughput on modern Tensor Cores while preserving training stability.

3. **Limitation of FP8 E4M3:** FP8 E4M3 has only 4 exponent bits (bias 7), giving it a narrow dynamic range. While useful for forward activations or low-precision inference, using FP8 across deep backpropagation loops risks severe gradient underflow unless paired with fine-grained per-tensor scaling factors.

---

## Repository Structure

```text
.
├── Assignment10_Transformer_Internals.ipynb   # Complete execution notebook with code & plots
└── README.md                                 # Technical documentation and analysis report
```

## How to Run

1. Open `Assignment10_Transformer_Internals.ipynb` in Google Colab or local Jupyter environment.
2. Execute cells sequentially. All dependencies rely on standard PyTorch and Matplotlib distributions.
3. Notebook runs efficiently on standard CPU runtimes.
