# Observability of a training step

**ERA V5 · Session 10**  
Finite differences, token-weighted accumulation, gradient-norm causality, MFU, and the bit layout of `0.1`.

Numbers below are from the local CPU run of `session10_gradients.py` / `.ipynb`.

---

## Abstract

A trainer is a map from tokens to a scalar, then from that scalar onto every weight. Session 9 asked whether the scalar was the correct next-token cross-entropy. This notebook asks four sharper questions about the rest of the step:

1. Does `autograd` differentiate the loss we think it does?
2. When micro-batches have unequal length, does the optimizer see a token-average or an average of averages?
3. Can the residual stream move before the logged loss does?
4. What fraction of peak FLOPs does this loop burn, and in which floating-point box would we store `0.1`?

Vehicle: 2-block causal Transformer, `d_model = 64`, `V = 65`, \(N = 109{,}312\), AdamW on synthetic token IDs. Instrumentation, not language quality.

---

## 0. Forward contract

\[
\mathcal{L}
= \mathrm{CE}\big(\mathrm{logits}_{:,0:L-1,:},\; x_{:,1:L}\big)
\]

`ToyGPT.forward` returns `(hidden, logits)`. No tying. No dropout. Device: CPU.

---

## 1. Anatomy of one step

| Tensor | Shape | Role |
|---|---|---|
| `tokens` | `[4, 32]` | \(x \in \{1,\ldots,V-1\}^{B \times L}\) |
| `tok_emb.weight` | `[65, 64]` | lookup \(V \times D\) |
| `pos_emb.weight` | `[32, 64]` | absolute positions |
| `hidden` | `[4, 32, 64]` | post-LN residual stream |
| `logits` | `[4, 32, 65]` | unnormalized class scores |
| `loss` | `[]` | mean CE over \(B(L-1)\) pairs |
| `lm_head.weight` | `[65, 64]` | output projection |
| `lm_head.weight.grad` | `[65, 64]` | \(\partial\mathcal{L}/\partial W\) |
| `blocks[0].attn.w_q.weight` | `[64, 64]` | first-layer \(Q\) map |

The last logit vector is computed and dropped by the shift. A causal LM has no target to the right of the final token.

---

## 2. A scalar derivative that does not use autograd

Probe \(w = W^{\mathrm{LM}}_{3,7}\) in `float64` (fp32 CE is too flat for one entry of a \(65 \times 64\) head):

\[
\widehat{\partial_w \mathcal{L}}
= \frac{\mathcal{L}(w+\varepsilon)-\mathcal{L}(w-\varepsilon)}{2\varepsilon},
\qquad \varepsilon = 10^{-6}.
\]

| Estimator | Value |
|---|---|
| \(L(w+\varepsilon)\) | 4.320576905498 |
| \(L(w-\varepsilon)\) | 4.320576907117 |
| central difference | \(-8.0922157863 \times 10^{-4}\) |
| `loss.backward()` | \(-8.0922178441 \times 10^{-4}\) |
| \(\lvert\Delta\rvert\) | \(2.06 \times 10^{-10}\) |
| relative error | \(2.54 \times 10^{-7}\) |

Eight-digit agreement. The graph from `lm_head` back through the residual stream is the graph of this \(\mathcal{L}\), not of a shifted or detached cousin.

---

## 3. Accumulation as a change of measure

A micro-batch of length \(\ell\) contributes \(n = B(\ell-1)\) terms to the corpus average. Lengths \(\{8,16,32\}\) therefore have token masses in the ratio \(7:15:31\).

**Broken** (average of averages): each micro-batch gets weight \(1/M\). The shortest sequence is over-represented by \(31/7 \approx 4.4\times\).

**Correct** (token measure):

\[
\bar{\mathcal{L}}
= \frac{\sum_m n_m \,\operatorname{mean}(\mathcal{L}_m)}{\sum_m n_m}.
\]

Implementation: `(mean_m * n_m).backward()` per micro-batch, then divide accumulated `.grad` by \(\sum n_m\) before `opt.step()`.

Same init, same AdamW, 40 steps:

| Estimator | Terminal loss |
|---|---|
| equal vote per micro-batch | 4.2794 |
| token-weighted | 4.2403 |
| gap | 0.0391 |

`accum_gap.png` is the evidence. Any trainer that runs `F.cross_entropy(..., reduction="mean")` once per uneven micro-batch and then `opt.step()` is using the broken estimator.

---

## 4. When the vector moves before the scalar

\[
\lVert g \rVert_2
= \Big(\sum_p \lVert \nabla_p \mathcal{L} \rVert_F^2\Big)^{1/2}.
\]

Logged for 80 AdamW updates. At step 29:

| | \(t=28\) | \(t=29\) | \(\Delta\) |
|---|---|---|---|
| \(\mathcal{L}\) | 4.1712 | 4.1800 | \(+8.8 \times 10^{-3}\) |
| \(\lVert g \rVert_2\) | 0.4464 | 0.5507 | \(+0.104\) |

Parameter-space velocity changed by ~23% while the objective moved by ~0.2%. Loss is a 1-D projection of a \(10^5\)-dimensional update. Clip / warmup / “is this run alive?” belong on \(\lVert g \rVert_2\).

`gradnorm_vs_loss.png`.

---

## 5. MFU without a borrowed datasheet

Matmul-dominated accounting:

\[
\mathrm{FLOPs/step} \approx 6NT,
\qquad T = B(L-1) = 248,
\qquad N = 109{,}312.
\]

| Quantity | Value |
|---|---|
| FLOPs / step | \(1.627 \times 10^8\) |
| steps / s | 118.32 |
| achieved FLOP/s | \(1.925 \times 10^{10}\) |
| peak used | \(5 \times 10^{11}\) (CPU/MPS ballpark) |
| **MFU** | **3.85%** |

Using an H100 peak as the denominator would be a category error.

Distance to 40%:

- \(D=64\) GEMMs are memory-bound
- eager Python: LayerNorm / GELU / CE are separate launches
- no FlashAttention, no fused CE, no `torch.compile`
- no tensor cores on this device
- 40% MFU is large-\(N\), large-\(T\), packed-bf16

A T4 rerun changes the denominator, not the diagnosis.

---

## 6. `0.1` as three different numbers

Sign / exponent / trailing significand. fp32 and bf16 from the IEEE word. E4M3: nearest representable OCP value (bias 7).

**fp32** — `1 + 8 + 23`  
`0 01111011 10011001100110011001101`  
stored \(0.1000000015\) · abs error \(1.49 \times 10^{-9}\)

**bf16** — `1 + 8 + 7`  
exponent field identical to fp32.  
`0 01111011 1001100`  
stored \(0.1000976562\) · abs error \(9.77 \times 10^{-5}\)

**fp8 E4M3** — `1 + 4 + 3`  
`0 0011 101`  
unbiased exp \(= 3-7 = -4\), significand \(1 + 5/8 = 1.625\)  
\(1.625 \times 2^{-4} = 0.1015625\) · relative error **1.6%**

### Training dtype: **bf16**

fp32 doubles traffic for digits Adam will not use. E4M3 already mis-states `0.1` by more than a percent and will flush \(10^{-4}\) updates without a scale tensor — which this loop does not implement. bf16 keeps the fp32 exponent (small grads still exist) and halves the width. That is the default on a real GPU.

---

## Failure modes

| Lie | How it shows up if you only watch `loss.item()` |
|---|---|
| Detached head / wrong shift | Finite difference and `backward()` disagree |
| `mean` per uneven micro-batch | Two accumulation recipes, two curves |
| “training stalled” | \(\lVert g \rVert_2\) still walking |
| “we got 40% MFU” | \(6NT\) against an honest peak |
| “fp8 is free accuracy” | `0.1` is already a different number |

---

## Reproduce

```bash
python session10_gradients.py
```

Writes `accum_gap.png` and `gradnorm_vs_loss.png`. The notebook is the same source in one cell.

---

*A monotonic loss is the weakest statement a trainer can make.*
