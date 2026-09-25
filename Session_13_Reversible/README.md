hi
# Reversible Transformer Benchmarking & Memory Optimization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository provides a custom PyTorch implementation and performance benchmark comparing standard causal **Transformer Language Models (Standard Transformer)** against ODE-inspired **Reversible Transformers** (Euler and Midpoint numerical integration formulations). 

Reversible architectures trade backpropagation compute for memory efficiency by dropping intermediate activation caching during the forward pass. Instead, activations are reconstructed on-the-fly during the backward pass using custom `torch.autograd.Function` vector-Jacobian products (VJP).

---

## Key Features

* **Custom Autograd Engine:** Hand-crafted PyTorch autograd primitives for exact gradient reconstruction without standard autograd activation storage.
* **Multiple Numerical Discretizations:**
  * **Standard Transformer:** Non-reversible baseline with standard residual connections.
  * **Euler Reversible Transformer:** First-order additive coupling integration scheme.
  * **Midpoint Reversible Transformer:** Second-order central difference integration scheme.
* **Comprehensive Benchmarking Suite:** Measures peak VRAM (MB), training throughput (tokens/sec), parameter footprint, and loss convergence dynamics.

---

## Mathematical Formulations

### 1. Standard Transformer Layer
$$x_{l+1} = x_l + \text{MLP}(\text{LN}(x_l + \text{Attn}(\text{LN}(x_l))))$$

### 2. Euler Reversible Layer
The hidden state vector $x \in \mathbb{R}^{d_{\text{model}}}$ is split into two equal channels $x = [x_1, x_2]$:
$$\begin{aligned} y_1 &= x_1 + F(x_2) \\ y_2 &= x_2 + G(y_1) \end{aligned}$$

**Inversion (Backward Pass):**
$$\begin{aligned} x_2 &= y_2 - G(y_1) \\ x_1 &= y_1 - F(x_2) \end{aligned}$$

### 3. Midpoint Reversible Layer
Uses central difference discretization to achieve second-order integration accuracy:
$$p_{l+1} = p_{l-1} + 2h \cdot f_\theta(p_l)$$

**Inversion (Backward Pass):**
$$p_{l-1} = p_{l+1} - 2h \cdot f_\theta(p_l)$$

---

## Architectural Comparison
