# Character-Based Fourier/Phasor Transformer

A complete, dependency-light demonstration of how to make a transformer's
continuous output vectors usable for next-word prediction, when words are
represented as **holographic phasor embeddings** (a Fourier-domain
Vector Symbolic Architecture) instead of a learned embedding table.

Built entirely on top of NumPy — no PyTorch, no TensorFlow, no autograd.
Every gradient in the attention block is hand-derived and backpropagated
manually.

---

## Table of contents

- [Problem statement](#problem-statement)
- [Solution overview](#solution-overview)
- [How it works](#how-it-works)
  - [1. Encode — characters become waves](#1-encode--characters-become-waves)
  - [2. Predict — a real self-attention transformer](#2-predict--a-real-self-attention-transformer)
  - [3. Decode — cleanup memory](#3-decode--cleanup-memory)
- [Requirements](#requirements)
- [Usage](#usage)
- [Interactive mode](#interactive-mode)
- [Configuration](#configuration)
- [Adding new vocabulary](#adding-new-vocabulary)
- [Known limitations](#known-limitations)
- [Roadmap](#roadmap)

---

## Problem statement

Kronecker-style tensor binding gives a way to turn a word into a fixed-size
vector, but it doesn't answer a critical question: once a transformer
predicts the *next* word's vector, how do you turn that raw, continuous
prediction back into an actual word? Without a vocabulary lookup step,
a transformer trained on these embeddings can encode input perfectly and
still have no way to produce usable output.

## Solution overview

This script closes that loop end-to-end:

```
 characters  --encode-->  word vectors  --predict-->  next-word vector  --decode-->  word
(phasor waves)          (self-attention)             (continuous output)      (nearest-neighbor
                                                                                 cleanup memory)
```

Run the script and it will:

1. Build a character-level phasor codebook and use it to encode a small
   vocabulary of words.
2. Train a single-head, causal self-attention transformer from scratch on a
   toy corpus to predict each next word's vector.
3. Decode the transformer's raw output back into real words using
   similarity search against the vocabulary.
4. Drop into an interactive prompt where you can type any word (or short
   phrase) from the vocabulary and see what the model predicts next.

## How it works

### 1. Encode — characters become waves

Every letter `a`–`z` is assigned a random **phasor**: a vector of unit-
magnitude complex numbers, `exp(iθ)`, i.e. a set of waves with random
phase. A word is built by *binding* each character to a position phasor
(via elementwise complex multiplication, which adds phase) and summing the
results:

```python
word_vector = Σ_k  bind(char_k, position_k)
```

This is a **Holographic Reduced Representation (HRR)** in its Fourier
form (FHRR) — the practical, fixed-dimensionality alternative to Kronecker
tensor products. Order is preserved (`"cat"` and `"act"` encode to
different vectors), and the encoding is entirely deterministic — no
training required for this step.

Since the transformer that follows operates on ordinary real-valued
vectors, each complex `D`-dimensional word vector is converted to a real
`2D`-dimensional vector by concatenating its real and imaginary parts.

### 2. Predict — a real self-attention transformer

A genuine (if intentionally minimal) transformer decoder block:

- Learned query/key/value projections
- Scaled dot-product attention with causal masking (no looking ahead)
- Residual connections around both the attention and feedforward sub-blocks
- A two-layer feedforward network with ReLU
- Sinusoidal positional encoding, as in the original Transformer paper

The **forward pass** (`forward()`) and **backward pass** (`backward()`)
are both implemented by hand — every gradient is derived analytically and
applied with plain gradient descent plus gradient clipping (needed because
this minimal implementation has no layer normalization).

The model is trained to minimize mean-squared error between its predicted
vector and the true next word's vector — a regression objective, not
classification, since there is no discrete softmax layer over a
vocabulary.

### 3. Decode — cleanup memory

The transformer's raw output is just a vector of floating-point numbers.
To turn it into a word, `decode_to_word()` compares it against every
vector in `vocab_codebook_real` (cosine similarity) and returns the
closest match:

```python
predicted_word = argmax_w  similarity(y_hat, vocab_codebook_real[w])
```

This is the same role a learned "unembedding" matrix plays in a
conventional transformer — except here the comparison vectors come
directly from the deterministic phasor encoding rather than being learned.

## Requirements

- Python 3.8+
- NumPy

```bash
pip install numpy
```

No other dependencies.

## Usage

```bash
python char_ft_transformer.py
```

Running the script will, in order:

1. Print the character and vocabulary codebook sizes.
2. Train the transformer for 400 epochs, printing loss every 80 epochs.
3. Generate a sample continuation from the prompt
   `["the", "quick", "brown"]`, showing the top-3 decode candidates at
   each step.
4. Enter **interactive mode** (see below).

## Interactive mode

After training, the script prompts you for input:

```
your word(s) > fox
  after ['fox'] -> predicted next word: 'jumps'
  top-5 candidates: ['jumps (0.97)', 'the (0.27)', 'ran (0.06)', 'over (0.06)', 'fast (0.03)']

your word(s) > the cat
  after ['the', 'cat'] -> predicted next word: 'sat'
  top-5 candidates: ['sat (0.83)', 'mat (0.55)', 'cat (0.50)', 'fox (0.46)', 'over (0.32)']

your word(s) > quit
```

- Enter one or more space-separated words from the vocabulary printed at
  the top of this section.
- The model predicts whatever word comes *after* what you typed.
- The similarity score next to each candidate indicates confidence — a
  large gap between the top candidate and the runner-up means the model is
  confident; a narrow gap means the prediction is closer to a coin flip.
- Type `quit`, `exit`, or press Enter on an empty line to exit.

> **Note:** if you're piping input into the script or running it in an
> environment with no attached input stream, the prompt will detect this
> (`EOFError`) and exit gracefully rather than hang.

## Configuration

Key parameters near the top of the script:

| Variable | Meaning | Default |
|---|---|---|
| `D` | Phasor dimension per character/word (complex) | `32` |
| `d_model` | Real-valued model dimension (`2 * D`) | `64` |
| `d_k` | Attention query/key/value dimension | `24` |
| `d_ff` | Feedforward hidden layer size | `64` |
| `lr` | Learning rate | `0.005` |
| `EPOCHS` | Training epochs | `400` |

If you increase `lr` without also relying on the built-in gradient
clipping, training can diverge (loss → `NaN`) — this architecture has no
layer normalization to keep it stable at higher learning rates.

## Adding new vocabulary

Every word that appears anywhere — in `corpus` sentences or as something
you want the model to predict or accept as interactive input — **must**
be listed in `vocab`:

```python
vocab = ["the", "quick", "brown", "fox", "jumps", "over",
         "lazy", "dog", "cat", "sat", "on", "mat", "ran", "fast"]
```

Add new sentences to `corpus` using only words present in `vocab`,
otherwise you'll hit a `KeyError` when the script tries to look up a
vector that was never encoded:

```python
corpus = [
    ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
    # add more sentences here
]
```

For reliable predictions, make sure any new word appears in enough
training sentences to establish a consistent pattern — a word seen only
once will have a weak, unreliable decode margin.

## Known limitations

- **Toy scale.** One attention head, one layer, a 14-word vocabulary, and
  five training sentences. This is built to demonstrate the encode →
  predict → decode mechanism clearly, not to serve as a production
  language model.
- **No layer normalization.** Omitted to keep the from-scratch backward
  pass tractable; compensated for with gradient clipping, but this limits
  how aggressively the model can be trained.
- **Spelling-based similarity.** Because word vectors are derived from
  character overlap, spelling-similar words (e.g. `"cat"` and `"cot"`)
  will be closer in vector space than semantically similar ones, unlike
  learned embeddings such as word2vec or GloVe.
- **Exact-vocabulary decoding only.** The interactive prompt only accepts
  words already present in `vocab_codebook_real`; there is no fallback for
  out-of-vocabulary input.

## Roadmap

A follow-up version of this project replaces character-derived phasor
vectors with real word embeddings (e.g. word2vec/GloVe-style) as the
atomic unit, using circular convolution (classic HRR) instead of complex
multiplication for binding, since real embeddings are not unit-magnitude
phase vectors. This introduces a new challenge — cleanup-memory decoding
becomes harder to keep accurate once vocabulary vectors are no longer
quasi-orthogonal by construction, since semantically similar words sit
close together on purpose.
