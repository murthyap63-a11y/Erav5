"""
COMPLETE CHARACTER-BASED FOURIER/PHASOR SOLUTION
=================================================
End to end, no external ML libraries -- just numpy.

  1. ENCODE   : characters -> waves -> words        (bind + sum, as before)
  2. PREDICT  : a real single-head self-attention transformer, trained with
                plain gradient descent, consumes a sequence of word-vectors
                and predicts the NEXT word's vector (continuous output)
  3. DECODE   : cleanup memory -- nearest neighbor against vocab_codebook
                turns that continuous prediction back into an actual word

This is the missing piece from the original problem statement, now
wired together into one working pipeline.
"""

import numpy as np
import sys

rng = np.random.default_rng(0)

# ======================================================================
# 1. ENCODE: characters -> waves -> words
# ======================================================================
D = 32  # phasor dimension per character/word

def random_phasor(d):
    theta = rng.uniform(-np.pi, np.pi, size=d)
    return np.exp(1j * theta)

def bind(a, b):
    return a * b

alphabet = list("abcdefghijklmnopqrstuvwxyz")
char_codebook = {c: random_phasor(D) for c in alphabet}
char_pos_codebook = {i: random_phasor(D) for i in range(12)}

def encode_word(word):
    vec = np.zeros(D, dtype=complex)
    for k, ch in enumerate(word):
        vec += bind(char_codebook[ch], char_pos_codebook[k])
    return vec

def complex_to_real(v):
    """A complex D-vector -> a real 2D-vector (concat real & imag parts),
    since the transformer below operates on ordinary real-valued vectors."""
    return np.concatenate([v.real, v.imag])

vocab = ["the", "quick", "brown", "fox", "jumps", "over",
         "lazy", "dog", "cat", "sat", "on", "mat", "ran", "fast"]
vocab_codebook = {w: encode_word(w) for w in vocab}          # complex, D-dim
vocab_codebook_real = {w: complex_to_real(v) for w, v in vocab_codebook.items()}  # real, 2D-dim

d_model = 2 * D  # = 64
print(f"Character codebook: {len(alphabet)} letters, each a {D}-dim wave")
print(f"Vocabulary codebook: {len(vocab)} words, each a {d_model}-dim real vector "
      f"(derived from spelling)\n")

# ======================================================================
# 2. PREDICT: a small real self-attention transformer, trained from scratch
# ======================================================================
d_k = 24     # attention key/query/value dim
d_ff = 64    # feedforward hidden dim
lr = 0.005

def init(*shape):
    return rng.normal(0, 1 / np.sqrt(shape[0]), size=shape)

Wq, Wk, Wv = init(d_model, d_k), init(d_model, d_k), init(d_model, d_k)
Wo = init(d_k, d_model)
W1, b1 = init(d_model, d_ff), np.zeros(d_ff)
W2, b2 = init(d_ff, d_model), np.zeros(d_model)
Wout, bout = init(d_model, d_model), np.zeros(d_model)

def sinusoidal_pe(T, dim):
    pe = np.zeros((T, dim))
    pos = np.arange(T)[:, None]
    i = np.arange(dim)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / dim)
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def forward(X):
    """X: (T, d_model) token vectors (already includes positional encoding)."""
    T = X.shape[0]
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    scores = (Q @ K.T) / np.sqrt(d_k)
    mask = np.triu(np.ones((T, T)), k=1).astype(bool)  # causal: no peeking ahead
    scores = np.where(mask, -1e9, scores)
    A = softmax(scores, axis=-1)
    O = A @ V
    Attn_out = O @ Wo
    H1 = X + Attn_out                          # residual
    F_pre = H1 @ W1 + b1
    F_relu = np.maximum(F_pre, 0)
    F_out = F_relu @ W2 + b2
    H2 = H1 + F_out                            # residual
    pred = H2 @ Wout + bout                    # predicted NEXT-token vector, per position
    cache = (X, Q, K, V, scores, A, O, Attn_out, H1, F_pre, F_relu, F_out, H2, mask)
    return pred, cache

def backward(pred, targets, valid_mask, cache):
    X, Q, K, V, scores, A, O, Attn_out, H1, F_pre, F_relu, F_out, H2, mask = cache
    T = X.shape[0]
    n_valid = valid_mask.sum()

    dpred = np.where(valid_mask[:, None], 2 * (pred - targets) / n_valid, 0.0)
    dWout = H2.T @ dpred
    dbout = dpred.sum(axis=0)
    dH2 = dpred @ Wout.T

    dH1 = dH2.copy()
    dF_out = dH2.copy()
    dW2 = F_relu.T @ dF_out
    db2 = dF_out.sum(axis=0)
    dF_relu = dF_out @ W2.T
    dF_pre = dF_relu * (F_pre > 0)
    dW1 = H1.T @ dF_pre
    db1 = dF_pre.sum(axis=0)
    dH1 += dF_pre @ W1.T

    dX = dH1.copy()
    dAttn_out = dH1.copy()
    dO = dAttn_out @ Wo.T
    dWo = O.T @ dAttn_out

    dA = dO @ V.T
    dV = A.T @ dO

    dscores = A * (dA - (dA * A).sum(axis=-1, keepdims=True))
    dscores = np.where(mask, 0.0, dscores) / np.sqrt(d_k)

    dQ = dscores @ K
    dK = dscores.T @ Q

    dWq = X.T @ dQ
    dWk = X.T @ dK
    dWv = X.T @ dV
    dX += dQ @ Wq.T + dK @ Wk.T + dV @ Wv.T

    grads = dict(Wq=dWq, Wk=dWk, Wv=dWv, Wo=dWo, W1=dW1, b1=db1,
                 W2=dW2, b2=db2, Wout=dWout, bout=dbout)
    return grads

params = dict(Wq=Wq, Wk=Wk, Wv=Wv, Wo=Wo, W1=W1, b1=b1, W2=W2, b2=b2, Wout=Wout, bout=bout)

def apply_grads(grads):
    for name in params:
        g = grads[name]
        norm = np.linalg.norm(g)
        if norm > 1.0:               # gradient clipping to prevent blow-ups
            g = g * (1.0 / norm)
        params[name] -= lr * g
    globals().update(params)

# ----------------------------------------------------------------------
# toy training corpus (character-encoded words strung into sentences)
# ----------------------------------------------------------------------
corpus = [
    ["the", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"],
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "fast"],
    ["the", "fox", "ran", "over", "the", "dog"],
    ["quick", "fox", "jumps", "over", "lazy", "dog"],
]

print("=" * 70)
print("Training a real single-head self-attention transformer from scratch")
print("(predicting the NEXT word's phasor-derived vector, continuous output)")
print("=" * 70)

EPOCHS = 400
for epoch in range(EPOCHS):
    total_loss = 0.0
    for sentence in corpus:
        T = len(sentence)
        X = np.stack([vocab_codebook_real[w] for w in sentence]) + sinusoidal_pe(T, d_model)
        targets = np.zeros((T, d_model))
        valid_mask = np.zeros(T, dtype=bool)
        for t in range(T - 1):
            targets[t] = vocab_codebook_real[sentence[t + 1]]
            valid_mask[t] = True

        pred, cache = forward(X)
        loss = np.mean(((pred - targets) ** 2)[valid_mask])
        total_loss += loss

        grads = backward(pred, targets, valid_mask, cache)
        apply_grads(grads)

    if epoch % 80 == 0 or epoch == EPOCHS - 1:
        print(f"  epoch {epoch:4d}   avg MSE loss = {total_loss / len(corpus):.4f}")

# ======================================================================
# 3. DECODE: cleanup memory -- turn the continuous prediction back into a word
# ======================================================================
def decode_to_word(y_hat_real, codebook_real, top_k=3):
    scores = {}
    for w, v in codebook_real.items():
        sim = np.dot(v, y_hat_real) / (np.linalg.norm(v) * np.linalg.norm(y_hat_real) + 1e-12)
        scores[w] = sim
    return sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]

print("\n" + "=" * 70)
print("Feeding a prompt through the trained transformer, decoding each step")
print("=" * 70)

def generate(prompt_words, n_new=4):
    words = list(prompt_words)
    for _ in range(n_new):
        T = len(words)
        X = np.stack([vocab_codebook_real[w] for w in words]) + sinusoidal_pe(T, d_model)
        pred, _ = forward(X)
        y_hat = pred[-1]                     # prediction made at the LAST position
        top3 = decode_to_word(y_hat, vocab_codebook_real, top_k=3)
        next_word = top3[0][0]
        print(f"  after {words} -> predicted vector decodes to: "
              f"'{next_word}'  (top-3: {[f'{w}:{s:.2f}' for w, s in top3]})")
        words.append(next_word)
    return words

final = generate(["the", "quick", "brown"], n_new=4)
print(f"\n  full generated sequence: {final}")

# ======================================================================
# INTERACTIVE: type any word from the vocabulary, see the prediction
# ======================================================================
def predict_next_word(words):
    T = len(words)
    X = np.stack([vocab_codebook_real[w] for w in words]) + sinusoidal_pe(T, d_model)
    pred, _ = forward(X)
    y_hat = pred[-1]
    return decode_to_word(y_hat, vocab_codebook_real, top_k=5)

print("\n" + "=" * 70)
print("INTERACTIVE MODE")
print("=" * 70)
print(f"Vocabulary: {vocab}")
print("Type one or more words separated by spaces (must be in the vocabulary above).")
print("Type 'quit' to exit.\n")
sys.stdout.flush()

while True:
    try:
        sys.stdout.write("your word(s) > ")
        sys.stdout.flush()
        user_input = input().strip().lower()
    except EOFError:
        print("\n(no input stream detected -- see note below)")
        break
    if user_input in ("quit", "exit", ""):
        break
    words = user_input.split()
    unknown = [w for w in words if w not in vocab_codebook_real]
    if unknown:
        print(f"  '{unknown[0]}' is not in the vocabulary. Try one of: {vocab}\n")
        sys.stdout.flush()
        continue
    top5 = predict_next_word(words)
    print(f"  after {words} -> predicted next word: '{top5[0][0]}'")
    print(f"  top-5 candidates: {[f'{w} ({s:.2f})' for w, s in top5]}\n")
    sys.stdout.flush()
