# 120B Blueprint — Frontier-Class Model

**Architecture & training design · engineering blueprint**

> Recommendations are grounded in publicly documented practice (Llama 3, DeepSeek-V3, Qwen3, Gemma) — not insider knowledge of any specific closed lab's recipe.

A full design pass: architecture, data mixture, cleaning at scale, tokenizer sizing, a five-stage training curriculum, the compute-budget math for the requested mixture, agentic planning & failure-recovery design, controllable reasoning depth, mixture governance (protected floors + anneal reserve), a small-scale proxy-run validation plan, and the hardware to actually run it.

| | |
|---|---|
| **Total / active params (MoE)** | 120B / ~20B |
| **Pretrain tokens** | 30T |
| **Indic tiers** | 4 — verified / unverified / translated / synthetic |
| **Reasoning-depth bands** | 4, floor-protected |
| **Proxy validation** | 1B → 3B scale, before full scale |
| **Full run** | ~12 days on 10K H100-class GPUs |

## Table of contents

1. [Architecture](#1--architecture)
2. [Pretraining data](#2--pretraining-data)
3. [Cleaning pipeline, at scale](#3--cleaning-pipeline-at-scale)
4. [Tokenizer](#4--tokenizer)
5. [Training curriculum](#5--training-curriculum)
6. [Compute budget for the mixture](#6--compute-budget-for-the-mixture)
7. [Agentic depth, planning & failure recovery](#7--agentic-depth-planning--failure-recovery)
8. [Controllable reasoning depth](#8--controllable-reasoning-depth)
9. [Proxy-run validation](#9--proxy-run-validation)
10. [Hardware requirements](#10--hardware-requirements)
11. [Benchmark alignment](#11--benchmark-alignment)

---

## 1 · Architecture

### Mixture-of-Experts, not dense — the single biggest lever at this scale

At 120B "total size," the first real fork in the road is dense vs. sparse (MoE). Given the mixture spans code, agentic/tool-use, reasoning, long-context, and multilingual — a genuinely broad capability target — MoE is the right call: it lets you train on far more tokens for the same compute, which matters more for breadth of capability than raw parameter count does.

| Component | Recommendation | Why |
|---|---|---|
| Overall shape | MoE, 120B total / ~20B active per token (6× sparsity) | ~6× cheaper training FLOPs per token than a 120B dense model at equal quality-per-active-param; moderate sparsity (vs. DeepSeek-V3's ~18×) trades some efficiency for a smaller quality gap vs. dense |
| Experts | 128 routed experts, top-8 active + 2 always-on shared experts | Shared experts (DeepSeek-V3 pattern) capture common cross-domain features so routed experts can specialize more cleanly |
| Load balancing | Auxiliary-loss-free balancing via a learned per-expert bias term, not a large aux-loss coefficient | Avoids the classic MoE failure mode where a heavy balancing loss actively fights the main training objective |
| Attention | Grouped-Query Attention, 48 query heads / 8 KV heads, head dim 128 | Cuts KV-cache size ~6× vs. full multi-head attention — critical for the long-context targets in the mixture |
| Positional encoding | RoPE, base θ raised during the dedicated long-context stage (Sec. 5), YaRN-style scaling for inference-time extension beyond the trained length | Trains stably at a moderate context length, then extends further at inference without full retraining |
| Normalization | RMSNorm, pre-norm, QK-norm on attention logits | QK-norm specifically prevents attention-logit blowups, a common source of loss spikes at this scale |
| Activation | SwiGLU | Standard choice across current open frontier-class models; consistently outperforms ReLU/GeLU variants at scale |
| Depth × width | ~80 layers, hidden size 6,144 | Deeper-and-narrower generally reasons better per-FLOP than shallow-and-wide at this parameter class |
| Auxiliary objective | Multi-token prediction (predict token t+2 as well as t+1) via a lightweight extra head | Meaningfully improves sample efficiency and downstream coding/reasoning quality per token trained (DeepSeek-V3 finding), can be dropped at inference |
| Context length | Trained natively to 128K, extended to ~1M via YaRN + a dedicated long-context stage | Matches how current frontier-class models handle long-context — native training up to a point, extension technique beyond it |
| Vocabulary | ~230,000 tokens (see [Section 4](#4--tokenizer)) | Sized for the actual mixture — code, math, agentic syntax, and multiple Indic scripts all need dedicated budget, not a shared undersized vocab |

> **MoE's real cost isn't compute, it's communication.** Expert parallelism requires all-to-all GPU communication on every forward/backward pass. This is the architecture decision with the biggest hardware/networking implication — see [Section 10](#10--hardware-requirements).

> **Architecture choices get checked, not just asserted** — the active-parameter/sparsity ratio and context-length target above are hypotheses tested at 1B/3B proxy scale ([Section 9](#9--proxy-run-validation)) against MMLU-Pro (general capacity) and RULER (long-context) before being locked in for the full run, not fixed by this table alone.

---

## 2 · Pretraining data

### Sourcing each of the seven mixture components

Every bucket needs a different sourcing strategy — web-scale crawling doesn't work for verified reasoning traces, and verified-execution filtering doesn't apply to general web text. The benchmarks column names the evaluation suites used by current popular models (Claude, GPT, Gemini) for that capability — decontamination ([Section 3](#3--cleaning-pipeline-at-scale)) runs against all of them.

| Bucket | Primary sources | Sourcing method | Benchmarks it feeds |
|---|---|---|---|
| General web | Filtered CommonCrawl (FineWeb/Dolma-style pipelines), curated encyclopedic + reference sources | Broad crawl + quality/toxicity/dedup filtering (Section 3) | MMLU-Pro, HellaSwag, ARC |
| Code | Permissively-licensed GitHub (The Stack v2-style), PRs/issues/commit history, notebooks | License filtering, near-dedup across forks, syntax-validity check | HumanEval, MBPP, SWE-bench Verified, LiveCodeBench |
| STEM / math | arXiv, OpenWebMath, textbooks, olympiad archives, curated problem sets | Domain-specific crawls + LaTeX-preserving cleaning (Step 9) | GSM8K, MATH, AIME, GPQA-Diamond |
| Reasoning traces | Long chain-of-thought solutions, worked textbook proofs, distilled traces from a stronger verifier/teacher model, self-generated-and-verified traces — sourced and banded per difficulty (Section 8) | **Verified, not scraped** — kept only if the final answer checks out against a ground truth or executes correctly | GPQA-Diamond, ARC-AGI, BIG-Bench-Hard, MuSR |
| Agentic & tool-use | API docs, CLI transcripts, synthetic ReAct-style trajectories generated against real sandboxed tools, **plus explicit task-planning/decomposition traces and deliberately-induced-failure recovery traces** (Section 7) | Kept if the *overall* task completes in a sandbox — including trajectories with a failed intermediate step that recovers, not success-only trajectories | BFCL, AgentBench, GAIA, SWE-bench Verified, WebArena |
| Long-context | Full books, long code repositories (whole-repo context), multi-document concatenations, **and long agent sessions with growing tool-call/task history** — the same joint capability as the Agentic row above, viewed through a length lens (Section 7) | Selected by document/session length from the other buckets, not an independent content source — see Section 5 | RULER, LongBench v2, InfiniteBench, Needle-in-a-Haystack |
| Indic languages | Split across four tiers — verified / unverified / translated / synthetic — see the dedicated breakdown in Section 6, not a single headline source list | Per-tier sourcing method differs; see Section 6 | IndicXTREME, MILU, Aya evaluation suite, FLORES-200 |

---

## 3 · Cleaning pipeline, at scale

### Same 10-step pipeline, now distributed

The 10-step pipeline (normalization → language ID → dedup → quality/Gopher/C4 → toxicity → PII → decontamination → code secrets → LaTeX-aware → human audit) built earlier applies directly here. At 30T-token scale, several things change:

1. **Distributed execution is mandatory.** A single-machine multiprocessing pool (the earlier `--workers` approach) doesn't scale to petabytes. Run the same per-line logic on a distributed framework (Ray or Spark) across a CPU cluster, with the corpus sharded so dedup and language-ID stay correct across shard boundaries.

2. **MinHash/LSH needs a distributed index.** A single-process LSH index (as built earlier) can't hold a 30T-token corpus's shingle sets in memory. Use a distributed LSH implementation (or bucket-then-dedup: hash-partition by MinHash band, dedup within each partition, merge) to keep the corpus-wide near-dup guarantee at scale.

3. **Decontamination against a real benchmark suite.** Step 7, left empty in earlier test runs, needs to actually run here against the full evaluation suite you'll be measured on: MMLU-Pro, GPQA, HumanEval/SWE-bench, MATH/AIME, BFCL, AgentBench/GAIA, RULER, IndicXTREME/MILU. Skipping this is the single most common way a benchmark score turns out to be memorization, not capability.

4. **Cleaning throughput is redirected, not fixed upfront.** Cleaning continues toward the cumulative 30T-token target on an ongoing basis, but **where** that effort goes isn't decided once at the start. Each proxy run (Section 9) and each check against the protected floors (Section 6) can reveal a slot running short of its target — most likely Indic-verified, reasoning-traces, or agentic, since those are the hardest to source at volume. When that happens, cleaning/acquisition effort gets reallocated toward that specific starved slot rather than continuing to clean general web at the same rate.

---

## 4 · Tokenizer

### ~230,000 tokens — sized for this specific mixture, not a generic default

This follows the same fertility-driven sizing method as the earlier tokenizer design: keep adding vocabulary to a domain until its fertility (tokens/word) stops improving, rather than allocating proportionally to raw corpus share.

| Domain | Target fertility | Why it needs dedicated budget |
|---|---|---|
| English / general web | 1.20× | Baseline — largest single content source |
| Code (15+ languages) | 1.15× | Keywords and common identifiers as single tokens directly improve completion quality and reduce sequence length in repo-scale context |
| Math / scientific notation | 1.15× | Single-digit tokenization is specifically important for arithmetic reliability; LaTeX commands as whole tokens |
| Agentic / structural tokens | n/a (structural) | JSON keys, function-call schema, tool-call delimiters as single tokens — reduces the token overhead of every tool call, which compounds heavily in long agentic sessions |
| Reasoning-trace formatting | 1.20× | Long chain-of-thought text is mostly natural language, but step-marker/notation tokens benefit from dedicated merges given how much of the token budget reasoning traces consume at inference |
| Indic languages (pooled, ~18 languages) | 1.35–1.60× | Same rationale as the dedicated India-first design — see the earlier tokenizer report for the full per-language breakdown |

> **Verified by an earlier run report** — a generic, undersized, single-script-biased vocabulary produced 14.87 tokens/word on real Indic text against a ~1.4× target. That failure mode is exactly what a properly-sized, domain-aware vocabulary here is meant to prevent.

---

## 5 · Training curriculum

### Five stages, one learning-rate schedule (Warmup–Stable–Decay)

A single flat pretraining run wastes the fact that not all data is equally valuable at every point in training. A staged curriculum lets you control WHEN the model sees what, on top of a WSD (warmup–stable–decay) learning-rate schedule rather than a fixed cosine curve — WSD lets you extend the "stable" phase without redesigning the whole schedule if you decide to train longer.

```
Seed 2% ──┬── General — 78% ──┬── Reasoning 10% ──┬── Long-ctx 6% ──┬── Anneal 4%
```

| Stage | Share | Tokens | Purpose & LR behavior | Validated via |
|---|---|---|---|---|
| **Seed** | 2% | 0.6T | Small-scale warm-up on the cleanest, most-deduplicated slice of the mixture — verifies the data pipeline and training infra are actually correct before committing the full compute budget. LR ramps 0 → peak. | Loss-curve sanity only — no formal benchmark yet |
| **General** | 78% | 23.4T | Bulk of training on the full mixture (Section 6's proportions) at the stable, constant peak LR. Builds broad world knowledge, language, and code capability. | MMLU-Pro trend, checkpoint over checkpoint |
| **Reasoning** | 10% | 3.0T | Continued training with STEM/math and reasoning-traces oversampled well above their Section 6 share. Still stable LR. Builds latent multi-step reasoning capability before any RL post-training touches it. | GSM8K / MATH / GPQA-Diamond trend |
| **Long-context** | 6% | 1.8T | RoPE base θ increased; trains on long documents, whole repos, and long agent sessions drawn from the other buckets. Progressive length curriculum (e.g. 32K→64K→128K) rather than jumping straight to the target length. | RULER / Needle-in-a-Haystack at each length rung |
| **Anneal** | 4% | 1.2T | LR decays to ~0 (the "D" in WSD) on the highest-quality, most-curated blend across all buckets. **This 1.2T is a protected reserve, fenced off before the run begins** — it is not sourced by whatever happens to be left over after other stages consume their share, and the earlier stages' selectors cannot draw it down. Llama 3 and MiniCPM both report a disproportionate share of final capability crystallizing here. | Full Section 11 suite — the final gate before declaring pretraining complete |

> **Not double-counted** — the stage table above and the domain-mixture table in Section 6 are two different cuts of the same 30T tokens. Every stage draws from the domain mixture, just with different sampling weights — the Reasoning stage, for instance, oversamples STEM/Math and Reasoning-traces well above their overall 14%/12% share.

> **Related but distinct from the anneal reserve** — Section 6 also defines a *protected floor* per capability slot. The anneal reserve protects a fixed final-stage token pool; the floor protects individual slots (Indic, agentic, reasoning, long-context) from being starved by an automated mixture selector at any point during the run, not just at the end.

---

## 6 · Compute budget for the mixture

### The formula, worked through with real numbers

```
Training compute (FLOPs)
C ≈ 6 × N_active × D

N_active = active parameters per token (NOT total params for an MoE model)
D        = total training tokens
```

The number of tokens (D) you can afford is a function of your compute budget and N_active — for MoE, using active params instead of total params is the whole point: it's why MoE lets you train on dramatically more data for the same FLOPs.

| Variant | N (used in formula) | D (tokens) | Total FLOPs |
|---|---|---|---|
| MoE (recommended) — 120B total / 20B active | 20B | 30T | 3.6 × 10²⁴ |
| Dense equivalent — 120B active | 120B | 30T | 2.16 × 10²⁵ (6× more) |

> **The MoE payoff in plain terms** — for the identical 3.6 × 10²⁴ FLOP budget, a dense 120B model could only train on ~5T tokens instead of 30T. Given a broad seven-domain mixture, more diverse tokens at fixed compute generally beats more parameters at fixed tokens.

### 30T-token allocation across the seven buckets

| Bucket | Share | Tokens | Why this weight |
|---|---|---|---|
| General web | 28% | 8.4T | Still the backbone for broad world knowledge and fluent general language — every frontier-class model's largest single bucket |
| Code | 20% | 6.0T | Named as an explicit priority; code also measurably improves general reasoning ability, not just coding benchmarks |
| STEM / math | 14% | 4.2T | Direct driver of quantitative-reasoning benchmarks (GSM8K/MATH/GPQA-style) |
| Reasoning traces | 12% | 3.6T | Deliberately large — the highest-leverage bucket for closing the gap to frontier-level multi-step reasoning |
| Indic languages | 10% | 3.0T | Meaningful, real weight without letting it dominate a mixture where code/reasoning/English are also explicit priorities |
| Long-context corpus | 8% | 2.4T | Dedicated budget for the length-extension stage — drawn from, not separate from, the other six buckets |
| Agentic & tool-use | 8% | 2.4T | Smaller by necessity — genuinely verified agentic trajectories are scarce and expensive to produce; quality over volume here |

### Indic's 10% (3.0T tokens), split into tiers — not a single headline number

A single "Indic = 10%" figure hides quality composition. The same 3.0T tokens split into four tiers, weighted toward the highest-trust sources:

| Tier | Share of Indic budget | Tokens | What qualifies, and why this weight |
|---|---|---|---|
| **Verified** | 35% | 1.05T | Native-authored, quality-checked (native-speaker reviewed or passed the Section 3 cleaning + audit pipeline). Largest tier deliberately — this is the highest-trust source for a capability stated as "must understand and generate natively," not "must be able to translate into." |
| **Unverified** | 30% | 0.9T | Native-authored but not individually reviewed (bulk web/crawl text post-cleaning). Second-largest — real volume filler that's still genuinely native, not translated. |
| **Translated** | 25% | 0.75T | Machine- or human-translated from English/other languages (e.g. via IndicTrans2). Fills topic/domain gaps native corpora don't cover well (technical, scientific, some agentic/tool content) — capped so it can't become the majority signal for how a language "sounds." |
| **Synthetic** | 10% | 0.3T | Model-generated Indic content (e.g. distilled from a stronger model). Smallest tier deliberately — highest risk of quality issues and hallucination amplification, especially in lower-resource languages where there's less native data to catch errors against. |

### Protected floor — the minimum a mixture selector cannot cross

If mixture proportions are adjusted by an automated selector (e.g. re-weighted based on validation loss per domain), some slots are naturally at risk of being starved toward zero — an optimizer chasing aggregate loss tends to favor abundant, easy-to-fit data like general web over scarce, harder-to-fit data like agentic trajectories. These four slots get a hard floor the selector cannot go below, independent of what it would otherwise choose:

| Slot | Target share | Protected floor | Why this slot needs a floor | Tracked via |
|---|---|---|---|---|
| Indic languages | 10% | 6% | Named as an explicit "must" capability — cannot be optimized away in favor of English-dominant loss reduction | IndicXTREME / MILU trend |
| Reasoning traces | 12% | 8% | Scarce relative to general web by construction (verified-only sourcing) — an optimizer chasing raw loss would under-sample it | GPQA-Diamond / MATH trend |
| Agentic & tool-use | 8% | 5% | Smallest, hardest-to-source bucket — most vulnerable to being crowded out entirely | BFCL / AgentBench trend |
| Long-context corpus | 8% | 4% | Directly determines whether the long-horizon agentic capability (Section 7) has enough long-session training data at all | RULER trend at fixed length |

> **General web, code, and STEM/math get no explicit floor** — they're naturally abundant and not at risk of being optimized toward zero, so a floor there would be redundant, not protective.

---

## 7 · Agentic depth, planning & failure recovery

### Long-horizon agentic capability is one joint capability, not two separate buckets

Planning a long task, calling tools across many steps, reading results, recovering from a failed call, and holding a growing task history in context — this is a single behavioral capability. Treating "agentic" and "long-context" as unrelated mixture slots misses that the hardest version of this capability only shows up when both are stressed together.

**Planning & task decomposition** — A distinct data type from tool-execution traces: examples where the model states a plan *before* acting, breaking a goal into ordered sub-tasks. Sourced by prompting a strong teacher model to plan explicitly, then verifying the plan was actually followed during execution — not just that the final answer was correct.

**Failure recovery — the gap in the earlier draft** — The earlier data-sourcing criterion ("kept only if the trajectory completes the task") **silently filters out recovery behavior**, since any trajectory with a failed intermediate step looks the same as a bad trajectory under a success-only filter. Fix: deliberately inject a fraction of sandbox failures (timeouts, malformed responses, wrong tool chosen) into trajectory generation, and keep the ones where the model recognizes the failure and recovers to still complete the overall task.

**Growing task history in context** — Sourced from the Long-context bucket's long-agent-session data (Section 2), but curated specifically for session *length in tool-call turns*, not just token count — a 50-turn agent session and a 50,000-token book stress very different things, even at similar token length.

**Where this trains** — Spans three places: pretraining exposure (Agentic bucket, Section 2), the dedicated Long-context stage (Section 5) for the session-length axis specifically, and execution-verified RL post-training for the recovery behavior specifically — recovery is hard to teach through imitation alone, since a single bad tool call in training data with no consequence doesn't teach the model to notice or correct it.

**Benchmarks to adopt** — **BFCL** (Berkeley Function-Calling Leaderboard) for tool-call correctness, **AgentBench** and **GAIA** for end-to-end multi-step task completion, **SWE-bench Verified** for agentic coding, **WebArena** for browsing-agent tasks. None of these directly measure recovery-from-failure specifically — that needs a custom internal metric (% of trajectories that recover from a deliberately injected failure), since no popular public benchmark isolates that behavior yet.

---

## 8 · Controllable reasoning depth

### Four difficulty/reasoning-length bands, each with a concrete example

The goal isn't just "the model can reason at length" — it's that reasoning depth scales with problem difficulty, and that scaling is *controllable* rather than a fixed habit. This requires the reasoning-traces bucket to be explicitly banded by difficulty during sourcing, not just labeled "reasoning" as one undifferentiated pile.

| Band | Target trace length | Concrete example | Benchmarks that measure it |
|---|---|---|---|
| **Band 1 — Direct** | ~0–50 tokens | "What is the capital of France?" — answered directly, no visible reasoning needed. Training data for this band explicitly includes short, confident, non-reasoning responses so the model learns *not* to over-reason on easy inputs. | MMLU-Pro (easy subset) |
| **Band 2 — Light** | ~50–300 tokens | "A train travels 60 km in 45 minutes — what is its speed in km/h?" — one or two arithmetic steps, shown briefly. | GSM8K |
| **Band 3 — Moderate** | ~300–1,500 tokens | A multi-step word problem, or a debugging task requiring tracing program state across several function calls before finding the bug. | MATH, ARC-AGI |
| **Band 4 — Deep** | ~1,500–8,000+ tokens | An AIME-style olympiad problem requiring exploring and discarding a wrong approach before finding the right one, or a multi-file refactor requiring weighing several designs before committing. | AIME, GPQA-Diamond |

> **Controllability, not just correlation** — banding the training data by difficulty teaches the model that reasoning length varies, but doesn't by itself guarantee the model can be steered. Two additional mechanisms are needed: (1) pretraining exposure where trace length correlates consistently with band-appropriate problems, so the model learns the association rather than a fixed average length, and (2) RL post-training reward shaping that penalizes unnecessarily long reasoning on Band 1–2 problems and rewards thorough exploration on Band 4 — imitation alone tends to regress toward the mean trace length across training data. None of the benchmarks above measure length-calibration directly, either — that also needs a custom internal metric (tokens used vs. band-appropriate target, at matched accuracy).

---

## 9 · Proxy-run validation

### Every number above is a hypothesis until a 1B/3B-scale run tests it

Nothing in Sections 2–8 gets trusted at the full 120B/30T scale without cheap validation first. This is the single most important process commitment in the whole plan — a mixture ratio decided by reasoning alone, however well-argued, is still a guess.

**Stage 1 — 1B-parameter screening** — Train several 1B-parameter models, each on a fixed small token budget, one per candidate mixture variant (e.g. baseline vs. +5% agentic vs. alternate reasoning-band ratios vs. different Indic tier splits). Cheap enough to run many variants in parallel.

**Stage 2 — 3B-parameter confirmation** — The top 2–3 candidates from Stage 1 get re-run at 3B scale. Scaling trends can shift between 1B and 3B — this catches a variant that looked good only because it happened to suit a very small model.

**What gets measured** — Per-domain validation loss (not just aggregate loss — a mixture change can help one bucket while quietly hurting another), plus small-scale proxies of the Section 11 benchmark suite: a stratified MMLU-Pro subset, a small HumanEval/MBPP set, a GSM8K subset, a small IndicXTREME/MILU sample, a mini-BFCL agentic set, and RULER at short context lengths (4K/8K) as an early signal before the full 128K target.

**The decision rule** — A mixture change is adopted for the full 120B/30T run only if it improves the relevant proxy metric at **both** 1B and 3B scale beyond a pre-set threshold — not just at one scale, and not on a single run. This guards against promoting a change that's actually just noise from one small, cheap experiment.

> **The principle this whole section exists to enforce** — a data-mixture decision is a hypothesis until a cheap experiment has tested it. Every percentage in Sections 2, 5, and 6 should be read as "current best hypothesis, pending proxy-run confirmation," not as a locked-in number.

---

## 10 · Hardware requirements

### Sized against the 30T-token, 3.6×10²⁴-FLOP budget above

| Layer | Recommendation | Sizing logic |
|---|---|---|
| Data processing / chunking | Few-hundred-node CPU cluster (Ray/Spark), object storage (S3-compatible) for raw + intermediate stages | Cleaning 30T tokens' worth of raw crawl (likely several × more raw volume before filtering) is I/O- and CPU-bound, not GPU-bound — keep it off the GPU cluster entirely |
| Tokenization | Rust/rayon-parallel BPE, run across the same CPU cluster | Throughput is implementation- and hardware-dependent; budget it as CPU-core-hours ≈ D ÷ (per-core tokens/sec), and validate on a 1% sample before committing the full run |
| Training compute | ~10,000 H100-class GPUs (8/node, NVLink/NVSwitch intra-node, 400–800 Gbps InfiniBand inter-node) | At ~450 TFLOP/s sustained per GPU (~45% MFU), the 3.6×10²⁴ FLOP MoE budget completes in ~9 days ideal / ~12 days with realistic overhead |
| Networking | Low-latency, high-bandwidth InfiniBand fabric — non-negotiable for MoE | Expert-parallel all-to-all communication is the actual bottleneck for MoE, not compute — under-provisioned networking silently erodes your effective MFU |
| Memory (weights + optimizer) | ~2.16 TB total (120B params × ~18 bytes/param for bf16 weights + gradients + fp32 master + Adam states), sharded via ZeRO-3/FSDP across the cluster | No single GPU holds the full model — sharding strategy (data/tensor/pipeline/expert parallelism) is a first-class design decision, not an afterthought |
| Storage — raw ingest | ~2–5 PB | Raw crawl volume before cleaning/dedup typically runs several times the size of the final cleaned corpus |
| Storage — cleaned/deduped text | ~400–500 TB | Cleaned text corresponding to 30T tokens at ~4–5 bytes/token average across mixed scripts |
| Storage — tokenized shards | ~120 TB | 30T tokens × 4 bytes (uint32 — needed since the ~230K vocab exceeds uint16 range) |
| Storage — checkpoints | Fast parallel filesystem (Lustre/GPFS); ~2.16 TB per full checkpoint, rolling window of several kept | Frequent checkpointing is the primary defense against the node failures expected at this cluster scale over a multi-day run |

---

## 11 · Benchmark alignment

### What "meeting frontier benchmarks" actually requires beyond this document

Everything above gets you a strong pretrained base model. It does not, by itself, get you to Claude/GPT/Gemini-level benchmark scores — that gap is closed substantially in post-training, which is its own multi-month effort:

| Target capability | Evaluation suite |
|---|---|
| General reasoning | MMLU-Pro, GPQA-Diamond |
| Coding | HumanEval, SWE-bench Verified, LiveCodeBench |
| Math | MATH, AIME, GSM8K |
| Agentic / tool-use | Berkeley Function-Calling Leaderboard, AgentBench, GAIA |
| Long-context | RULER, needle-in-haystack variants at target length |
| Indic languages | IndicXTREME, MILU, Aya evaluation suite |

> **Being direct about this** — matching frontier-lab benchmark numbers takes large-scale RLHF/RLAIF, extensive human-preference data collection, iterative red-teaming, and many rounds of targeted post-training against exactly these evals — none of which is a pretraining architecture decision. This document is the foundation that post-training effort builds on, not a substitute for it.

---

*120B Blueprint · Architecture & training design · Recommendations grounded in public research, not proprietary lab data*
