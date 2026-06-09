# 10 — Neural Networks Reference

> The exact spec of the **one network V6 trains** (the brain's `BrainHead`) and the
> **frozen pretrained models** it consumes. For the idea behind the encoder+MLP
> shape see [The Latent Idea](02-the-latent-idea.md); for the training recipe see
> [Training](09-training.md).

Conventions: `Linear(a→b)` is a fully connected layer; "frozen" = `requires_grad =
False`, no gradients.

---

## 1. `BrainHead` — the trained policy MLP ★

> Code: `v6/brain.py`. This is the **only** network with trainable weights in V6.

### Architecture

```
situation vector x
   │  shape = SITUATION_DIM = 2·EMBED_DIM + OUTCOME_DIM = 2·1024 + 25 = 2073
   ▼
trunk = nn.Sequential(
          nn.Linear(2073, 256),     # the only wide layer
          nn.ReLU(),
          nn.Dropout(0.1),
        )                            # → h (256-d)
   │
   ├─ intent = nn.Linear(256, 6)    # softmax → greeting/meta/definition/data/unanswerable/off_topic
   ├─ action = nn.Linear(256, 5)    # softmax → rag/sql/chart/email/template
   └─ cont   = nn.Linear(256, 1)    # sigmoid → continue score (the seuil), squeezed to a scalar
```

| Property | Value |
|---|---|
| Input dim | **2073** (`= 1024 query + 1024 memory + 25 outcome`) |
| Hidden | **256** |
| Heads | intent (6), action (5), continue (1) |
| Activation | **ReLU** (trunk) |
| Regularization | **Dropout 0.1** (trunk) |
| Output transforms | softmax (intent, action), sigmoid (continue) |
| **Trainable params** | **≈ 534K** (`2073·256 + 256` trunk + `256·(6+5+1) + 12` heads) |

### The input — how `x` is built

```
x = BGE-M3(query) ⊕ BGE-M3(memory) ⊕ encode_outcome(step_log, grounding)
        1024              1024                    25                  = 2073
```

- Query/memory embeddings are **frozen BGE-M3** outputs (1024-d, L2-normalized),
  encoded **once per turn** and cached per `thread_id`. Only the 25-d outcome vector
  changes per tick.
- `encode_outcome` is the single source of truth for the 25-d vector, imported by
  both the live brain and the trace synthesizer so training and inference featurize
  identically.

### The 25-d outcome vector layout (`OUTCOME_DIM`)

| Slice | Size | Contents |
|---|---|---|
| last-action one-hot | 6 | `[none] + [rag, sql, chart, email, template]` |
| last-ok | 1 | 1 if the last action succeeded |
| error-type one-hot | 7 | `none, sql_error, sql_no_rows, sql_no_query, email_no_recipient, artifact_failed, rag_weak` |
| row-bucket one-hot | 4 | `none, zero, one, many` |
| attempt-count | 1 | `min(attempts, 3) / 3.0` |
| grounding | 1 | RAG cosine clamped to `[0, 1]` |
| done-actions multi-hot | 5 | which of `[rag, sql, chart, email, template]` already ran |

### Optimization (exact)

| Field | Value |
|---|---|
| Optimizer | **Adam** |
| Learning rate | **1e-3** |
| Weight decay | **1e-4** |
| Epochs | **160** default |
| Batching | **full-batch** (one step/epoch; data embedded once up front) |
| Val split | 15% |
| **Loss** | `BCEWithLogitsLoss(pos_weight)` on continue (all rows) + `CrossEntropyLoss(class_weight)` on intent (step-0 rows only) + `CrossEntropyLoss(class_weight)` on action (valid-label rows) |
| Class imbalance | inverse-frequency CE weights; `pos_weight = #(cont==0)/#(cont==1)` for BCE |
| Masking | intent → step-0 rows; action → rows with `label_action ≥ 0`; continue → all |

### Inference

`Brain.decide(query, memory, step_log, grounding, thread_id)` builds `x`, runs the
head under `torch.no_grad()`, takes `softmax` over intent/action logits and
`sigmoid` over the continue logit, and returns a `BrainDecision` (argmax intent +
action, their confidences, the continue score, and the full probability dicts for
debugging). It **refuses to construct** without `models/brain_head.pt` — an
untrained head would route randomly.

### Conventions

- **Multi-class, single winner** → softmax + cross-entropy (intent, action).
- **Binary gate** → sigmoid + BCE (continue/seuil).
- **Imbalance** → inverse-frequency class weights + `pos_weight`, plus label masking
  so a row only supervises the heads that apply to it.

---

## 2. The frozen pretrained models V6 consumes

None of these are trained in V6; they are feature providers / generators behind the
deterministic cage.

| Model | Params | Role | Notes |
|---|---|---|---|
| **BGE-M3** (XLM-RoBERTa-based) | ~1.5B | RAG retrieval **and** the brain's encoder (1024-d CLS, L2-normalized) | ❄ frozen; **CPU-forced on Apple Silicon** (MPS gives wrong batch-of-1 embeddings); GPU on CUDA |
| **Qwen3-4B-Instruct-2507** | 4B | dual-role SLM: router (phase 1) + SQL gen (phase 2) + polisher (4 roles) | default; shares a KV cache across phases; `enable_thinking=False` |
| Qwen3-0.6B | 0.6B | speculative-decoding drafter for the 4B | optional (`USE_SPECULATIVE`) |
| Qwen2.5-1.5B-Instruct | 1.5B | standalone polisher | only when `POLISHER_USE_MAIN=0` |
| faster-whisper `large-v3` | — | STT (CTranslate2) | `float16` GPU / `int8` fallback; domain-primed + fuzzy-corrected |
| Coqui **XTTS-v2** | — | TTS, voice cloning | streaming; per-language latents cached |

Model sizes are configurable: `V6_SLM_SIZE` (`3b`/`4b`/`7b`), `V6_4BIT` (NF4 to fit
7B on a 16 GB T4), `V6_SLM_OVERRIDE` (force a Hub id).

---

## 3. Why only one trained network

The expensive capabilities — understanding multilingual text, writing SQL,
transcribing, synthesizing voice — are already solved by pretrained models. The one
thing nothing off-the-shelf provides is **the policy**: given this situation, what
should the agent do next, and is it done? That is the brain, and it is deliberately
tiny (~534K params) because the heavy lifting (turning the situation into meaningful
vectors) is done by the frozen BGE-M3 encoder. This is the encoder+MLP pattern in
its purest form — see [The Latent Idea](02-the-latent-idea.md).

→ Next: [Benchmarks](11-benchmarks.md).
