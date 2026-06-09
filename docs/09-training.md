# 09 — Training

> Code: `v6/brain_data.py` (synthesize traces), `v6/train_brain.py` (train),
> `v6/eval_brain.py` (held-out eval). Data: `v6/data/brain_train.jsonl`,
> `v6/data/brain_eval.jsonl`.

V6 trains exactly **one** thing: the brain's MLP head. Everything else — the SLM
that writes SQL, the BGE-M3 encoder, the STT/TTS models — is pretrained and used
as-is. The expensive capabilities are delegated; the only thing worth training is
the **policy** (what to do), because nothing off-the-shelf provides it.

```
python3 -m v6.brain_data            # 1. synthesize traces → data/brain_train.jsonl
python3 -m v6.train_brain --epochs 160   # 2. train → models/brain_head.pt
python3 -m v6.cli                   # 3. the agent loads the trained brain
```

On Colab, the training cells (`FORCE_RETRAIN=True`) do both. Re-run after any change
to `brain_data.py` (e.g. adding an intent, which resizes the intent head).

---

## 1. Step 1 — synthesize the policy traces (`brain_data.py`)

`brain_data.py` **is** the editable specification of the agent's behavior. Each
trace defines an `intent`, a template query, and a **gold sequence** of
`(action, outcome)` pairs; `_expand(trace)` turns each trace into **one training row
per loop tick** (tick 0 sees an empty `step_log`, tick 1 sees the first action's
result, …). The final tick of every trace has `label_continue = 0` — where the
brain must fire the seuil and stop — and `_terminal(trace)` adds extra
stopping-state rows to strengthen that signal.

Trace families (full table in [The Brain](04-the-brain.md)): the non-data intents
(→ communicator only), `data_only`, `data_chart`, `data_email`, `data_template`,
`data_chart_email`, `sql_retry_ok`, `sql_fail_twice`, `email_no_recipient`,
`followup`. Phrasings are seeded from `data/planner_prototypes.json`. Output:
`brain_train.jsonl` (~8K rows).

---

## 2. Step 2 — train the head (`train_brain.py`)

### Pipeline

1. Read `brain_train.jsonl`, shuffle (`Random(0)`).
2. **Encode each unique (query, memory) string once with BGE-M3** (cached and
   reused across a trace's ticks — strings repeat).
3. Build the 2073-d situation per row: `BGE-M3(query) ⊕ BGE-M3(memory) ⊕
   encode_outcome(step_log, grounding)`.
4. 85/15 train/val split.
5. Optimize three losses jointly; save `models/brain_head.pt`.

### The exact recipe

| Field | Value |
|---|---|
| Optimizer | **Adam** |
| Learning rate | **1e-3** (`--lr` overridable) |
| Weight decay | **1e-4** |
| Epochs | **160** default (`--epochs` overridable; Colab sometimes 200) |
| Batching | **full-batch gradient descent** — one forward/backward/step over the whole training set per epoch (small dataset, embedded once up front) |
| Validation split | 15% (`val_frac = 0.15`) |
| Featurizer | BGE-M3, frozen |

### The three losses (summed)

```
loss =  BCEWithLogitsLoss(pos_weight)        on continue,  ALL rows
      + CrossEntropyLoss(class_weight)       on intent,    STEP-0 rows only
      + CrossEntropyLoss(class_weight)       on action,    rows with a valid action label
```

- **Continue — BCE, all rows.** `pos_weight = (#continue==0) / (#continue==1)` so
  "keep going" and "stop" are balanced (the seuil must be right at *every* tick).
- **Intent — CE, step-0 rows only.** Intent is decided at tick 0 and reused all turn,
  so it is supervised only on step-0 rows (the ticks whose situation matches
  inference). Training it on later ticks teaches a never-read signal *and* inflates
  the multi-tick `data` class, which the inverse-freq weighting then over-corrects
  into a "count→unanswerable" basin. **This step-0 restriction is the change that
  moved held-out accuracy ~86% → ~97%.**
- **Action — CE, continuing rows only.** The last tick of a trace has no action
  label (index −1, masked out) — nothing left to do.
- **Class weighting:** both CE losses use **inverse-frequency** weights so rare
  classes (chart/email/template; greeting/meta) aren't drowned by `data`/`rag`/`sql`.

### Cost

Trains in **seconds on CPU** once the embeddings are cached (~3–4 min on a T4
including the one-time BGE-M3 encode). ~534K trainable parameters.

---

## 3. Step 3 — held-out evaluation (`eval_brain.py`)

High accuracy on the synthetic split could be **template memorization.** So
`brain_eval.jsonl` is a separate, **hand-written** set whose phrasings are
intentionally not from `brain_data.py`'s templates. `train_brain.py` auto-runs it
after saving and prints it as the number to trust.

| Metric | Value |
|---|---|
| IID synthetic val (runs hot) | intent ~0.999 / action ~0.997 / continue ~0.960 |
| **Held-out (trust this)** | **~97%** overall, up from ~86% before the step-0 fix |

**The discipline:** never grade a learned router on the data that trained it.

---

## 4. The pretrained models V6 uses (not trained here)

| Model | Role | Status |
|---|---|---|
| **BGE-M3** (XLM-RoBERTa-based, 1024-d) | RAG retrieval **and** the brain's encoder | ❄ frozen |
| **Qwen3-4B-Instruct-2507** | dual-role SLM (router + SQL gen) + polisher | used as-is |
| Qwen3-0.6B | speculative-decoding drafter | used as-is |
| Qwen2.5-1.5B-Instruct | standalone polisher (when `POLISHER_USE_MAIN=0`) | used as-is |
| faster-whisper `large-v3` | STT | used as-is |
| Coqui XTTS-v2 | TTS (voice cloning) | used as-is |

This is the point of the design: train a ~534K-param policy, rent everything
expensive. See [The Latent Idea](02-the-latent-idea.md) for why the encoder+MLP
split makes that possible, and [Neural Networks Reference](10-neural-networks-reference.md)
for the exact `BrainHead` spec.

→ Next: [Neural Networks Reference](10-neural-networks-reference.md).
