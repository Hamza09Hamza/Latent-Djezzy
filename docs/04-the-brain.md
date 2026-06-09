# 04 — The Brain

> Code: `v6/brain.py` (the model), `v6/brain_data.py` (the policy spec),
> `v6/train_brain.py` (training), `v6/eval_brain.py` (held-out eval),
> `v6/data/brain_train.jsonl` (synthesized rows), `v6/data/brain_eval.jsonl`
> (hand-written held-out cases).

The brain is the heart of V6: a **trained 3-head MLP**, called **once per loop
tick**, that decides what the agent does. It does not generate text or touch the
database — it looks at the situation and answers three questions at once.

---

## 1. What it decides

1. **What kind of question is this?** — `intent` (decided at tick 0, held all turn).
2. **What should I do next?** — `action`.
3. **Do I want another step, or am I done?** — `continue` score, the **seuil**.

The loop is: *brain decides → an action runs → its outcome is encoded back into the
brain's input → brain decides again.* A learned policy over a tiny action space,
with a learned stopping criterion.

---

## 2. The situation vector (2073-d)

Every tick, the brain's input is built fresh:

```
situation = query_emb (1024)  ⊕  memory_emb (1024)  ⊕  outcome_vec (25)
          =                      2073-dimensional vector
```

- **`query_emb`** — the question, encoded by **BGE-M3** (1024-d, L2-normalized).
  Computed **once per turn** and cached per `thread_id`.
- **`memory_emb`** — the conversation memory (recent turns + compacted summary),
  encoded by BGE-M3. Also cached per turn. This is how the brain knows it's a
  follow-up: the memory vector shifts the query into the right region of space.
- **`outcome_vec`** — a 25-d engineered feature vector encoding everything done so
  far this turn, rebuilt every tick from `step_log`.

Only the 25-d outcome vector changes between ticks, so re-deciding is cheap: two
cached 1024-d vectors + a fresh 25-d slice through a 256-wide MLP.

### The 25-d outcome vector, slice by slice

The brain's "proprioception" — its sense of what it has done and how it went:

| Slice | Size | Meaning |
|---|---|---|
| last-action one-hot | 6 | `[none, rag, sql, chart, email, template]` |
| last-ok | 1 | 1 if the last action succeeded |
| error-type one-hot | 7 | `none, sql_error, sql_no_rows, sql_no_query, email_no_recipient, artifact_failed, rag_weak` |
| row-bucket one-hot | 4 | `none, zero, one, many` (how many rows SQL returned) |
| attempt-count (normalized) | 1 | `min(attempts, 3) / 3.0` |
| grounding score | 1 | RAG cosine in `[0, 1]` |
| done-actions multi-hot | 5 | `[rag, sql, chart, email, template]` already executed |

This is what lets the brain **react**. A zero-row SQL result lights up
`row-bucket=zero`; a bad column lights up `error-type=sql_error`; a missing
recipient lights up `email_no_recipient`. Because the training traces taught the
brain what to do when those bits are hot, it can retry SQL, skip a chart, or stop
and clarify — *with no `if` statement in the graph.* `encode_outcome` is the single
source of truth for this vector, imported by both the live brain and the trace
synthesizer, so training and inference featurize identically.

---

## 3. The model (`BrainHead`)

```
situation (2073)
   │  Linear(2073 → 256) → ReLU → Dropout(0.1)
   ▼
trunk (256)
   ├─ Linear(256 → 6)  → intent logits   (softmax → 6 probabilities)
   ├─ Linear(256 → 5)  → action logits   (softmax → 5 probabilities)
   └─ Linear(256 → 1)  → continue logit  (sigmoid → seuil ∈ [0, 1])
```

A shared trunk with three task-specific heads. **≈534K trainable parameters.** The
intent head sizes itself from `len(INTENTS)`; adding an intent (as `off_topic` was)
auto-resizes the layer and requires a retrain. Full layer spec in
[Neural Networks Reference](10-neural-networks-reference.md).

**It fails loudly without weights.** The brain refuses to run if
`models/brain_head.pt` is missing — an untrained head would route randomly and
silently corrupt every answer. So the system stops with build instructions instead.

### The six intents

`greeting · meta · definition · data · unanswerable · off_topic`

Everything except **`data`** short-circuits straight to the communicator.
**`off_topic`** ("write me code," "what's the weather," "translate this") gets a
**deterministic, language-matched deflection** and is **never** sent to the polisher
— a small model can be coaxed into *attempting* the off-topic task, so the safe
design is to never give it the chance. (`meta` — "what can you do?" — is distinct:
it gets a warm capability reply through the chat persona.)

### The five actions

`rag · sql · chart · email · template` — the spokes of the star graph.

---

## 4. The policy is data: `brain_data.py`

This file **is** the specification of what the system should do; there is no
behavior spec anywhere else.

**How a trace becomes rows.** Each trace defines an `intent`, a template query, and
a **gold sequence** of `(action, outcome)` pairs. `_expand(trace)` generates **one
training row per tick**: tick 0 has an empty `step_log` (the brain sees nothing
done yet), tick 1 sees the result of the first action, etc. The **final tick of
every trace has `label_continue = 0`** — exactly where the brain must fire the
seuil and stop. A helper `_terminal(trace)` adds extra rows on the stopping state
after a terminal action to strengthen the stop signal.

**The trace families** (the agent's behavior, in editable form):

| Family | Gold action sequence |
|---|---|
| `greeting`/`meta`/`definition`/`unanswerable`/`off_topic` | (no actions) → communicator |
| `data_only` | rag → sql → communicator |
| `data_chart` | rag → sql → chart → communicator |
| `data_email` | rag → sql → email → communicator |
| `data_template` | rag → sql → template → communicator |
| `data_chart_email` | rag → sql → chart → email → communicator |
| `sql_retry_ok` | rag → sql(fail) → sql(ok) → communicator |
| `sql_fail_twice` | rag → sql(fail) → sql(fail) → communicator |
| `email_no_recipient` | rag → sql → email(no_recipient) → communicator |
| `followup` | sql → communicator (no rag — inherits context) |

Want a new behavior? Add a trace family and retrain. Traces are seeded from query
prototypes (`data/planner_prototypes.json`: intents
`greeting/data/definition/meta/unanswerable`, capabilities `viz/email/template`)
for realistic multilingual phrasings. Output: `v6/data/brain_train.jsonl` (~8K
rows).

---

## 5. Training (summary)

The featurizer encodes each unique `(query, memory)` once with BGE-M3, builds the
2073-d situation per row, and optimizes **three masked, class-weighted losses
jointly**:

- **CE on intent** — **step-0 rows only** (intent is decided once per turn).
- **CE on action** — rows with a valid action label (continuing rows).
- **BCE on continue** — all rows, `pos_weight`-balanced (the seuil must be right
  every tick).

Optimizer **Adam, lr 1e-3, weight_decay 1e-4**, **full-batch**, **160 epochs**
default. Full recipe in [Training](09-training.md) and
[Neural Networks Reference](10-neural-networks-reference.md).

The step-0 restriction on the intent loss is the subtle, important part: training
intent on later ticks both teaches a never-read signal and inflates the multi-tick
`data` class (which the inverse-freq weighting then over-corrects into a
"count→unanswerable" basin). Restricting it to step-0 rows is what moved held-out
accuracy ~86% → ~97%.

---

## 6. Held-out evaluation (`eval_brain.py`)

High accuracy on the synthetic split could be **template memorization.** So there
is a separate, **hand-written** held-out set, `brain_eval.jsonl`, whose phrasings
are intentionally **not** drawn from `brain_data.py`'s templates — accuracy there
measures **generalization**. `train_brain.py` auto-runs it after saving and prints
it as the number to trust.

- IID synthetic val (runs hot): intent ~0.999 / action ~0.997 / continue ~0.960.
- **Held-out (trust this): ~97%** overall, up from ~86% before the step-0 intent
  fix + targeted data. (Documented in the git history.)

**The discipline:** never grade a learned router on the data that trained it.

---

## 7. Why a trained MLP and not rules or an LLM controller

- **Not regex:** rules match surface forms and need a new line per phrasing; they
  die on Darija and ASR noise. The MLP classifies *meaning* in BGE-M3 space and
  generalizes. (See [The Latent Idea](02-the-latent-idea.md).)
- **Not a free-running LLM controller:** an LLM controller is slow, nondeterministic,
  and hard to constrain. A ~534K MLP decides in well under a millisecond, is
  reproducible, and its behavior is fully specified by an editable trace file.
- **Why the seuil gates everything, including terminals:** a brain that exempts
  terminals can fire a report on every SQL result (the action argmax might be
  "template" even when the brain is done). Treating the seuil as "do I want
  *another* step, regardless of which" makes a deliberate "put it in a report"
  (trained to continue ≈ 1.0) different from an incidental argmax (continue ≈ 0.01).

→ Next: [RAG, Knowledge & Entities](05-rag-knowledge-entities.md).
