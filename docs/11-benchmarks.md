# 11 — Benchmarks

> Code: `v6/benchmark.py`, fixtures in `v6/data/bench_queries.json`, held-out brain
> eval in `v6/data/brain_eval.jsonl`.

> ⚠️ **Read this first.** The result *numbers* here are **illustrative
> placeholders** — replace them with measured values. The **methodology** is
> accurate: what each metric means, how it's computed, why it's the right one. The
> one figure with documented provenance is the brain's **86% → 97%** held-out jump.

---

## 1. How V6 benchmarks itself (`benchmark.py`)

The harness runs a fixed **20-query fixture set** (`bench_queries.json`) through the
**live graph**, and **mirrors the notebook's terminal routing exactly**, so the
answers it grades are the same answers the user sees and hears (clean frozen
numbers, the correct polisher role, verbatim deflections).

The fixtures alternate **French / English**, span **every intent and capability**,
and queries sharing a `thread` run in order so **follow-ups inherit context.** Each
fixture carries an `intent` (graded automatically), an `expects` hint (the polisher
rephrases, so final-answer correctness is verified by hand), and `category`/`lang`.

### What it measures

| Metric | Definition |
|---|---|
| **Intent accuracy** | predicted intent vs. expected label, per query |
| **SQL exec rate** | share of *data* queries whose SQL ran **and returned rows** |
| **Latency** | brain ticks, RAG, SQL, total — by category |
| **Answer review** | per query: the *spoken* text (post-polish) beside the *raw* data block |
| **Voice round-trip** (GPU only) | synthesize each query → WAV → STT → **WER**, then diff voice-vs-text and list only the queries that **regressed under voice** |

The **voice-regression** idea is the clever part: a query is flagged only when the
*voice* answer diverges from the *text* answer for the **same** query — almost
always traceable to the reported WER / an STT mishearing, **not** the brain. So
transcription noise is never mistaken for a pipeline bug.

```
python3 -c "from v6.benchmark import run_text; run_text()"        # text only (fast)
python3 -c "from v6.benchmark import run_full; run_full(save=True)"  # + voice round-trip (GPU)
```

The polisher routing inside the benchmark (`_spoken_role`) is kept **byte-for-byte
in sync** with the notebook's `_pick_role`, so `off_topic` and cross-turn report
messages are spoken **verbatim** (role `None`), exactly as in production.

### Reading a result

- A figure shown as `253.4 million DZD` in the **spoken** text and `253,387,711.02`
  in the **raw** line is the number-formatting fix working: `numfmt` froze the clean
  figure, the analyst copied it.
- A voice regression with a high WER on a wilaya name is an STT problem, not a
  routing problem.

---

## 2. Illustrative results

### Intent & routing (text, 20-query fixture)

| Metric | Illustrative value |
|---|---|
| Intent accuracy (overall) | **19/20 (95%)** |
| greeting / meta / definition | 8/8 |
| data | 9/9 |
| unanswerable / off_topic | 2/3 (one ambiguous definition↔data) |
| SQL exec rate (data queries) | **9/9 (100%)** |
| Brain recovery (injected SQL failure → retry succeeds) | ✓ |

### Held-out brain (generalization, `brain_eval.jsonl`)

| Metric | Illustrative value |
|---|---|
| Intent (held-out) | ~0.97 |
| Action (held-out) | ~0.95 |
| Continue / stop timing | ~0.94 |
| **Overall held-out routing** | **~97%** (up from ~86% before step-0 intent training + targeted data) |

The ~86% → ~97% jump is the one result with documented provenance (the git
history): a held-out eval exposed gaps, targeted data closed them.

### Latency (illustrative, Qwen3-4B on a T4, KV-cache reuse on)

| Stage | Time |
|---|---|
| Brain (per tick, ~12 ticks worst case) | < 5 ms/tick |
| RAG (BGE-M3 encode + retrieve) | ~120 ms |
| Router (phase 1) | ~1.8 s |
| SQL-gen (phase 2, KV reused) | ~0.9 s (≈2.4 s without reuse) |
| SQL execution | ~30 ms |
| Polish (streaming, first token) | ~0.4 s |
| **Total data query (text)** | **~3.5 s** (first token ~0.5 s) |
| Greeting / definition (no SQL) | ~0.8 s |

The brain's overhead is **negligible** — the SLM dominates. KV-cache reuse saves
~1.5 s/query on the phase-1→phase-2 hand-off.

### Voice round-trip (illustrative, GPU)

| Metric | Illustrative value |
|---|---|
| STT WER (clean fixtures, primed + fuzzy-corrected) | ~6% |
| WER on wilaya-heavy queries | ~14% pre-correction → ~5% after fuzzy correction |
| Voice regressions vs. text (out of 20) | 1 (a mis-heard wilaya, traced to WER) |
| TTS first-audio latency | ~1 sentence behind the first text token |

---

## 3. What to actually measure (for whoever replaces these numbers)

1. **Intent accuracy** on the 20-query fixtures **and** the held-out
   `brain_eval.jsonl` — report both (fixtures can memorize, held-out can't).
2. **SQL exec rate *and* answer correctness** — exec rate alone can be gamed by a
   query that runs but answers the wrong question; grade `expects` by hand.
3. **Latency by category** with KV-reuse on vs. off, to keep the hand-off saving
   honest.
4. **WER and voice-regression count** separately — never blame the brain for an STT
   mishearing.
5. **The number-fidelity check** — for every figure, confirm the spoken/charted
   value equals the raw DB value to the rounding rule. This is the one that, if it
   ever fails, is a critical bug.

→ Next: [Design Decisions](12-design-decisions.md).
