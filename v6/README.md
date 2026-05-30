# LatentMind V6

An agentic analytics assistant for an Algerian telecom database (Djezzy),
built as a **LangGraph** state machine. It answers KPI questions in French,
English, or Arabic/Darija — and on request charts the result, drafts an email,
or fills a report — while remembering the conversation across turns. A voice
layer (speech in, speech out) wraps the same pipeline.

V6 replaced V5's one-shot planner with a **trained policy loop**: a small MLP
"brain" picks one action at a time, watches what happens, and re-decides until
it is confident the turn is done. For the full design see
[docs/architecture.md](docs/architecture.md).

## The brain — a trained policy, not a rules engine

Routing is **latent**: the query, the conversation memory, and an encoded
outcome of what has happened so far are fed to a trained 3-head MLP that
predicts the intent, the next action, and a *continue* score (the **seuil**).
There is no regex routing and no free-running LLM controller — behaviour lives
in the training traces ([brain_data.py](brain_data.py)), not in `if` statements.

```
START
  │
  ▼
brain ───────────────┐   (every action loops back; the brain re-decides)
  ├─→ rag ───────────┤
  ├─→ sql ───────────┤   sql = router(phase 1) → validate → generate(phase 2) → execute
  ├─→ chart ─────────┤
  ├─→ email ─────────┤
  ├─→ template ──────┘
  │
  └─→ communicator → END   (when continue_score < BRAIN_SEUIL)
```

The brain reacts to failure: a 0-row result, a bad column, an email with no
recipient all show up in the outcome vector, and the brain re-picks (retry SQL,
skip the chart, stop and clarify) because the traces taught it to.

### Six intents

`greeting · meta · definition · data · unanswerable · off_topic`. Everything
except `data` short-circuits straight to the communicator. **`off_topic`**
(write me code, what's the weather, translate this) gets a *deterministic,
language-matched deflection* and is **never** sent to the polisher — a 1.5B
model can't be coaxed into actually doing it.

## Numbers are formatted in Python, never by the model

This is an analytics tool, so a wrong number is a critical failure. The polisher
is a 1.5B model and *will* mangle a long figure (it once turned
`1,087,355,290.78` into `52,590,189,81` while "rounding"). So formatting is
pulled out of the model entirely:

- [numfmt.py](numfmt.py) rounds, scales, and unit-tags every figure
  deterministically (`253,387,711.02 → "253.4 million DZD"`,
  `42.4247 → "42.42%"`), in the query's language, **before** the polisher sees it.
- The analyst's job becomes pure prose: it copies the frozen figure verbatim —
  it never re-rounds, does arithmetic, or invents a `$`.

## Voice layer

[speech.py](speech.py) wraps the text pipeline: faster-whisper (`large-v3`) for
STT, Coqui XTTS-v2 (streaming) for TTS. Before synthesis, `speakable()`
rewrites each sentence into what it should *sound* like — `DZD`→"dinars",
`%`→"percent", drops chart/report file-path lines, and collapses any stray long
number — so the voice never spells "D-Z-D" or reads ten digits one by one.
Voice quality is tuned via the `V6_TTS_*` knobs; cloning a reference WAV
(`V6_TTS_SPEAKER_WAV_*`) is the biggest quality lever.

## Why V6 exists — the V5 bugs, and the fix

| V5 failure | Root cause | V6 fix |
|---|---|---|
| `no such column: wilaya` | schema/glossary/prompts all *claimed* a metric table had a `wilaya` column | [schema.py](schema.py) introspects the live DB; the join map is **derived** — every metric table joins `dim_location` via `location_id` |
| churn comparison silently dropped a city | DB stores `Alger`, the query said `Algiers` | [entities.py](entities.py) resolves mentions to real values (alias + accent-fold + fuzzy) |
| "hello" generated SQL | the model routed in free text, parsed by keyword grep | the [brain](brain.py) classifies intent in embedding space; a greeting never reaches SQL |
| a trend query invented a `WHERE Oran` filter | history bled into the prompt + KV cache | [sql_tools.py](sql_tools.py) `consistency_check` flags any filter not in the resolved intent; bounded regenerate |
| no recovery when a step failed | rigid linear pipeline | the brain re-decides after every step and can retry SQL, skip a capability, or stop |
| a long figure read back corrupted | a small model reformatting raw numbers | [numfmt.py](numfmt.py) freezes clean figures; the polisher only copies them |

## Capabilities

- **chart** — matplotlib (line for trends, bar for comparisons), saved to disk.
- **template** — a Jinja2 report ([templates/report.md.j2](templates/report.md.j2))
  filled with the data and written to disk.
- **email** — resolves a recipient from the `contacts` table and **drafts** an
  email. It never sends; `capabilities.send_email(draft)` is a separate explicit
  action (needs `V6_SMTP_*`).

## Build the brain (required once)

The brain is inert without `models/brain_head.pt`:

```bash
python3 -m v6.brain_data            # synthesize traces → data/brain_train.jsonl
python3 -m v6.train_brain --epochs 200   # train → models/brain_head.pt
```

On Colab, the training cells (`FORCE_RETRAIN=True`) do both. Re-run them after
any change to `brain_data.py` (e.g. adding an intent).

## Run locally

```bash
pip install -r v6/requirements.txt
python3 -m v6.test_v6              # verification harness
python3 -m v6.numfmt              # show the number-formatting table (en + fr)
python3 -c "from v6.benchmark import run_text; run_text()"   # objective benchmark
```

The pipeline auto-detects `interndb.sqlite` in the repo root; otherwise it uses
MySQL. Place a model under `models/` or set `V6_SLM_OVERRIDE`.

## Run on Colab (T4/L4)

Open `v6_colab.ipynb`, set a GPU runtime, and run the cells top to bottom: it
clones the repo, reads `interndb.sqlite` from Drive, retrains the brain, runs
the tests, then the text and voice benchmarks.

## Benchmarking

[benchmark.py](benchmark.py) runs the 20-query fixture set
([data/bench_queries.json](data/bench_queries.json)) through the live graph and
**mirrors the notebook's terminal routing exactly**, so it grades the same
answers the user sees/hears. It reports intent accuracy, SQL exec rate, latency
by category, a per-query spoken-vs-raw answer review (the number-rounding fix is
visible here), and — on GPU — a `.wav → STT` voice round-trip with WER and a
voice-vs-text **regression** list so STT noise is never mistaken for a pipeline
bug. See the *Benchmarking* section of [docs/architecture.md](docs/architecture.md).

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `V6_USE_SQLITE` | `0` | `1` for the SQLite backend (Colab) |
| `V6_SQLITE_PATH` | auto | path to the SQLite DB |
| `V6_SLM_SIZE` | `4b` | `3b` / `4b` / `7b` router+sqlgen model |
| `V6_SLM_OVERRIDE` | – | force a HuggingFace model id |
| `V6_BRAIN_SEUIL` | `0.5` | continue threshold (below → stop) |
| `V6_4BIT` | `0` | 4-bit NF4 quantization |
| `V6_TTS_SPEAKER_EN` / `_FR` | `Claribel Dervla` | XTTS-v2 voice (or `_WAV_*` to clone) |
| `V6_TTS_TEMPERATURE` / `_REPETITION_PENALTY` | `0.6` / `2.5` | XTTS quality knobs |
| `V6_SMTP_USER` / `_PASSWORD` / `_HOST` | – | only to actually send a drafted email |
| `LATENTMIND_MYSQL_*` | localhost/interndb | MySQL connection (local) |

## Module map

| File | Role |
|---|---|
| [graph.py](graph.py) | LangGraph assembly + `LatentMindV6.ask()` |
| [nodes.py](nodes.py) | the graph node functions + `route_after_brain` |
| [brain.py](brain.py) | the policy MLP — intent + action + continue (seuil) |
| [brain_data.py](brain_data.py) | synthetic trace synthesis (the editable policy spec) |
| [train_brain.py](train_brain.py) | brain MLP training |
| [orchestrator.py](orchestrator.py) | deterministic validation + plan assembly |
| [schema.py](schema.py) | live DB introspection + join map |
| [entities.py](entities.py) | wilaya / date / segment resolution |
| [knowledge.py](knowledge.py) | BGE-M3 encoder + RAG retriever |
| [slm.py](slm.py) | Qwen dual-role engine + KV-cache hand-off + Polisher + `lang_code` |
| [numfmt.py](numfmt.py) | deterministic number humanization (the figure trust boundary) |
| [prompts.py](prompts.py) | router / SQL prompts + parsing |
| [sql_tools.py](sql_tools.py) | SQL safety, consistency check, execution |
| [capabilities.py](capabilities.py) | chart / email-draft / report |
| [speech.py](speech.py) | STT + TTS + `speakable()` |
| [benchmark.py](benchmark.py) | text + voice benchmark harness |
| [test_v6.py](test_v6.py) | verification harness |
