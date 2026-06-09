# 03 — Architecture

> Code: `v6/graph.py`, `v6/nodes.py`, `v6/state.py`, `v6/slm.py`, `v6/config.py`.
> Module-by-module reference: [`v6/docs/architecture.md`](../v6/docs/architecture.md).

V6 is a **LangGraph state machine** with a **star topology**: the trained brain is
the hub, the capability actions are the spokes. The brain picks one action, the
action runs and records its outcome, and control returns to the brain to re-decide.

---

## 1. The four design principles

1. **The brain decides, everything else executes.** No routing `if`-statements in
   the graph — the trained MLP picks each action and judges when to stop. Policy
   lives in training traces, not code.
2. **Determinism at the trust boundary.** Every SLM output that reaches the DB or a
   figure is validated/frozen by deterministic code first.
3. **One model, two roles.** The Qwen SLM is both router (phase 1) and SQL
   generator (phase 2), sharing a KV cache.
4. **The policy is trainable data.** `brain_data.py` is the editable spec of
   behavior; add a trace, retrain, behavior changes — no graph code touched.

---

## 2. The star-topology graph

```
START
  │
  ▼
brain ─────────────────────────┐   every action loops back; the brain re-decides
  ├─► rag ──────────────────────┤
  ├─► sql ──────────────────────┤   sql = router → orchestrator-validate → generate → consistency-check → execute
  ├─► chart ────────────────────┤
  ├─► email ────────────────────┤
  ├─► template ─────────────────┘
  │
  └─► communicator → END        (fires when continue_score < BRAIN_SEUIL)
```

Every action node returns to `brain`. The brain re-evaluates with the new
**outcome** and either picks another action or drops below the seuil and exits to
the communicator. The graph is compiled with a **`MemorySaver` checkpointer**, so
conversation memory (recent turns, a compacted summary, the last result rows)
persists across turns per `thread_id` — this is how follow-ups work.

---

## 3. The shared state (`state.py`)

`AgentState` is a `TypedDict` every node reads and partially updates. Grouped by
stage:

- **Input:** `query`, `thread_id`.
- **Cross-turn memory** (NOT reset between turns): `turns` (last 2 raw turns with
  their tables/columns), `memory_summary` (compacted older turns), `last_rows` /
  `last_columns` (for follow-up reports), `carried_entities`.
- **Brain loop:** `brain_step`, `step_log` (one outcome dict per executed action),
  `intent` (decided at step 0, held all turn), `next_action`, `continue_score`,
  `brain_scores`.
- **Retrieval:** `knowledge` (the RAG block), `grounding` (max non-wilaya cosine).
- **SQL pipeline:** `router_raw`, `routing`, `feedback`, `entities`, `sql`,
  `sql_valid`, `sql_issues`, `rows`, `columns`, `exec_ok`.
- **Capability artifacts:** `chart_path`, `email_draft`, `document_path`.
- **Output:** `thoughts` (the streamed thinking feed), `final_answer`, `errors`,
  `trace`, `timings`.

`initial_state()` resets all per-turn fields but deliberately preserves cross-turn
memory.

---

## 4. The brain node + the seuil gate

**`brain_node`** calls `get_brain().decide(query, memory, step_log, grounding,
thread_id)`, locks intent on the first tick, appends a human-readable thought
("Let me check the reference knowledge first."), and returns the brain's outputs.
Full model detail in [The Brain](04-the-brain.md).

**`route_after_brain`** is the one place deterministic control flow is allowed — it
*gates* the brain's learned decision, it doesn't make one:

- Non-data intents (`greeting/meta/definition/unanswerable/off_topic`) →
  **communicator** directly (the action head is ignored — a guard so a misfiring
  action can't push a greeting into SQL).
- `brain_step ≥ BRAIN_MAX_STEPS` (8) → communicator (safety cap).
- `continue_score < BRAIN_SEUIL` (0.5) → communicator. **Gates all actions,
  including terminals** — an incidental "template" argmax after plain SQL gets
  continue ≈ 0.01 and is blocked.
- `action_conf < BRAIN_CONF_MIN` (0.35) → communicator.
- A terminal (chart/email/template) already attempted this turn → communicator
  (prevents retry loops).
- Otherwise → the chosen action.

---

## 5. The nodes

### `rag_node`

Calls the retriever's `knowledge_block(query)`, records the **grounding** score
(max cosine among non-wilaya chunks), and appends an outcome (`rag_weak` if
grounding < 0.45). See [RAG, Knowledge & Entities](05-rag-knowledge-entities.md).

### `sql_node` → `run_sql_pipeline` (the SQL chain)

1. **Build router messages** — system prompt with the live schema + LOCATION RULE +
   TIME RULE + routing rules; user turn with the query, knowledge, history.
2. **`slm.run_router`** — phase 1; stash the KV cache.
3. **`parse_router_output`** — parse the routing JSON.
4. Non-data → short-circuit, empty SQL.
5. **`orchestrator.assemble`** — validate tables/columns against the **live
   schema**, inject `dim_location` when a wilaya filter needs a join, rate
   confidence (high/medium/low). Low → return a clarification.
6. **`entities.resolve_all`** — resolve wilaya mentions to canonical French
   spellings (alias + accent-fold + fuzzy).
7. **Micro-retry loop** (1 + `SQL_MAX_RETRIES`): build the SQL-gen instruction →
   `slm.run_sqlgen` (phase 2, KV reused) → `clean_sql` → `validate_sql` (static) →
   `consistency_check` (hallucinated `alias.column`, inline ID lists, non-canonical
   wilayas) → on issues, append a `correction_hint` and retry; on pass,
   `enforce_limit` → `execute_sql`.
8. Compose `final_answer` from a **humanized** row summary (numbers already frozen
   by `numfmt`) or an appropriate error message.

### Capability nodes

- **`chart_node`** — `capabilities.make_chart` + a typed JSON spec (`chartspec.py`)
  the web client renders as an interactive Recharts chart; type chosen from data
  shape + executed SQL + question (ranking → sorted bar, trend → line, share →
  donut, single KPI → stat). Figures are pre-frozen, so the chart can't re-round. A
  matplotlib PNG is saved as a fallback.
- **`email_node`** — `compose_email_draft` resolves a recipient from the `contacts`
  table and fills a Jinja2 template. **Drafts only**; `send_email()` is a separate
  explicit action needing SMTP env vars. `status: "needs_recipient"` if nobody
  matched.
- **`template_node`** — `fill_report` fills a Jinja2 report; falls back to
  `last_rows`/`last_columns` for a cross-turn "put it in a report" request.

### `communicator_node` (terminal)

Composes the final answer by intent: `greeting`/`meta` → canned text; `definition`
→ `retriever.definition_for`; `unanswerable` and `off_topic` → **fixed,
language-matched deflections, never sent to the polisher**; otherwise → the
`final_answer` set by the SQL/capability nodes (already carrying humanized
figures), plus emoji notes for artifacts. It then **rolls conversation memory**:
keep the last 2 raw turns, compact older ones deterministically (capped 600 chars,
no LLM), and persist `last_rows`/`last_columns`.

---

## 6. The SLM engine (`slm.py`)

**`DualRoleSLM`** — one loaded model (default **Qwen3-4B-Instruct-2507**), two
phases, shared KV cache:

- **`run_router`** (phase 1) — applies the chat template (`enable_thinking=False`
  for Qwen3), generates with `past_key_values` returned, stores them per
  `thread_id`.
- **`run_sqlgen`** (phase 2) — tries **KV reuse** first (`_sqlgen_kv`: splice a new
  user turn onto phase-1's sequence + cache, continue), falls back to a plain
  re-encode (`_sqlgen_plain`) if the cache is missing/corrupt. Identical output,
  slower fallback. KV reuse saves ~1.5 s/query.
- Optional **speculative decoding** with a small drafter (Qwen3-0.6B) and optional
  **grammar-constrained SQL** via `lm-format-enforcer` (`USE_CONSTRAINED_SQL`).
- **`lang_code` / `detect_lang`** — the single source of truth for language
  (`ar`/`fr`/`en`); the persona, the off-topic deflection, and the TTS voice all
  delegate here, so spoken language always matches written.

**The Polisher** — a natural-language refiner with **four roles**:

| Role | Trigger | Job |
|---|---|---|
| `analyze` | SQL rows returned | wrap **already-formatted** figures in a sentence — copy numbers **verbatim** |
| `polish` | RAG / definition | rewrite into natural prose |
| `clarify` | error / missing info | explain what went wrong, ask what's needed |
| `chat` | greeting / thanks / meta | measured reply in the question's language |

By default **the polisher *is* the 4B** (`POLISHER_USE_MAIN=1`): roles run through
the already-loaded Qwen3-4B (streamed + speculative drafter) instead of a separate
1.5B — it obeys "reply in French," keeps KPI names ("churn" never becomes
"chômage"), and won't drift off-scope, at negative VRAM cost. `unanswerable` and
`off_topic` are never polished. **The analyst never formats numbers** — by the time
a figure reaches it, `numfmt` has frozen it; the prompt's first rule is *copy each
figure exactly.* (See [Data, Schema & Numbers](06-data-schema-numbers.md).)

---

## 7. A complete query trace

Query: *"Montre-moi la marge brute pour Oum El Bouaghi le trimestre dernier"*

```
brain (tick 0): outcome all-zero → intent=data, action=rag, continue=0.99
rag: top-5 chunks (gross_margin def + Oum El Bouaghi wilaya chunk); grounding=0.52
brain (tick 1): last=rag,ok → action=sql, continue=0.98
sql:
  router → {tables:[fpa_profitability, dim_location], columns:[gross_margin,...],
            filters:{wilayas:[Oum El Bouaghi], period:Q3 2025}}
  orchestrator.assemble → confidence=high, dim_location injected
  entities.resolve_all → Oum El Bouaghi → 29 location_ids
  run_sqlgen (KV reused) → SELECT AVG(gross_margin) ... WHERE location_id IN (subquery)
  validate + consistency_check → clean ; enforce_limit ; execute → {avg_gross_margin: 42.3903}
  _summarize_rows(lang=fr) → final_answer = "avg_gross_margin: 42,39 %"   (frozen)
brain (tick 2): last=sql,ok,row=one → continue=0.01 < seuil → communicator
communicator: final_answer already set, roll memory
→ Polisher (analyze) copies the frozen figure:
   "La marge brute moyenne pour Oum El Bouaghi le trimestre dernier était de 42,39 %."
→ TTS speakable("…42,39 %", fr) → "…était de 42,39 pour cent." (spoken)
```

The brain fired exactly the actions it needed, stopped the instant it had the
answer, and the number reaching the user is the *same frozen string* in text,
chart, and voice.

---

## 8. Configuration (`config.py`)

Env-driven (`V6_` prefix, falls back to `V5_`). The ones that change behavior most:

| Knob | Default | Effect |
|---|---|---|
| `V6_SLM_SIZE` | `4b` | `3b`/`4b`/`7b` router+SQL model |
| `V6_BRAIN_SEUIL` | `0.5` | continue threshold (below → stop) |
| `V6_BRAIN_MAX_STEPS` | `8` | loop safety cap |
| `V6_4BIT` | `0` | NF4 quantization (7B on a 16 GB T4) |
| `V6_POLISHER_USE_MAIN` | `1` | polish via the loaded 4B vs a separate 1.5B |
| `V6_USE_SQLITE` | `0` | SQLite (Colab) vs MySQL (local) |
| `V6_REFERENCE_DATE` | wall clock | pin "today" for relative periods |
| `V6_TTS_SPEAKER_WAV_*` | – | clone a reference WAV (biggest voice lever) |

`config.py` also resolves model IDs (local cache vs Hub), the speculative drafter,
output paths (Drive on Colab, repo root locally), and device quirks (the encoder is
forced to CPU on Apple Silicon).

→ Next: [The Brain](04-the-brain.md).
