# LatentMind V6 — Full Architecture Documentation

## What V6 Is

LatentMind V6 is an **agentic telecom analytics system** built for the Algerian market (Djezzy). A user asks a question in French, English, or Arabic/Darija — "what was the ARPU in Oran last quarter?" — and the system:

1. Retrieves relevant domain knowledge (RAG),
2. Routes the query to the right database table,
3. Generates and executes a SQL query,
4. Optionally charts the result, drafts an email, or generates a report,
5. Writes a natural-language answer.

Every step is decided dynamically by a trained **brain MLP** that watches what happens and re-decides until it is confident the turn is done. This replaces the old one-shot planner that committed to a fixed plan before execution started and could not react to failures.

---

## Core Design Principles

**1. The brain decides, everything else executes.**
No hardcoded routing logic in the graph. The trained MLP picks each action and judges when to stop. Heuristics that belong in "policy" live in training traces, not in Python `if` statements.

**2. Determinism at the trust boundary.**
The SLM is probabilistic — it can hallucinate columns, wilaya names, or SQL syntax. Every SLM output is post-processed by deterministic validators (`sql_tools.py`, `orchestrator.py`, `entities.py`, `schema.py`) that catch and reject bad outputs before they reach the database. **Numbers are part of this boundary**: figures are rounded and unit-tagged in plain Python (`numfmt.py`) *before* they reach the 1.5B polisher, so the model never reformats a raw value and cannot corrupt it. The polisher copies a frozen figure; it never computes one.

**3. One model, two roles.**
The Qwen SLM plays both the router (phase 1) and the SQL generator (phase 2) in the same conversation, sharing a KV cache. Phase 2 does not re-encode phase 1's context — it inherits it. This is the LatentMAS (Multi-Agent System) core: latent communication through attention state.

**4. The policy is trainable.**
The brain traces in `brain_data.py` are the readable, editable specification of what the system should do. Add a trace pattern, retrain, the brain learns it. No code change required.

---

## Repository Layout

```
v6/
├── config.py          — all tunables, env-driven, one class
├── state.py           — LangGraph AgentState TypedDict
├── graph.py           — graph topology + LatentMindV6 public API
│
├── brain.py           — BrainHead MLP + Brain.decide()
├── brain_data.py      — synthetic trace synthesis (the policy spec)
├── train_brain.py     — MLP training script
│
├── nodes.py           — all node functions + route_after_brain
├── slm.py             — DualRoleSLM (router + sqlgen) + Polisher + lang_code
├── numfmt.py          — deterministic number humanization (the trust boundary for figures)
├── knowledge.py       — BGE-M3 encoder + Retriever
├── orchestrator.py    — schema validation + plan assembly
├── entities.py        — wilaya name resolution
├── schema.py          — live DB introspection
├── sql_tools.py       — SQL validation + execution
├── capabilities.py    — chart, email, report generation
├── speech.py          — STT (faster-whisper) + TTS (XTTS-v2) + speakable()
├── server.py          — FastAPI + WebSocket API (streams text + cloned audio)
├── benchmark.py       — text + voice benchmark harness
├── prompts.py         — all prompt templates
│
├── data/
│   ├── brain_train.jsonl       — synthesized training rows
│   ├── glossary.json           — KPI definitions, business context
│   ├── kpi_catalog.json        — multilingual KPI synonyms
│   ├── planner_prototypes.json — query prototypes for trace synthesis
│   └── wilaya_aliases.json     — canonical → alias mappings
│
├── templates/
│   ├── report.md.j2            — Jinja2 report template
│   └── email_report.md.j2     — Jinja2 email template
│
├── v6_colab.ipynb     — Colab training + demo notebook
├── test_v6.py         — integration tests
└── requirements.txt
```

---

## The Graph Topology

```
START
  │
  ▼
brain ──────────────────────┐
  │                         │ (loop back after each action)
  ├─→ rag ──────────────────┤
  ├─→ sql ──────────────────┤
  ├─→ chart ────────────────┤
  ├─→ email ────────────────┤
  ├─→ template ─────────────┘
  │
  └─→ communicator → END
```

Every action returns to `brain`. The brain re-evaluates with the new outcome. When `continue_score < BRAIN_SEUIL` (default 0.5), the loop exits to `communicator`. This is the **star topology**: brain is the hub, actions are the spokes.

The graph is compiled with a `MemorySaver` checkpointer. Conversation memory (turns, memory_summary, last_rows) persists across turns per `thread_id`.

---

## Module-by-Module Reference

---

### `config.py` — Configuration

**Purpose:** Central configuration class. Every tunable is read from the environment with a `V6_` prefix (falls back to `V5_`), so no config file needs to be edited between environments.

**Key attributes:**

| Attribute | Default | Meaning |
|---|---|---|
| `SLM_SIZE` | `"4b"` | Model size: `"3b"`, `"4b"`, `"7b"` |
| `BRAIN_SEUIL` | `0.5` | Continue threshold — below this, the loop stops |
| `BRAIN_CONF_MIN` | `0.35` | Minimum action confidence to proceed |
| `BRAIN_MAX_STEPS` | `8` | Loop safety cap |
| `SQL_MAX_RETRIES` | `1` | Micro-retry attempts per SQL generation |
| `SQL_MAX_ROWS` | `1000` | LIMIT injected if the model omits one |
| `RAG_TOP_K` | `5` | Top-k chunks returned by the retriever |
| `RAG_LOW_CONF` | `0.45` | Grounding threshold below which RAG is considered weak |
| `EMBED_DIM` | `1024` | BGE-M3 embedding dimension |
| `USE_4BIT` | `False` | NF4 quantization (needed for 7B on 16GB T4) |
| `USE_SPECULATIVE` | `True` | Speculative decoding with a small drafter model |
| `USE_CONSTRAINED_SQL` | `False` | Grammar-constrained SQL generation via lm-format-enforcer |

**Key methods:**

- `slm_id()` — resolves to a local model directory or HuggingFace Hub ID, checking for locally cached models first.
- `draft_slm_id(main_id)` — returns the speculative decoding drafter for a given main model. Qwen3 → Qwen3-0.6B; Qwen2.5-Coder → Qwen2.5-Coder-0.5B.
- `bge_m3_id()` — returns local path if cached, else Hub ID `BAAI/bge-m3`.
- `output_dir()`, `chart_dir()`, `report_dir()` — resolve output paths (Google Drive on Colab, repo root locally).
- `device()` — returns `cuda`, `mps`, or `cpu`.
- `encoder_device()` — always returns `cpu` on Apple Silicon because BGE-M3 returns wrong embeddings for batch-of-1 encodes on MPS.

---

### `state.py` — The LangGraph State

**Purpose:** Defines `AgentState`, the TypedDict shared by all nodes. Every node receives it and returns a partial update. `total=False` means no key is required.

**Fields grouped by stage:**

**Input:**
- `query` — the user's raw question
- `thread_id` — conversation identifier (default `"default"`)

**Cross-turn memory** (persisted by the checkpointer, NOT reset between turns):
- `turns` — last 2 raw turns (Q+A+tables+columns)
- `memory_summary` — compacted text summary of older turns
- `last_rows`, `last_columns` — rows from the previous data turn (used by `template_node` for cross-turn reports)
- `carried_entities` — entities resolved on the previous turn

**Brain loop:**
- `brain_step` — loop iteration counter (0-indexed, guards against BRAIN_MAX_STEPS)
- `step_log` — one outcome dict per executed action: `{action, ok, error_type, row_bucket, attempt}`
- `intent` — decided once at step 0, held for the whole turn: `greeting|meta|definition|data|unanswerable|off_topic`
- `next_action` — action the brain chose on this tick
- `continue_score` — the seuil signal [0, 1]
- `brain_scores` — full probability distributions for debugging

**Retrieval:**
- `knowledge` — formatted RAG context block (injected into the router prompt)
- `grounding` — max cosine score among non-wilaya RAG chunks

**SQL pipeline:**
- `router_raw` — raw text output from the router SLM
- `routing` — parsed and schema-validated routing object: `{intent, tables, columns, filters, ...}`
- `feedback` — failure note carried into an SQL macro-retry
- `entities` — resolved wilayas: `{wilayas, wilaya_ids_map, unresolved_wilayas}`
- `sql` — the final SQL string (with LIMIT injected)
- `sql_valid`, `sql_issues` — validation results
- `rows`, `columns` — query results
- `exec_ok` — True if the query ran successfully and returned rows

**Capability artifacts:**
- `chart_path` — path to the saved PNG chart
- `email_draft` — `{to, to_name, subject, body, status: "draft"|"needs_recipient"}`
- `document_path` — path to the saved Markdown report

**Output:**
- `thoughts` — streamed thinking feed: `[{kind: "thinking"|"answer", text}]`
- `final_answer` — the answer string passed to the communicator/polisher
- `errors` — accumulated error strings
- `trace` — human-readable step log for debugging
- `timings` — `{node_name_ms: float}` performance breakdown

**`initial_state(query, thread_id)`** resets all per-turn fields. Deliberately does NOT reset cross-turn memory fields so follow-ups work.

---

### `brain.py` — The Policy MLP

**Purpose:** The decision-making heart of the loop. Called once per loop tick. Given the current situation (query + memory + what happened so far), predicts what to do next and whether to keep going.

**The situation vector (2073-d):**
```
query_emb (1024) ⊕ memory_emb (1024) ⊕ outcome_vec (25)
```
- `query_emb` and `memory_emb` are BGE-M3 embeddings. Cached at step 0 — only encoded once per turn, reused on every tick.
- `outcome_vec` is rebuilt every tick from `step_log`.

**The outcome vector (25-d):**

Encodes everything that happened so far this turn into a fixed 25-dimensional feature vector:

| Slice | Size | Meaning |
|---|---|---|
| last-action one-hot | 6 | `[none, rag, sql, chart, email, template]` |
| last-ok | 1 | 1 if the last action succeeded |
| error-type one-hot | 7 | `none, sql_error, sql_no_rows, sql_no_query, email_no_recipient, artifact_failed, rag_weak` |
| row-bucket one-hot | 4 | `none, zero, one, many` |
| attempt-count (normalized) | 1 | `min(attempts, 3) / 3.0` |
| grounding score | 1 | RAG cosine score [0, 1] |
| done-actions multi-hot | 5 | `[rag, sql, chart, email, template]` already executed |

**`BrainHead` (nn.Module):**
```
Input (2073) → Linear(256) → ReLU → Dropout(0.1) → trunk (256)
trunk → Linear(6)   = intent logits  (softmax → 6 probabilities)
trunk → Linear(5)   = action logits  (softmax → 5 probabilities)
trunk → Linear(1)   = continue logit (sigmoid → [0, 1])
```
The intent head sizes itself from `len(INTENTS)` in `brain.py`; adding an intent (as `off_topic` was) auto-resizes the layer and requires a retrain.

**`Brain.decide()`:**
1. Embeds query + memory (from cache if already done this turn).
2. Calls `encode_outcome(step_log, grounding)` to build the outcome vector.
3. Concatenates all three → 2073-d situation vector.
4. Forward pass through `BrainHead`.
5. Returns `BrainDecision(intent, action, action_conf, continue_score, ...)`.

**Why no fallback if `brain_head.pt` is missing:** the brain is a trained model. An untrained head would give random outputs. The system fails loudly with instructions on how to train it rather than silently running wrong.

---

### `brain_data.py` — Synthetic Policy Traces

**Purpose:** Generates the training dataset for the brain. This file IS the editable specification of what the system should do. Change a trace template here, retrain, the brain learns the new behaviour.

**How it works:**
- Each trace defines an `intent`, a template query, and a `gold` sequence of `(action, outcome)` pairs.
- `_expand(trace)` generates ALL ticks: tick 0 has an empty `step_log` (brain sees nothing done yet), tick 1 has one entry (brain sees the result of the first action), etc.
- The final tick of every trace has `label_continue=0` — this is where the brain must fire the seuil and stop.
- `_terminal(trace)` adds extra rows concentrated on the stopping state after a terminal action (chart/email/template) to strengthen the stop signal.

**Trace families:**
- `greeting` → communicator only (no actions)
- `meta` → communicator only
- `definition` → communicator only
- `unanswerable` → communicator only
- `off_topic` → communicator only (code/translation/trivia/advice — a DETERMINISTIC, language-matched deflection; never sent to the polisher, so a small model can't be coaxed into actually doing it)
- `data_only` → rag → sql → communicator
- `data_chart` → rag → sql → chart → communicator
- `data_email` → rag → sql → email → communicator
- `data_template` → rag → sql → template → communicator
- `data_chart_email` → rag → sql → chart → email → communicator
- `sql_retry_ok` → rag → sql(fail) → sql(ok) → communicator
- `sql_fail_twice` → rag → sql(fail) → sql(fail) → communicator
- `email_no_recipient` → rag → sql → email(no_recipient) → communicator
- `followup` → sql → communicator (no rag, inherits context)

**Output:** `v6/data/brain_train.jsonl`, one JSON object per training row.

---

### `train_brain.py` — Brain Training

**Purpose:** Loads the synthesized traces, featurizes them with BGE-M3, and trains the `BrainHead` MLP with three masked losses.

**Training pipeline:**
1. Read `brain_train.jsonl`.
2. Encode all unique `(query, history)` pairs with BGE-M3 (one-time, batched).
3. For each row, build the full 2073-d situation vector from embeddings + `encode_outcome(step_log, grounding)`.
4. Train with 3 losses:
   - **CE on intent** — only on step-0 rows (intent is decided once per turn).
   - **CE on action** — only on rows with `label_continue=1` (the brain only needs to pick an action when it's going to continue).
   - **BCE on continue** — on all rows (the seuil must fire correctly at every step).
5. Save to `models/brain_head.pt`.

**Default:** 200 epochs, Adam optimizer, batch size 64. Val accuracies are printed every 40 epochs. On a T4 GPU with 8k rows, this takes about 3-4 minutes.

---

### `nodes.py` — The Node Functions

**Purpose:** Implements every node in the LangGraph graph. Each function receives the full `AgentState` dict and returns a partial update.

**`brain_node(state)`**
- Calls `get_brain().decide(query, memory, step_log, grounding, thread_id)`.
- Intent is locked on the first tick: `intent = state.get("intent") or decision.intent`.
- Appends a thought string ("Let me check the reference knowledge first.", "I have what I need — writing the answer now.", etc.) to `thoughts`.
- Returns: `brain_step`, `intent`, `next_action`, `continue_score`, `brain_scores`, `thoughts`, `trace`, `timings`.

**`route_after_brain(state)`**
- The seuil gate. Returns the next node name.
- Non-data intents (`greeting`, `meta`, `definition`, `unanswerable`, `off_topic`) → `communicator` directly. The action head's pick is ignored for these — this guard stops a misfiring action head from sending a greeting or off-topic query into the SQL pipeline.
- `brain_step >= BRAIN_MAX_STEPS` → `communicator`.
- `action not in ACTIONS` → `communicator`.
- `continue_score < BRAIN_SEUIL` → `communicator`. (This gates ALL actions — including terminals. An incidental terminal pick after plain SQL gets continue≈0.01 and is correctly blocked.)
- `action_conf < BRAIN_CONF_MIN` → `communicator`.
- Terminal actions (`chart`, `email`, `template`) that have already been attempted this turn → `communicator`. (Prevents retry loops on failure.)
- Otherwise: returns the action name.

**`rag_node(state)`**
- Calls `get_retriever().knowledge_block(query)`.
- Records `grounding` (max cosine among non-wilaya chunks).
- Appends outcome to `step_log`: `{action: "rag", ok: True, error_type: "rag_weak" if grounding < 0.45 else "none", ...}`.

**`run_sql_pipeline(state)`** (helper, called by `sql_node`)
The full SQL chain:
1. `build_router_messages(...)` → construct the router prompt.
2. `slm.run_router(messages, thread_id)` → run phase 1, stash KV cache.
3. `parse_router_output(rr)` → parse the routing JSON.
4. If router says intent is non-data → short-circuit, return empty SQL.
5. `orchestrator.assemble(...)` → validate tables/columns against live schema, inject `dim_location`, check confidence.
6. If confidence is `"low"` → return a clarification message.
7. `entities.resolve_all(...)` → resolve wilaya names to canonical French spellings.
8. Micro-retry loop (1 + SQL_MAX_RETRIES attempts):
   - `build_sqlgen_instruction(...)` → construct the SQL-gen prompt.
   - `slm.run_sqlgen(thread_id, instruction)` → run phase 2 (reuses KV cache).
   - `clean_sql(output)` → strip markdown fences, extract the SELECT.
   - `validate_sql(sql, schema)` → static validation (read-only, no blocked keywords, tables exist).
   - `consistency_check(sql, entities, query, schema)` → hallucination checks (alias.column, inline id lists, non-canonical wilaya names).
   - If any issue → `correction_hint(issues, entities, exec_error)` appended to the retry prompt.
   - If static checks pass → `enforce_limit(sql)` → `execute_sql(sql)`.
   - If execution succeeds → break.
9. Compose `final_answer` from the row summary or an appropriate error message.

**`sql_node(state)`**
Calls `run_sql_pipeline`, records the outcome entry, appends to `step_log`.

**`chart_node(state)`**
Calls `capabilities.make_chart(rows, cols, query)`. Records `{ok, error_type: "artifact_failed" if failed}`.

**`email_node(state)`**
Calls `capabilities.compose_email_draft(query, final_answer, rows, cols)`. The draft is always written; `status: "needs_recipient"` if no contact matched.

**`template_node(state)`**
Calls `capabilities.fill_report(query, rows, cols, final_answer, entities)`. Falls back to `last_rows`/`last_columns` from state when no SQL ran this turn (cross-turn report request like "put it in a report").

**`communicator_node(state)`**
Terminal node. Composes the final answer:
- `greeting` → canned greeting text.
- `meta` → canned capability description.
- `definition` → `retriever.definition_for(query)`.
- `unanswerable` → canned "that metric isn't in the database" text.
- `off_topic` → fixed, language-matched deflection (`_OFFTOPIC_TEXT[lang_code(query)]`) — never sent to the polisher.
- Otherwise: uses whatever `final_answer` the SQL/capability nodes set (already carrying humanized figures from `numfmt`); adds emoji notes for chart/report/email artifacts.
Rolls conversation memory: keeps last 2 raw turns, compacts older ones with `_compact_turns()`. Persists `last_rows`/`last_columns` for follow-up reports.

**`_history_text(turns, memory_summary)`**
Builds the conversation context string injected into prompts and the brain's memory embedding. For each prior data turn it includes the tables and columns the router chose — this lets short follow-ups ("and for Tiaret?") inherit the right KPI without relying on text keywords.

**`_compact_turns(turns)`**
Deterministic compression without LLM overhead. Extracts `Q: ... A: ...` snippets for data turns, `Q: ... [definition]` for definitions. Capped at 600 characters.

---

### `slm.py` — The Dual-Role Language Model

**Purpose:** One loaded model, two phases. The router and the SQL generator share weights; phase 2 inherits phase 1's KV cache instead of re-encoding.

**`DualRoleSLM`**

Constructor loads:
- Main model (`Qwen3-4B-Instruct-2507` by default).
- Optionally a speculative decoding drafter (`Qwen3-0.6B` for Qwen3 family). If vocab sizes differ, both tokenizers are passed.
- Resolves `<|im_end|>` token id for the KV-cache splice.

**`run_router(messages, thread_id)`** — Phase 1
- Applies chat template (with `enable_thinking=False` for Qwen3 to disable reasoning mode).
- Generates with `return_dict_in_generate=True, use_cache=True` so `past_key_values` are returned.
- Stores `{router_output, _seq1 (sequences tensor), _cache (past_key_values), _messages}` in `_store[thread_id]`.

**`run_sqlgen(thread_id, instruction)`** — Phase 2
Tries KV-cache reuse first (`_sqlgen_kv`). Falls back to plain re-encode (`_sqlgen_plain`) if the cache is missing or corrupted.

`_sqlgen_kv`: Splices a new user turn onto the phase-1 sequence tensor, appends the `<|im_start|>assistant\n` prompt, feeds the combined sequence with `past_key_values=cache` into `model.generate`. The model continues from the phase-1 state — no re-encoding.

`_sqlgen_plain`: Reconstructs the full conversation as a message list (router_messages + assistant router_output + new user instruction) and encodes from scratch. Slower but identical output.

If `USE_CONSTRAINED_SQL=True`, `_build_sql_logits_processor` builds a `lm-format-enforcer` regex processor that masks non-SQL tokens at every generation step.

**`lang_code(text)` / `detect_lang(text)`** — the single source of truth for language selection. `lang_code` returns `'ar' | 'fr' | 'en'` from Arabic script, French strong/weak markers, else English; `detect_lang` returns the human-readable name for prompting. The chat persona, the off-topic deflection, and the TTS voice (`speech.language_for`) all delegate here, so the spoken language always matches the written one. The explicit `Reply ONLY in <lang>.` line the chat role prepends stops a small model from mirroring the (possibly French) conversation memory instead of the current question.

**`Polisher`** — 4-role natural-language refiner (Qwen2.5-1.5B-Instruct)

Four roles, four system prompts:

| Role | Trigger | Job |
|---|---|---|
| `analyze` | SQL data rows returned | Wrap **already-formatted** figures in an insightful sentence — copy numbers verbatim |
| `polish` | RAG / definition | Rewrite into natural prose (teach, don't transcribe) |
| `clarify` | Error or missing info | Explain what went wrong, ask what's needed |
| `chat` | greeting / thanks / small talk / meta | Professional, measured reply in the question's language |

**The analyst does NOT format numbers.** By the time a figure reaches the `analyze` role it is already rounded, scaled, and unit-tagged by `numfmt` (e.g. "253.4 million DZD"). The prompt's first rule is that figures are FINAL: copy each one exactly, never re-round, never do arithmetic, never use "$". This is the fix for a 1.5B model corrupting "1,087,355,290.78" into "52,590,189,81" — the raw number it could mangle no longer exists in its input.

**By default the polisher IS the 4B** (`POLISHER_USE_MAIN=1`): the roles run through the already-loaded `Qwen3-4B` SLM (`stream_generate`, greedy + speculative drafter) instead of a separate 1.5B. The 4B reliably obeys "reply in French", keeps KPI names (so "churn" never becomes "chômage"), and won't drift off-scope — fixing the language/quality bugs a 1.5B kept producing, at no extra VRAM (it frees ~1.5 GB). Every role's user message now names the target language explicitly (from the *current* question, not the memory). `unanswerable` is no longer polished at all — like `off_topic` it returns a fixed, language-aware deflection (`nodes._UNANSWERABLE_TEXT`), because a free rewrite leaked scope ("to find the stock price, check financial news…"). Set `V6_POLISHER_USE_MAIN=0` to fall back to the standalone 1.5B.

`stream(raw_answer, question, role, memory)`: yields polished tokens one-by-one via `TextIteratorStreamer`. Uses `do_sample=True, temperature=0.5, top_p=0.9` for natural variation. `memory` is only used by the `chat` role.

`complete(system, user)`: blocking single-shot completion used for recipient resolution.

---

### `numfmt.py` — Number Humanization (the figure trust boundary)

**Purpose:** Render every numeric value as a clean, rounded, speech-ready string *before* it reaches the polisher or the screen. This is why an analytics assistant built on a 1.5B refiner can still be trusted with numbers: the model never formats a figure, it only copies one.

**`unit_for_column(col)`** — infers a unit (`DZD | % | count | GB | min | months | days`) for a possibly aggregated/aliased SQL column. Resolution order: exact hit in `kpi_catalog.json` → hit after stripping an aggregation prefix (`avg_`, `total_`, `sum_`, …) → keyword heuristic (`*rate|margin|ratio|share` → `%`; `revenue|income|opex|arpu|…` → `DZD`; `subscriber|adds|base` → `count`). Returns `None` if nothing matches.

**`humanize(value, unit, lang)`** — the core formatter. Locale-aware (`en`/`fr`/`ar`), with a decimal comma and French scale words for `fr`:

| Input | unit | `en` | `fr` |
|---|---|---|---|
| 1,087,355,290.78 | DZD | `1.09 billion DZD` | `1,09 milliard DZD` |
| 253,387,711.02 | DZD | `253.4 million DZD` | `253,4 millions DZD` |
| 42.4247 | % | `42.42%` | `42,42 %` |
| 3.071 | DZD | `3.07 DZD` | `3,07 DZD` |
| `None` | any | `—` | `—` |

Thresholds: `≥1e9` → billions (2 dp), `≥1e6` → millions (1 dp), `≥1e3` → grouped whole number, `<1e3` → up to 2 dp. Percentages keep 2 dp and the symbol. Non-numeric values (dates, wilaya names) pass through untouched.

**`humanize_cell(col, value, lang)`** — `humanize(value, unit_for_column(col), lang)`. Called by `nodes._summarize_rows`, which formats every cell in the query's language so the frozen figure is already in the right locale.

Run `python3 -m v6.numfmt` to see the full formatting table for both languages.

---

### `knowledge.py` — RAG Encoder and Retriever

**Purpose:** In-memory vector store over four knowledge sources. Grounds the router SLM and catches unanswerable queries.

**`BGEM3Encoder`**
- Loads `BAAI/bge-m3` (1.5B XLM-RoBERTa-based model, 1024-d CLS embeddings).
- Always runs on CPU on Apple Silicon (MPS gives wrong embeddings for batch-of-1).
- L2-normalizes all outputs — cosine similarity becomes a dot product.

**The four knowledge sources:**

1. **`data_catalog` table** — the database documenting its own columns. Each row: `(table_name, column_name, description)`. Gives the SLM authoritative descriptions of every column.

2. **`dim_location` + `wilaya_aliases.json`** — one chunk per wilaya. Contains: canonical French name, wilaya code (Algeria's official numeric), aliases, commune count, and the **correct subquery pattern** to use in SQL:
   ```
   WHERE <table>.location_id IN (
       SELECT location_id FROM dim_location WHERE wilaya = 'Oum El Bouaghi'
   )
   ```
   Teaching this pattern in the knowledge base is what prevents the SLM from hardcoding `location_id IN (1, 2, 3, ...)`.

3. **`kpi_catalog.json`** — per KPI: canonical column name, table, segment, unit, description, and multilingual synonyms. Example: "net income" → `fpa_profitability.net_income`, also called "revenu net", "bénéfice net", "صافي الربح".

4. **`glossary.json`** — definitions, business context (what ARPU means in a telecom context), and table relationship rules (how metric tables join to `dim_location`).

**`Retriever.knowledge_block(query)`**
- Encodes the query with BGE-M3.
- Returns top-k chunks by cosine similarity.
- Grounding score = max cosine among non-wilaya chunks. Wilaya chunks are excluded because a user saying "Oran" has found a location, not evidence the database can answer the KPI.

**`Retriever.definition_for(query)`**
- Returns the best `definition` or `kpi` chunk for a "what does X mean" question.
- Used by `communicator_node` for definition intent.

---

### `orchestrator.py` — Deterministic Schema Validation

**Purpose:** Fact-checks the router SLM's table and column picks against the live schema. Pure logic — no probabilistic calls.

**`assemble(query, routing, capabilities, followup, grounding, turns, schema)`**
1. **Validate tables**: drop any table the SLM named that doesn't exist in the schema.
2. **Validate columns**: drop any column that doesn't exist in ANY table.
3. **Structural rescue**: if no valid metric table came back, look up the last data turn and inherit its tables+columns. This handles short follow-ups where the SLM returned nothing because the query has no KPI keywords.
4. **dim_location injection**: if any wilaya filter is present and any metric table has `location_id`, add `dim_location` to the table list.
5. **Confidence rating**:
   - `"high"` — router gave a metric table AND a KPI column AND grounding ≥ 0.45.
   - `"medium"` — grounding is above the floor, or tables were inherited.
   - `"low"` — no metric table and low grounding → send to clarify.

---

### `entities.py` — Wilaya Name Resolution

**Purpose:** Maps free-text entity mentions onto the canonical French spellings that exist in `dim_location`.

**The problem:** Users write "Algiers", "الجزائر", "Alger" (correct), "alger", "ALGER", "Bejaia" (missing accent), "M'Sila" (apostrophe variant). The database stores "Alger", "Béjaïa", "M'Sila". Resolution bridges that gap.

**`_norm(s)`** — normalizes a string for comparison: lowercases, strips accents (NFKD decomposition), removes apostrophes/dashes/punctuation, collapses spaces. `"Oum El Bouaghi"` and `"oum el bouaghi"` become the same key.

**`Resolver`**
- Loads all `(wilaya, location_id)` pairs from `dim_location` at init.
- Builds `_index: {normalized_form → canonical_name}`.
- Loads `wilaya_aliases.json` and adds alias → canonical mappings to the index.
- `max_words` = max n-gram length among all known wilaya names (for greedy longest-match scanning).

**`resolve_wilaya(name)`** — exact normalized lookup first; fuzzy match (difflib, cutoff 0.86) as fallback.

**`scan_query(query)`** — finds wilaya names directly in free text. Used as a backup to the router's extraction. Greedy longest-match (matches "Tizi Ouzou" before "Tizi").

**`resolve_all(query, router_filters, max_date)`** — combines router-supplied names + direct query scan. Returns `{wilayas, wilaya_ids_map, unresolved_wilayas}`.

---

### `schema.py` — Live Database Introspection

**Purpose:** Reads the database's own structure at startup. Never hardcodes table/column names.

**`DBSchema`**
- `_introspect_sqlite` / `_introspect_mysql` — reads all table/column/type info.
- `_load_descriptions` — reads the `data_catalog` table for human column descriptions.
- `_find_date_range` — reads `MIN(week_start)` and `MAX(week_start)` from the main metric tables so the SQL-gen SLM knows what date range is actually available.
- **Join map**: any table (other than `dim_location`) that has a `location_id` column is added to `join_map`. This is the structural fix for the old hardcoded assumption that metric tables had a `wilaya` column directly.

**`needs_location_join(table)`** — returns True if a `dim_location` JOIN is needed for wilaya filtering on this table.

**`prompt()`** — builds the schema block injected into the router SLM's prompt. Includes table+column lists, the LOCATION RULE (filter via `location_id IN (subquery)`), and the TIME RULE (`week_start` is the date column).

---

### `sql_tools.py` — The SQL Trust Boundary

**Purpose:** Everything that makes raw SLM SQL output safe to run. The SLM writes text; this module decides whether it may execute.

**`clean_sql(raw)`** — strips markdown fences, extracts the first SQL statement, removes prose.

**`validate_sql(sql, schema)`** — static validation:
- Must be a `SELECT` or `WITH` statement.
- Must not contain any blocked DDL/DML keyword (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, ...).
- Referenced tables must exist in the schema.

**`consistency_check(sql, entities, query, schema)`** — three semantic checks:
1. **Hallucinated columns**: for every `alias.column` reference in the SQL, the column must exist in the table the alias points to. Catches `f.wilaya` when `fpa_profitability` has no `wilaya` column.
2. **Inline location_id lists**: if the SQL contains `location_id IN (1, 2, 3, ...)` instead of the required subquery, flag it with a correction hint.
3. **Non-canonical wilaya names**: if the SQL uses `wilaya = 'Algiers'` but the DB stores `wilaya = 'Alger'`, flag it with the canonical spelling.

**`correction_hint(issues, entities, exec_error)`** — builds the targeted correction text appended to the retry prompt. If a database error is present, surfaces it verbatim (e.g. "no such column: total_revenue") so the SLM can reason from the actual error.

**`enforce_limit(sql)`** — appends `LIMIT 1000` if the query has no LIMIT clause.

**`execute_sql(sql)`** — runs the query, coerces Decimal/date/bytes to JSON-safe types, never raises. Returns `{ok, rows, columns, error}`.

---

### `capabilities.py` — Chart, Email, Report

**`make_chart(rows, cols, query)`**
- Detects the date column by value shape (ISO date strings), not by name.
- Chooses chart type automatically:
  - Date column + multiple rows + category column → multi-series line chart (one line per category).
  - Date column + multiple rows + no category → single-series line chart.
  - No date column + category column → bar chart.
  - Single row → bar chart over metrics.
- Saves to `V6Config.chart_dir()` with a timestamp filename.

**`compose_email_draft(query, answer, rows, cols)`**
- Loads contacts from the `contacts` database table.
- Calls `resolve_recipient(query, contacts)` which uses the Polisher SLM (non-streaming `complete()`) to pick the best contact.
- Fills `email_report.md.j2` with the answer and data table.
- Returns a draft dict; `status: "needs_recipient"` if no contact matched.
- Never sends — `send_email()` is a separate explicit function.

**`fill_report(query, rows, cols, answer, entities)`**
- Fills `report.md.j2` with query, answer, Markdown table, column stats.
- Saves to `V6Config.report_dir()` with a timestamp filename.

---

### `speech.py` — Voice Layer (STT + TTS)

**Purpose:** Wraps the text pipeline with speech. `audio → STT → text → [graph] → polished answer → TTS → spoken audio`. The text pipeline is unchanged; this module only adds the front (transcription) and back (synthesis).

**`STT`** — faster-whisper (`large-v3`), CTranslate2. `float16` on GPU, `int8` fallback on CPU. `transcribe()` primes Whisper with a wilaya+KPI `initial_prompt` and fuzzy-corrects the transcript (see `transcribe.py`); `raw_text` keeps the pre-correction output for the benchmark's WER report.

**`TTS`** — Coqui XTTS-v2, driven through its **streaming** inference so speech starts one sentence after generation begins. `sentence_buffer()` groups the polisher's token stream into speakable sentences (a boundary is sentence punctuation followed by whitespace, so "253.4 million" is never split at the decimal). Conditioning latents are computed once per language and cached.

**`speakable(text, lang)` — the spoken-text normalizer (applied per sentence inside `stream()`):**
- **Currency words:** `DZD`/`DA` → "dinars" (`en`/`fr`) / "دينار" (`ar`). XTTS otherwise spells "D-Z-D".
- **Percent:** `%` → " percent" / " pour cent" / " بالمئة".
- **Artifact lines dropped:** a line that is purely a chart/report/email note or a file path (`📊 Chart saved: …`, `[chart]`) returns `""` and is skipped — the voice never reads a path.
- **Long-number safety net:** any 7-digit / comma-grouped number ≥ 1,000,000 that slipped past `numfmt` is collapsed to a scale phrase ("1.09 billion") so the voice never reads digits one by one.

**Voice quality:** the default speaker is `Claribel Dervla` (clearer than the old `Daisy Studious`); `TTS_SPEAKER_WAV_*` clones a reference WAV instead (the biggest quality lever). Sampling is tuned for stability via `TTS_TEMPERATURE` (0.6), `TTS_REPETITION_PENALTY` (2.5), `TTS_TOP_K`/`TTS_TOP_P`, and `TTS_ENABLE_SPLITTING`. XTTS-v2 is a local model and will not equal a hosted service like ElevenLabs; for that, a reference-WAV clone of a professional voice is the closest path, or swap the TTS backend behind this module's `get_tts()` seam.

**`language_for(text)`** delegates to `slm.lang_code` so the spoken language matches the written answer.

---

### `prompts.py` — All Prompt Templates

Contains:
- `build_router_messages(query, schema_prompt, knowledge, history, feedback)` — builds the router SLM's message list. The system prompt contains the schema, LOCATION RULE, TIME RULE, and 5 routing rules. The user turn contains the query, knowledge block, and conversation history.
- `parse_router_output(text)` — parses the router's JSON output into a `routing` dict.
- `build_sqlgen_instruction(query, routing, entities, schema)` — builds the SQL-gen turn injected as phase-2 user message. Contains the mandatory columns, date range, wilaya subquery pattern if needed, and aggregation rules.

The router's 5 rules (from the system prompt):
1. Map the query to a table and columns.
2. Extract wilaya mentions to `filters.wilayas`.
3. Extract time period to `filters.period`.
4. Output JSON only.
5. For follow-ups, inherit the last turn's tables/columns.

---

### `graph.py` — The Public API

**`build_graph()`** — assembles and compiles the LangGraph `StateGraph`. Sets up all nodes, the conditional edge from `brain`, and the fixed edges from each action back to `brain`. Returns the compiled graph with `MemorySaver`.

**`LatentMindV6`** — the agent class.
- `ask(query, thread_id, verbose)` — one turn. Calls `graph.invoke(initial_state(...))`. Returns the full result dict including `final_answer`, `timings`, `trace`.
- `reset(thread_id)` — clears SLM KV caches, brain embed cache, and rebuilds the graph (which creates a new `MemorySaver` checkpointer, erasing all conversation history).

**`get_agent()`** — module-level singleton.

---

## Training Pipeline (Required Before First Use)

The brain is inert without `models/brain_head.pt`. Build it once:

```bash
# 1. Synthesize agentic traces (~instant)
python3 -m v6.brain_data
# Output: v6/data/brain_train.jsonl (~8000 rows)

# 2. Train the MLP (3-4 min on T4, ~10 min on M1 CPU)
python3 -m v6.train_brain --epochs 200
# Output: models/brain_head.pt
```

On Colab, cells 6 and 7 do this. FORCE_RETRAIN=True in those cells always rebuilds.

---

## Data Flow: A Complete Query Trace

Query: `"Montre-moi la marge brute pour Oum El Bouaghi le trimestre dernier"`

```
initial_state(query)
        │
        ▼
brain_node
  encode_outcome([]) → outcome_vec all zeros (nothing happened yet)
  brain.decide(query, memory="", step_log=[], grounding=0.0)
  → intent="data", action="rag", continue=0.99
  thought: "Let me check the reference knowledge first."
        │
        ▼
route_after_brain → "rag"
        │
        ▼
rag_node
  retriever.knowledge_block("Montre-moi la marge brute pour Oum El Bouaghi le trimestre dernier")
  → top-5 chunks including: fpa_profitability.gross_margin definition,
    Oum El Bouaghi wilaya chunk (29 communes, subquery pattern)
  grounding = 0.52  (fpa_profitability chunk cosine score)
  step_log = [{action:"rag", ok:True, error_type:"none", ...}]
        │
        ▼
brain_node
  encode_outcome([{rag, ok, none}]) → last-action=rag, last-ok=1, done=[rag]
  → intent="data", action="sql", continue=0.98
  thought: "I'll query the database for the numbers."
        │
        ▼
route_after_brain → "sql"
        │
        ▼
sql_node → run_sql_pipeline(state)
  build_router_messages(query, schema, knowledge, history="", feedback="")
  slm.run_router(messages, thread_id) → routing JSON:
    {intent:"data", tables:["fpa_profitability","dim_location"],
     columns:["gross_margin","week_start","location_id"],
     filters:{wilayas:["Oum El Bouaghi"], period:"Q3 2025"}}
  orchestrator.assemble(...) → validated routing, confidence="high"
  entities.resolve_all(query, filters) → {wilayas:["Oum El Bouaghi"],
    wilaya_ids_map:{"Oum El Bouaghi": [id1, id2, ..., id29]}}
  build_sqlgen_instruction(query, routing, entities, schema)
  slm.run_sqlgen(thread_id, instruction) → SQL (phase 2, KV cache reused)
  validate_sql + consistency_check → valid, no issues
  enforce_limit → appends LIMIT 1000
  execute_sql → {ok:True, rows:[{avg_gross_margin:42.3903}], columns:["avg_gross_margin"]}
  _summarize_rows(rows, cols, lang="fr") → humanizes the figure (fr query)
  final_answer = "avg_gross_margin: 42,39 %"   ← frozen, already formatted
  step_log = [{rag}, {sql, ok:True, row_bucket:"one"}]
        │
        ▼
brain_node
  encode_outcome([{rag}, {sql,ok,one}]) → last-action=sql, last-ok=1, row-bucket=one, done=[rag,sql]
  → intent="data", action="template", continue=0.01
  thought: "I have what I need — writing the answer now."
        │
        ▼
route_after_brain
  continue_score=0.01 < BRAIN_SEUIL=0.5 → "communicator"
        │
        ▼
communicator_node
  intent="data", final_answer already set by sql_node
  no chart/report/email artifacts
  rolls memory: adds this turn to turns list
  → final_answer = "avg_gross_margin: 42,39 %"
        │
        ▼
END

→ Polisher (analyze) copies the frozen figure verbatim:
   "La marge brute moyenne pour Oum El Bouaghi le trimestre dernier
    était de 42,39 %."
→ TTS speakable("…42,39 %", "fr") → "…était de 42,39 pour cent." (spoken)
```

---

## Database Schema

All metric tables store data as weekly snapshots (`week_start` column) at the commune level (`location_id` integer). The join topology is simple: every metric table joins `dim_location` on `location_id`.

**Metric tables:**
- `fpa_profitability` — gross_margin, net_income, EBITDA, total_revenue, ...
- `postpaid_kpi` — subscribers, churn_rate, arpu, ...
- `prepaid_kpi` — active_base, arpu, recharge_amount, ...
- `opex_capex` — opex, capex, by category

**Dimension tables:**
- `dim_location` — location_id, wilaya, wilaya_code, commune, region
- `data_catalog` — table_name, column_name, description (self-documenting schema)
- `contacts` — id, email, name, role, department (for email recipient resolution)

---

## Environment Variables Reference

| Variable | Default | Effect |
|---|---|---|
| `V6_SLM_SIZE` | `4b` | `3b`, `4b`, or `7b` model |
| `V6_SLM_OVERRIDE` | `""` | Force a specific Hub model id |
| `V6_4BIT` | `0` | Enable NF4 4-bit quantization |
| `V6_SPECULATIVE` | `1` | Enable speculative decoding |
| `V6_CONSTRAINED_SQL` | `0` | Enable grammar-constrained SQL generation |
| `V6_USE_SQLITE` | `0` | Use SQLite instead of MySQL |
| `V6_SQLITE_PATH` | auto | Path to SQLite database file |
| `V6_BRAIN_SEUIL` | `0.5` | Continue threshold |
| `V6_BRAIN_MAX_STEPS` | `8` | Loop safety cap |
| `V6_OUTPUT_DIR` | auto | Output directory for charts/reports/emails |
| `V6_POLISHER_HUB_ID` | `Qwen/Qwen2.5-1.5B-Instruct` | Polisher model |
| `V6_STT_MODEL` | `large-v3` | faster-whisper model |
| `V6_STT_COMPUTE` | `float16` | `int8_float16` to save ~1.5 GB VRAM |
| `V6_TTS_SPEAKER_EN` / `_FR` | `Claribel Dervla` | XTTS-v2 built-in studio voice |
| `V6_TTS_SPEAKER_WAV_EN` / `_FR` | `""` | Path to a reference WAV to clone (overrides the name) |
| `V6_TTS_TEMPERATURE` | `0.6` | XTTS sampling temperature (lower = steadier) |
| `V6_TTS_REPETITION_PENALTY` | `2.5` | Reduces slurring/artefacts |
| `V6_TTS_TOP_K` / `V6_TTS_TOP_P` | `50` / `0.8` | XTTS nucleus sampling |
| `V6_TTS_ENABLE_SPLITTING` | `1` | Let XTTS handle its own internal sentence splitting |
| `V6_TTS_SPEED` | `1.0` | Speech rate |

---

## Benchmarking

`v6/benchmark.py` is the objective harness. It runs the fixed query set in
`data/bench_queries.json` (20 queries spanning every intent and capability,
EN + FR) through the live graph, then **mirrors the notebook's terminal
routing exactly** — so the answers it grades are the same answers the voice/text
path produces (clean frozen numbers, the right polisher role, verbatim
deflections).

### What it measures

- **Intent accuracy** — predicted intent vs. the expected label per query.
- **SQL exec rate** — share of data queries whose SQL ran and returned rows.
- **Latency** — brain ticks, RAG, SQL, and total, broken down by category.
- **Answer review** — for each query: the *spoken* text (post-polish, the
  number-rounding fix is visible here) alongside the *raw* data block.
- **Voice round-trip** (GPU only) — synthesizes each query to a WAV, runs it
  back through STT, reports **WER**, then diffs voice vs. text and lists only
  the queries that **regressed under voice** (e.g. an STT mishearing flipping
  an intent), so transcription noise is never mistaken for a pipeline bug.

### How to run

```python
from v6.benchmark import run_text, run_full

# Text pipeline only (fast, no GPU needed for the graph):
run_text(polish=True)          # polish=False shows raw figures, skips the 1.5B

# Text + voice round-trip (needs GPU + faster-whisper + XTTS):
run_full(save=True)            # writes bench_results_*.json / bench_voice_*.json
```

The polisher routing is `_spoken_role(row)`, kept byte-for-byte in sync with the
notebook's `_pick_role`. `off_topic` (and cross-turn report messages) return
`role=None` → spoken verbatim, never through the polisher.

### Reading a result

- A figure shown as `253.4 million DZD` in the spoken text and `253,387,711.02`
  in the raw line is the number-formatting fix working: `numfmt` froze the
  clean figure, the analyst copied it.
- A **voice regression** is reported only when the voice answer diverges from
  the text answer for the *same* query — almost always traceable to the listed
  WER / STT mishearing, not the brain.

---

## Serving (`server.py`)

A FastAPI app exposes the agent over HTTP + WebSocket so a web/mobile client can
call it. On Colab it runs behind an **ngrok** tunnel (the notebook's *Serve*
cells start `uvicorn` in a thread and open the tunnel).

**Everything runs server-side** — STT, brain, SLM, polisher, cloned-voice TTS.
A browser cannot run the XTTS clone or the Qwen/BGE models, and the Web Speech
API is robotic and can't clone a voice, so streaming audio from the server is
the only way to deliver the recorded voice.

**Streaming over one WebSocket (`/ws`)** — the client gets text and audio:
- `{type:"thinking"}` the 💭 trace as the brain works
- `{type:"meta", intent, role}` how the answer is produced
- `{type:"token"}` the answer text, streamed (renders ChatGPT-style)
- `{type:"audio", seq, sr, data}` cloned-voice **PCM16 base64** chunks, ~1
  sentence behind the text (XTTS already streams per sentence)
- `{type:"artifact"}` chart/report/email pointers, then `{type:"answer"}`, `{type:"done"}`

The blocking pipeline runs in a thread; events are marshalled to the event loop
through a queue, so the socket stays responsive while the GPU works.

**REST fallbacks:** `POST /ask` (JSON, non-streaming), `POST /ask_voice`
(multipart audio → STT → pipeline, returns text + PCM16 chunks),
`GET /chart/{name}`, `GET /health`, and `GET /` (a minimal browser client that
plays the streamed audio gaplessly via the Web Audio API).

**Constraints:** one GPU → an `asyncio.Lock` serializes requests (one at a
time). ngrok is public → every endpoint requires `V6_API_TOKEN` (bearer header
or `?token=`); if unset, one is generated and printed at startup. A Colab
session is ephemeral — this is a demo/dev server. For production, run the same
app on a persistent GPU host (Modal / RunPod / HF Endpoints) behind a queue.

---

## Key Design Decisions and Why

**Why a trained MLP brain, not a rules-based planner?**
The old planner was a one-shot classifier: decide intent + capabilities → run a fixed sequence. It could not react to a failed SQL query, a 0-row result, or an email with no recipient. The MLP re-decides after every step, so it can: retry SQL on failure (by seeing `error_type=sql_error` in the outcome vector), skip chart when no rows came back, stop early when continue < 0.5.

**Why no fallback to heuristics in `route_after_brain`?**
Heuristics that mimic the brain's job belong in training traces, not in Python conditions. A condition like "if intent == data and no rag yet: go to rag" is a policy decision — encode it in a trace and let the brain learn it. This keeps the routing logic in one place (brain_data.py) and prevents the policy from splitting across training data and code.

**Why does the seuil gate ALL actions, including terminals?**
A brain that exempts terminals from the seuil can accidentally trigger a report/chart on every SQL result — the action head might pick "template" as its argmax even when continue=0.01 (the brain is done). The fix is to treat the continue score as "do I want ANOTHER step" regardless of which step it is. A deliberate "put it in a report" request trains the brain to output continue≈1.0 before template; an incidental argmax trains it to output continue≈0.01.

**Why KV-cache hand-off between router and SQL generator?**
Re-encoding the same ~512-token context twice is the dominant latency cost. Phase 2 adds about 150 tokens to phase 1's already-computed sequence. Inheriting the KV cache means phase 2 only processes those 150 new tokens instead of re-computing all 512 + 150. On a T4, this saves about 1.5 seconds per query.

**Why format numbers in Python instead of letting the polisher round them?**
The polisher is a 1.5B model. Asked to round "1,087,355,290.78" it produced "52,590,189,81" — a corrupted figure with two decimal commas. For an analytics tool that is a critical failure: a confidently wrong number is worse than no number. So formatting is pulled out of the model entirely. `numfmt` rounds, scales, and unit-tags every figure deterministically *before* the polisher sees it; the analyst prompt then says figures are FINAL — copy them, never recompute. A frozen "253.4 million DZD" is trivial to copy verbatim; the raw 12-digit value that invited corruption never enters the model's context. The same clean figure is what the TTS layer speaks, so the voice says a short phrase instead of reading ten digits.

**Why a `speakable()` pass before TTS instead of just feeding it the answer?**
XTTS reads text literally: "DZD" becomes the letters D-Z-D, "%" is hit or miss, and a file path in a chart note gets read aloud. `speakable()` rewrites each sentence into what it should *sound* like — currency and percent become words, artifact/path lines are dropped, and any long number that escaped `numfmt` is collapsed. It runs per sentence inside the TTS stream, so every voice path (notebook, benchmark) gets it for free without changing the text answer shown on screen.

**Why subquery for wilaya filtering, not inline ID list?**
`dim_location` is commune-level. Alger has 57 communes, Oran has 26. An inline `location_id IN (1, 2, ..., 57)` would consume ~200 tokens in the SQL and make the SQL generator prone to hallucinating IDs. The subquery `WHERE location_id IN (SELECT location_id FROM dim_location WHERE wilaya = 'Alger')` is 15 tokens and delegates ID resolution to the database.
