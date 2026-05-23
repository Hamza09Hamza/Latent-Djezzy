# LatentMind V6

An agentic analytics assistant for an Algerian telecom database (`interndb`),
built as a **LangGraph** state machine. It answers KPI questions in natural
language, and on request charts the result, drafts an email, or fills a
report template — while remembering the conversation across turns.

V6 is a ground-up rethink of V5. It keeps the LatentMAS idea (one small model,
two roles, KV-cache hand-off) but puts a real agent graph around it and fixes
the five bugs that made V5 unreliable.

## Why V6 exists — the V5 bugs, and the fix

| V5 failure | Root cause | V6 fix |
|---|---|---|
| `no such column: wilaya` | the schema, glossary and prompts all *claimed* `global_revenue` had a `wilaya` column | [schema.py](schema.py) introspects the live DB; the join map is **derived** — every metric table joins `dim_location` via `location_id`, no exceptions |
| churn comparison silently dropped a city | DB stores `Alger`, the query said `Algiers` | [entities.py](entities.py) resolves mentions to real values (alias + accent-fold + fuzzy) |
| "hello" generated SQL | the model routed in free text, parsed by keyword grep | the [latent planner](planner.py) classifies intent in embedding space; a greeting never reaches SQL |
| a trend query invented a `WHERE Oran` filter | history bled into the prompt + KV cache | [sql_tools.py](sql_tools.py) `consistency_check` flags any filter not in the resolved intent; bounded regenerate |
| no recovery when a step failed | rigid linear pipeline | the graph re-plans: a failed execution loops back through the router with feedback |

## Architecture

```
                          ┌──────────┐
   user query ───────────▶│   plan   │  latent planner (BGE-M3 embedding space)
                          └────┬─────┘  → intent + capabilities
              greeting/meta/   │   data
              definition       │
            ┌──────────────────┴───────────────┐
      ┌─────▼──────┐                    ┌───────▼────┐
      │direct_answer│                   │  retrieve  │ BGE-M3 RAG
      └─────┬──────┘                    ├───────▼────┤
            │                           │   router   │ SLM phase 1 — schema mapping
            │                           ├───────▼────┤
            │                           │orchestrator│ deterministic: validate vs live schema
            │                           └───┬────┬───┘
            │                          clarify    resolve_entities
            │                              │          │
            │                              │   sql_generate ⇄ sql_validate   (SLM phase 2,
            │                              │          │       reuses KV cache)
            │                              │     sql_execute → answer
            │                              │          │
            │                              │   (fail) replan ──▶ router
            │                              │          │
            │                              │   visualize → template → email
            └──────────────────────────────┴──────────┴───────┐
                                       ┌──────────┐            │
                                       │ finalize │◀───────────┘
                                       └────┬─────┘
                                           END
```

The **planner decides, the orchestrator validates.** The plan — intent plus a
capability set — is decoded once into a discrete object; execution then runs
predictably. Dynamic decision, predictable execution.

## The latent planner

The route is *not* chosen by regex and *not* chosen by a free-running LLM. The
query (and recent history) is classified in **BGE-M3 embedding space** — the
encoder already loaded for RAG, so it costs one matmul.

- **`prototype` mode (default):** nearest-prototype. Each intent/capability has
  a few example phrases in [data/planner_prototypes.json](data/planner_prototypes.json).
  Add a phrasing there and the planner learns it — no training.
- **`mlp` mode:** a trained head (`query⊕history` embedding → 256 → intent +
  capabilities). Train it whenever you want:

  ```bash
  python3 -m v6.planner_data     # synthesize data/planner_train.jsonl
  python3 -m v6.train_planner    # → models/planner_head.pt
  V6_PLANNER=mlp python3 -m v6.cli
  ```

## Capabilities

- **visualize** — matplotlib chart (line for trends, bar for comparisons).
- **template** — a Jinja2 report ([templates/report.md.j2](templates/report.md.j2))
  filled with the data and written to disk.
- **email** — resolves a recipient from the `contacts` table and **drafts** an
  email. It never sends. `capabilities.send_email(draft)` is a separate,
  explicit action (needs `V6_SMTP_*` env vars).

## Run locally

```bash
pip install -r v6/requirements.txt
python3 -m v6.cli              # demo queries
python3 -m v6.cli -i           # interactive REPL
python3 -m v6.cli -t "what was the total revenue in Oran"
python3 -m v6.test_v6          # verification harness (no SLM needed)
```

The CLI auto-detects `interndb.sqlite` in the repo root; otherwise it uses
MySQL. Place a model under `models/qwen2.5-coder-1.5b-instruct/` or set
`V6_SLM_OVERRIDE`.

## Run on Colab (T4/L4)

Open `v6_colab.ipynb`, set a GPU runtime, and run the cells. It clones the
repo, reads `interndb.sqlite` from Google Drive, and runs Qwen2.5-Coder-3B.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `V6_USE_SQLITE` | `0` | `1` for the SQLite backend (Colab) |
| `V6_SQLITE_PATH` | auto | path to the SQLite DB |
| `V6_SLM_OVERRIDE` | – | force a HuggingFace model id |
| `V6_PLANNER` | `prototype` | `prototype` or `mlp` |
| `V6_4BIT` | `0` | 4-bit NF4 quantization |
| `V6_FLASH_ATTN` | `0` | Flash Attention 2 |
| `V6_SMTP_USER` / `V6_SMTP_PASSWORD` / `V6_SMTP_HOST` | – | only needed to actually send a drafted email |
| `LATENTMIND_MYSQL_*` | localhost/interndb | MySQL connection (local) |

## Module map

| File | Role |
|---|---|
| [graph.py](graph.py) | LangGraph assembly + `LatentMindV6.ask()` |
| [nodes.py](nodes.py) | the graph node functions |
| [planner.py](planner.py) | latent planner — intent + capabilities |
| [orchestrator.py](orchestrator.py) | deterministic validation + plan assembly |
| [schema.py](schema.py) | live DB introspection + join map |
| [entities.py](entities.py) | wilaya / date / segment resolution |
| [knowledge.py](knowledge.py) | BGE-M3 encoder + RAG retriever |
| [slm.py](slm.py) | Qwen dual-role engine + KV-cache hand-off |
| [prompts.py](prompts.py) | router / SQL prompts + parsing |
| [sql_tools.py](sql_tools.py) | SQL safety, consistency check, execution |
| [capabilities.py](capabilities.py) | chart / email-draft / report |
| [cli.py](cli.py) | REPL + demo |
| [test_v6.py](test_v6.py) | verification harness |

this ReadME.md is generated by AI, so we apologize for any mistake 