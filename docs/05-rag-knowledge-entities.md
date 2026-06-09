# 05 — RAG, Knowledge & Entities

> Code: `v6/knowledge.py` (RAG encoder + retriever), `v6/entities.py` (wilaya/date
> resolution), `v6/schema.py` (live introspection), `v6/orchestrator.py`
> (deterministic plan assembly).

Retrieval and resolution are the two halves of *grounding* — making sure the model
talks about KPIs that exist, for wilayas that exist, over dates that exist. Four
independent, deterministic layers each catch a different class of mistake.

---

## 1. The encoder — BGE-M3 (`knowledge.py`)

- **`BAAI/bge-m3`** — an XLM-RoBERTa-based model producing **1024-d** CLS
  embeddings, strongly multilingual (FR/EN/AR/Darija in one space).
- **L2-normalized outputs**, so cosine similarity is a dot product.
- **Always runs on CPU on Apple Silicon.** BGE-M3 returns *wrong* embeddings for a
  batch-of-1 encode on MPS, so `encoder_device()` forces CPU there (CUDA runs on
  GPU).

The same encoder also featurizes the **brain's** query/memory vectors — so the
brain and the retriever literally share one semantic space.

---

## 2. The four knowledge sources

The retriever builds an in-memory vector store over four sources, each chosen to
ground a different failure mode:

### a) `data_catalog` table — the database documenting itself

Each row is `(table_name, column_name, description)`, giving the router SLM
**authoritative** descriptions of every column straight from the DB. The schema
documents itself, so the knowledge base never drifts from reality.

### b) `dim_location` + `wilaya_aliases.json` — one chunk per wilaya

Each wilaya chunk contains its canonical French name, official numeric wilaya code,
aliases, commune count, and — critically — **the correct subquery pattern** to
filter by it:

```
WHERE <table>.location_id IN (
    SELECT location_id FROM dim_location WHERE wilaya = 'Oum El Bouaghi'
)
```

Teaching this pattern *in the knowledge base* is what stops the SLM from hardcoding
`location_id IN (1, 2, 3, …)`.

### c) `kpi_catalog.json` — the KPI dictionary (58 entries)

Per KPI: canonical column name, table, segment, unit, description, and
**multilingual synonyms**. Distribution: `prepaid_kpi` (14), `postpaid_kpi` (13),
`fpa_profitability` (11), `opex_capex` (12), `global_revenue` (8). The `unit` field
is what lets `numfmt` know `arpu` is DZD and `churn_rate` is `%`; the synonyms let
"churn"/"désabonnement"/"خسارة" resolve to one column.

### d) `glossary.json` — definitions + business rules

System persona, **business context** (5: coverage, geography, segments, …),
**table relationships** (6: the join rules), **definitions** (7: ARPU, churn,
EBITDA, …), and a **query-type guide** (4). Answers "what does ARPU mean?"
(definition intent) and encodes the rule that **relative time resolves against the
most recent available week, not the wall clock.**

---

## 3. Retrieval API & the grounding score

- **`knowledge_block(query)`** — encode the query, return the **top-k** chunks by
  cosine. Injected into the router SLM's prompt.
- **The grounding score** — the **max cosine among *non-wilaya* chunks.** Wilaya
  chunks are excluded: a user saying "Oran" has found a *location*, not evidence the
  DB can answer the *KPI*. Below the floor (`RAG_LOW_CONF ≈ 0.45`), the orchestrator
  down-rates confidence and the brain may stop to clarify. This is the
  unanswerable-question detector — and grounding is also a feature in the brain's
  25-d outcome vector, so the policy itself can react to weak retrieval.
- **`definition_for(query)`** — returns the best definition/KPI chunk for a "what
  does X mean" turn; used directly by the communicator for the `definition` intent.

---

## 4. Entity resolution (`entities.py`) — the Algiers ≠ Alger fix

Users write "Algiers," "الجزائر," "Alger," "alger," "ALGER," "Bejaia" (no accent),
"M'Sila" (apostrophe variant). The database stores exactly "Alger," "Béjaïa,"
"M'Sila." A mismatch silently drops a city from a comparison.

- **`_norm(s)`** — the normalization key: lowercase, strip accents (NFKD), remove
  apostrophes/dashes/punctuation, collapse spaces. "Oum El Bouaghi" and
  "oum el bouaghi" become the same key.
- **`Resolver`** — loads every `(wilaya, location_id)` from `dim_location` at
  startup, builds a `{normalized_form → canonical_name}` index, and folds in
  `wilaya_aliases.json`. Resolution is **exact normalized lookup first, then fuzzy**
  (`difflib`, cutoff 0.86). `scan_query()` does **greedy longest-match** scanning
  (so "Tizi Ouzou" matches before "Tizi").
- **`resolve_all(query, router_filters, max_date)`** — combines the router's
  extracted names with a direct query scan; returns `{wilayas, wilaya_ids_map,
  unresolved_wilayas}` — canonical names plus the commune-level `location_id`s each
  expands to. Everything downstream uses the **canonical** spelling, so the SQL
  never says `wilaya = 'Algiers'`.

There is also **date/period resolution**: relative expressions resolve against the
data's most recent available week (`schema._find_date_range`), and
`V6_REFERENCE_DATE` can pin "today" for a stale Colab clock.

---

## 5. Live schema introspection (`schema.py`) — the "no such column" fix

V6 **never hardcodes** the schema. `DBSchema` introspects the live database
(`_introspect_sqlite` / `_introspect_mysql`) at startup:

- reads every table/column/type,
- loads human descriptions from `data_catalog`,
- finds the available date range (`MIN/MAX(week_start)`) so the SQL-gen SLM knows
  what dates exist,
- **derives the join map**: any table (other than `dim_location`) with a
  `location_id` column is recorded as needing a `dim_location` join for wilaya
  filtering.

That derived join map is the structural reason the SLM never invents a `wilaya`
column on a metric table — metric tables have `location_id` and must join
`dim_location`. `prompt()` renders the schema block injected into the router SLM
with the **LOCATION RULE** (filter via `location_id IN (subquery)`) and the **TIME
RULE** (`week_start` is the date column).

**Why a subquery, not an inline ID list?** `dim_location` is commune-level — Alger
has 57 communes, Oran 26. An inline `location_id IN (1, …, 57)` costs ~200 tokens
and invites hallucinated IDs. The subquery is ~15 tokens and delegates ID
resolution to the database. The knowledge base teaches the pattern; the consistency
check enforces it.

---

## 6. The deterministic orchestrator (`orchestrator.py`)

Between the router SLM and the SQL generator sits a **pure-logic fact-checker** —
validation, not classification. `assemble(...)`:

1. **Validate tables** — drop any table the SLM named that doesn't exist.
2. **Validate columns** — drop any column not present in any table.
3. **Structural rescue** — if no valid metric table came back (a follow-up with no
   KPI keywords), inherit the last data turn's tables+columns.
4. **`dim_location` injection** — if a wilaya filter is present and a metric table
   has `location_id`, add `dim_location`.
5. **Confidence rating** — `high` (metric table + KPI column + grounding ≥ 0.45),
   `medium` (grounding above floor or tables inherited), `low` (no metric table +
   low grounding → clarify).

The result is a *validated plan*: a routing object referencing only real tables and
columns, with the join injected and a confidence label the brain can act on.

---

## 7. The whole grounding stack, in order

```
query
  │  knowledge.knowledge_block(query)        → top-k chunks + grounding score
router SLM (sees schema + knowledge + history)
  │  orchestrator.assemble(routing, schema)  → tables/columns validated, join injected, confidence rated
  │  entities.resolve_all(query, filters)    → wilayas → canonical names → location_ids
  │  SQL-gen SLM (sees the subquery pattern)
  │  sql_tools.consistency_check(sql, ...)   → reject hallucinated columns / inline IDs / non-canonical wilayas
  ▼
execute against the live DB
```

Four independent layers (semantic retrieval, schema validation, entity resolution,
SQL consistency), none of them the model, all deterministic. That redundancy is the
"verify hard" half of "think small, verify hard."

→ Next: [Data, Schema & Numbers](06-data-schema-numbers.md).
