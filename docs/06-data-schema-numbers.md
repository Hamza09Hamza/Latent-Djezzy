# 06 — Data, Schema & Numbers

> Code/data: `v6/schema.py`, `v6/data/glossary.json`, `v6/data/kpi_catalog.json`,
> `v6/data/wilaya_aliases.json`, `v6/numfmt.py`, `interndb.sqlite` (SQLite) / MySQL.

The database V6 queries, the artifacts that describe it, and the single most
important reliability mechanism in the system: **how numbers are kept correct.**

---

## 1. The database

Weekly KPI snapshots for an Algerian mobile operator, stored at the **commune**
level. The backend is MySQL locally and **SQLite** on Colab (`interndb.sqlite`);
V6 auto-detects which to use and **introspects the live schema at startup** —
nothing about the schema is hardcoded.

### Logical schema

**Metric tables** — one row per (commune, week), keyed by `location_id` +
`week_start`:

| Table | KPIs (examples) | Segment |
|---|---|---|
| `prepaid_kpi` | churn_rate, arpu, active_base, recharge_amount, … (14) | prepaid |
| `postpaid_kpi` | churn_rate, arpu, subscribers, … (13) | postpaid |
| `global_revenue` | company-wide revenue lines (8) | all |
| `fpa_profitability` | gross_margin, net_income, EBITDA, total_revenue, … (11) | all |
| `opex_capex` | opex, capex by category (12) | all |

**Dimension / support tables:**

| Table | Columns | Role |
|---|---|---|
| `dim_location` | location_id, wilaya, wilaya_code, commune, region | geography (58 wilayas → many communes) |
| `data_catalog` | table_name, column_name, description | self-documenting schema (feeds RAG) |
| `contacts` | id, email, name, role, department, phone, … | email recipient resolution |

The **join topology is uniform**: every metric table joins `dim_location` on
`location_id`. There is **no** `wilaya` column on a metric table — `schema.py`
*derives* the join map by detecting which tables carry `location_id`.

### Key data rules (`glossary.json`)

- **Time:** weekly snapshots keyed by `week_start`. Relative expressions ("last
  quarter," "recently") resolve against the **most recent available week**, not the
  real-world date.
- **Geography:** wilaya names stored in **French** — "Alger," not "Algiers." Wilaya
  queries filter through `dim_location.wilaya` after joining on `location_id`.
- **Segments:** prepaid → `prepaid_kpi`, postpaid → `postpaid_kpi`, company-wide →
  `global_revenue`, profitability → `fpa_profitability`, costs → `opex_capex`. No
  segment named and the KPI exists for both → **default to prepaid.**

---

## 2. The knowledge artifacts

Human-editable JSON that describes the DB to the retriever and the prompts (the
ground truth the model is grounded against).

- **`kpi_catalog.json`** — 58 entries: `id`, `column`, `table`, `description`,
  `segment`, `unit`, multilingual `synonyms`. The `unit` field drives `numfmt`.
- **`glossary.json`** — `system` (persona), `business_context` (5), `table_relationships`
  (6), `definitions` (7), `query_type_guide` (4).
- **`wilaya_aliases.json`** — canonical → alias map bridging user spellings to the
  58 canonical French names.

See [RAG, Knowledge & Entities](05-rag-knowledge-entities.md) for how these are
retrieved and resolved.

---

## 3. The number trust boundary (`numfmt.py`) — V6's most important fix

This is an analytics tool, so a wrong number is a **critical failure** — a
confidently wrong figure is worse than no figure, because a human acts on it.

### The bug that motivated it

A 1.5B polisher, asked to "round" `1,087,355,290.78`, produced `52,590,189,81` — a
**corrupted** figure with two decimal commas and the wrong magnitude. The lesson:
**a small language model cannot be trusted to reformat a raw number.**

### The fix: freeze the figure in Python *before* the model sees it

Number formatting is pulled **entirely** out of the model. `numfmt.py` rounds,
scales, and unit-tags every figure deterministically, in the query's language,
*before* the polisher (or the chart, or the TTS) ever sees it. The polisher's job
becomes pure prose: **copy the frozen figure verbatim** — never re-round, never do
arithmetic, never invent a `$`.

### How it formats

- **`unit_for_column(col)`** — infers a unit (`DZD | % | count | GB | min | months
  | days`) for a possibly aggregated/aliased column. Order: exact hit in
  `kpi_catalog.json` → hit after stripping an aggregation prefix (`avg_`, `total_`,
  `sum_`, …) → keyword heuristic (`*rate|margin|ratio|share` → `%`;
  `revenue|income|opex|arpu|…` → `DZD`; `subscriber|adds|base` → `count`). `None`
  if nothing matches (renders `—`).
- **`humanize(value, unit, lang)`** — the locale-aware core:

| Input | unit | `en` | `fr` |
|---|---|---|---|
| 1,087,355,290.78 | DZD | `1.09 billion DZD` | `1,09 milliard DZD` |
| 253,387,711.02 | DZD | `253.4 million DZD` | `253,4 millions DZD` |
| 42.4247 | % | `42.42%` | `42,42 %` |
| 3.071 | DZD | `3.07 DZD` | `3,07 DZD` |
| `None` | any | `—` | `—` |

Thresholds: `≥1e9` → billions (2 dp), `≥1e6` → millions (1 dp), `≥1e3` → grouped
whole number, `<1e3` → up to 2 dp. Percentages keep 2 dp + symbol. French uses a
decimal comma and French scale words. Non-numeric values (dates, wilaya names) pass
through untouched.

- **`humanize_cell(col, value, lang)`** = `humanize(value, unit_for_column(col),
  lang)`. The row-summarizer formats *every* cell in the query's language, so the
  frozen figure is already correct and locale-correct before any model or the
  screen sees it.

Run `python3 -m v6.numfmt` to print the full formatting table for both languages.

### Why it matters at three layers

The **same frozen string** flows to all three outputs:

1. **The polisher** copies it into prose ("…était de 42,39 %").
2. **The chart** (`chartspec.py`) renders pre-frozen figures — it can't re-round.
3. **The TTS** speaks it ("…42,39 pour cent"), and `speakable` collapses any long
   number that *still* slipped through (so the voice never reads ten digits one by
   one — see [Voice Layer](07-voice-layer.md)).

This is the cleanest expression of "think small, verify hard": a 1.5B–4B model is
plenty for prose, *if* you never let it touch a raw number.

→ Next: [Voice Layer](07-voice-layer.md).
