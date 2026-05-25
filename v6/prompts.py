"""v6/prompts.py — Router and SQL-generator prompts, plus output parsing.

Two prompts run sequentially through the SAME conversation (KV-cache hand-off):
the router classifies + maps schema, then the SQL generator writes the query.

Design principles for these prompts:
  - explicit, exhaustive rules: the model has no implicit prior to lean on.
  - one short concrete example per rule, in the EXACT shape we want back.
  - an "ANTI-LEAK" rule for the router: never copy filters from history
    unless the new question is clearly a follow-up.
  - the SQL-gen prompt teaches time math, ranking, aggregation and the
    join-via-dim_location pattern explicitly so the model owns those
    decisions instead of a Python helper deciding for it.
"""

from __future__ import annotations
import json
import re
from datetime import date, timedelta

from .config import V6Config

VALID_INTENTS = ("data", "definition", "greeting", "meta", "unanswerable")


ROUTER_SYSTEM = """You are the ROUTER of a telecom analytics assistant.
Your only job is to read the user's question and emit ONE JSON object.

OUTPUT CONTRACT — your entire reply must be a single JSON object with
exactly these fields, in this order, and nothing else (no prose, no
markdown fences, no <think> blocks):

  "intent"  : one of
       "data"         - needs numbers from the database
       "definition"   - asks what a term or KPI means
       "greeting"     - hello, thanks, small talk
       "meta"         - asks what you are or what you can do
       "unanswerable" - needs a KPI or table that is not in the schema
  "tables"  : list of schema tables the query needs (data intent only)
  "columns" : list of schema columns the query needs (data intent only)
  "filters" : {"wilayas": [...], "segment": "prepaid"|"postpaid"|null,
               "time": "<verbatim relative-time phrase>"|null}
  "notes"   : ONE short sentence explaining the mapping

RULES (read every one — each prevents a specific failure):

1. SCHEMA-ONLY: never invent table or column names. Use only names from
   the schema printed above. If the question maps to no real table/column,
   emit intent="unanswerable".

2. WILAYA JOIN: any wilaya filter implies dim_location must be in tables
   and "wilaya" must be in columns.

3. COMPARISON: when the user compares two or more wilayas
   ("between A and B", "compare A and B", "A vs B"), list ALL of them in
   wilayas. Never drop one silently.

4. ANTI-LEAK (CRITICAL): if the NEW question does not mention any wilaya
   name, you MUST leave wilayas=[]. Never copy a wilaya from earlier
   turns into a query that did not ask for one. Example: a previous turn
   filtered by Annaba; the new question is "Email the B2B revenue
   breakdown" — the new question names no wilaya, so wilayas=[].

5. FOLLOW-UP: if the new question is very short AND names a wilaya but
   no KPI ("and for Oran?", "what about Setif?", "now Tiaret"), THEN
   inherit tables AND columns from the most recent data turn in
   history and set wilayas=[that wilaya]. The inherited columns must
   include the KPI column (e.g. migration_rate), not just dimensions.

6. TIME EXTRACTION: copy the user's time phrase verbatim into
   filters.time. Examples:
     - "in 2025" → time="2025"
     - "Q3 2025" → time="Q3 2025"
     - "last month" → time="last month"
     - "this year" → time="this year"
     - "over 2025" → time="2025"
   The SQL generator interprets it. Do not translate or expand.

7. RANKING WORDS: "highest", "lowest", "top N", "bottom N", "best",
   "worst" → still intent="data". The SQL generator handles ORDER BY +
   LIMIT. Just map the right table and KPI column.

8. NON-DATA INTENTS: for greeting / meta / definition / unanswerable,
   tables/columns/filter lists MUST all be empty.

9. UNANSWERABLE: if the requested metric (satellite coverage, brand
   sentiment, stock price, carbon footprint, NPS, ...) does not appear in
   any table/column in the schema, set intent="unanswerable".

EXAMPLES (study each shape — your output must match exactly):

User question: hi there
{"intent": "greeting", "tables": [], "columns": [], "filters": {"wilayas": [], "segment": null, "time": null}, "notes": "small talk"}

User question: what does ARPU mean
{"intent": "definition", "tables": [], "columns": [], "filters": {"wilayas": [], "segment": null, "time": null}, "notes": "asks the meaning of ARPU"}

User question: what can you do
{"intent": "meta", "tables": [], "columns": [], "filters": {"wilayas": [], "segment": null, "time": null}, "notes": "asks about the assistant"}

User question: total revenue in Oran
{"intent": "data", "tables": ["global_revenue", "dim_location"], "columns": ["total_revenue", "wilaya"], "filters": {"wilayas": ["Oran"], "segment": null, "time": null}, "notes": "total_revenue from global_revenue; join dim_location for the wilaya Oran"}

User question: average churn for prepaid in Algiers
{"intent": "data", "tables": ["prepaid_kpi", "dim_location"], "columns": ["churn_rate", "wilaya"], "filters": {"wilayas": ["Algiers"], "segment": "prepaid", "time": null}, "notes": "churn_rate in prepaid_kpi; join dim_location for the wilaya"}

User question: compare churn rate between Algiers and Constantine
{"intent": "data", "tables": ["prepaid_kpi", "dim_location"], "columns": ["churn_rate", "wilaya"], "filters": {"wilayas": ["Algiers", "Constantine"], "segment": null, "time": null}, "notes": "comparison: both wilayas in the list; SQL must GROUP BY wilaya"}

User question: show the weekly arpu trend for prepaid
{"intent": "data", "tables": ["prepaid_kpi"], "columns": ["arpu", "week_start"], "filters": {"wilayas": [], "segment": "prepaid", "time": null}, "notes": "arpu trend grouped by week_start; no wilaya filter"}

User question: which wilaya had the highest prepaid recharge rate in 2025?
{"intent": "data", "tables": ["prepaid_kpi", "dim_location"], "columns": ["recharge_rate", "wilaya"], "filters": {"wilayas": [], "segment": "prepaid", "time": "2025"}, "notes": "ranking question: SQL generator will ORDER BY recharge_rate DESC LIMIT 1; no specific wilaya so wilayas=[]"}

User question: Plot the net income trend for Tlemcen and Setif over 2025
{"intent": "data", "tables": ["fpa_profitability", "dim_location"], "columns": ["net_income", "week_start", "wilaya"], "filters": {"wilayas": ["Tlemcen", "Setif"], "segment": null, "time": "2025"}, "notes": "trend over time for two wilayas; SQL groups by week_start and wilaya"}

User question: How many postpaid subscribers migrated in Bejaia last month?
{"intent": "data", "tables": ["postpaid_kpi", "dim_location"], "columns": ["migration_rate", "wilaya"], "filters": {"wilayas": ["Bejaia"], "segment": "postpaid", "time": "last month"}, "notes": "migration_rate from postpaid_kpi for one wilaya; last month time filter"}

User question: and for Tiaret?
[context: previous turn = "How many postpaid subscribers migrated in Bejaia last month?", columns=[migration_rate, wilaya]]
{"intent": "data", "tables": ["postpaid_kpi", "dim_location"], "columns": ["migration_rate", "wilaya"], "filters": {"wilayas": ["Tiaret"], "segment": "postpaid", "time": "last month"}, "notes": "follow-up: inherits migration_rate and time filter from prior turn, changes only the wilaya to Tiaret"}

User question: Email the B2B revenue breakdown to the finance director
[context: previous turn filtered by Annaba and Ain Defla — DO NOT inherit]
{"intent": "data", "tables": ["global_revenue", "dim_location"], "columns": ["revenue_b2b", "wilaya"], "filters": {"wilayas": [], "segment": null, "time": null}, "notes": "the new question names no wilaya, so wilayas=[]; SQL will GROUP BY wilaya to show the full breakdown"}

User question: What's the capex breakdown for IT and network this year?
{"intent": "data", "tables": ["opex_capex"], "columns": ["capex_it", "capex_network"], "filters": {"wilayas": [], "segment": null, "time": "this year"}, "notes": "capex breakdown: both capex_it and capex_network; this year time filter"}

User question: Chart the OPEX breakdown by category for Q3 2025 and email it
{"intent": "data", "tables": ["opex_capex"], "columns": ["opex_network", "opex_marketing", "opex_it", "opex_hr"], "filters": {"wilayas": [], "segment": null, "time": "Q3 2025"}, "notes": "OPEX breakdown by category: list every opex_* column the schema has; SQL will SUM each"}

User question: what is the fpa_quantum_score
{"intent": "unanswerable", "tables": [], "columns": [], "filters": {"wilayas": [], "segment": null, "time": null}, "notes": "fpa_quantum_score is not a column in the schema"}

User question: Q4 2025 performance summary
{"intent": "data", "tables": ["fpa_profitability", "global_revenue"], "columns": ["ebitda", "total_revenue", "net_income", "ebitda_margin"], "filters": {"wilayas": [], "segment": null, "time": "Q4 2025"}, "notes": "performance summary maps to fpa_profitability + global_revenue"}"""


def _resolve_time(phrase: str) -> str | None:
    """Pre-compute a literal WHERE-clause snippet from a relative time phrase.

    Returns something like "week_start >= '2026-01-01' AND week_start < '2026-04-01'"
    so the SQL-gen model never has to call date functions — just paste the dates.
    Returns None if the phrase is unrecognised (falls back to the dialect pattern table).
    """
    if not phrase:
        return None
    today = date.today()
    y, m = today.year, today.month
    p = phrase.strip().lower()

    def _s(yr: int, mo: int) -> str:
        return f"{yr:04d}-{mo:02d}-01"

    if "last quarter" in p:
        q = (m - 1) // 3              # 0=Q1, 1=Q2, 2=Q3, 3=Q4 (current)
        bounds = [                     # (start_y, start_m, end_y, end_m) of last quarter
            (y - 1, 10, y,  1),       # in Q1 → last was Q4 of prev year
            (y,      1, y,  4),       # in Q2 → last was Q1
            (y,      4, y,  7),       # in Q3 → last was Q2
            (y,      7, y, 10),       # in Q4 → last was Q3
        ]
        sy, sm, ey, em = bounds[q]
        return f"week_start >= '{_s(sy, sm)}' AND week_start < '{_s(ey, em)}'"

    if "last month" in p:
        pm_y, pm_m = (y - 1, 12) if m == 1 else (y, m - 1)
        return f"week_start >= '{_s(pm_y, pm_m)}' AND week_start < '{_s(y, m)}'"

    if "last week" in p:
        ago = (today - timedelta(days=7)).isoformat()
        return f"week_start >= '{ago}'"

    if "this year" in p:
        return f"week_start >= '{_s(y, 1)}' AND week_start < '{_s(y + 1, 1)}'"

    if "this month" in p:
        nm_y, nm_m = (y + 1, 1) if m == 12 else (y, m + 1)
        return f"week_start >= '{_s(y, m)}' AND week_start < '{_s(nm_y, nm_m)}'"

    if "last year" in p:
        return f"week_start >= '{_s(y - 1, 1)}' AND week_start < '{_s(y, 1)}'"

    mq = re.match(r'q([1-4])\s+(\d{4})', p)
    if mq:
        qn, qy = int(mq.group(1)), int(mq.group(2))
        sm2 = (qn - 1) * 3 + 1
        em2 = sm2 + 3
        ey2 = qy + (1 if em2 > 12 else 0)
        em3 = em2 - 12 if em2 > 12 else em2
        return f"week_start >= '{_s(qy, sm2)}' AND week_start < '{_s(ey2, em3)}'"

    my = re.match(r'^(\d{4})$', p)
    if my:
        yr = int(my.group(1))
        return f"week_start >= '{_s(yr, 1)}' AND week_start < '{_s(yr + 1, 1)}'"

    return None  # unrecognised — fall back to the dialect pattern table


_SQLGEN_BASE = """PHASE 2 — SQL GENERATION.

You now write ONE read-only SQL SELECT that answers the user's question.
The routing JSON above tells you which tables, columns and filters to use.
Output ONLY the SQL statement — no JSON, no markdown, no comments (no --),
no explanation before or after. The first character of your reply MUST be 'S'.
Start with SELECT or WITH.

═══════════════════════════════════════════════════════════════════════
RULES (each is a hard requirement — break one and the query is rejected)
═══════════════════════════════════════════════════════════════════════

1. MANDATORY TABLES & COLUMNS
   Use ONLY the tables in routing.tables and columns from those tables
   (cross-check against the schema above). Do NOT substitute a different
   table even if one feels closer. The router already picked them.

2. WILAYA SUBQUERY — never inline location_ids
   `dim_location` is COMMUNE-level (a wilaya covers ~25 communes, up to
   60 for Alger/Sétif). Never list commune ids by hand. Use a subquery:

     -- one wilaya
     WHERE <table>.location_id IN (
         SELECT location_id FROM dim_location WHERE wilaya = '<canon>'
     )

     -- several wilayas
     WHERE <table>.location_id IN (
         SELECT location_id FROM dim_location WHERE wilaya IN ('<A>', '<B>')
     )

   Use the EXACT canonical French spellings supplied below.

3. NO HALLUCINATED FILTERS
   Apply ONLY the filters listed in routing.filters. If routing.filters
   .wilayas is empty, do NOT add ANY wilaya/location_id filter — even
   if a previous turn used one. To show every wilaya, JOIN dim_location
   and GROUP BY dl.wilaya.

4. COMPARISON ACROSS WILAYAS
   When routing.filters.wilayas has 2+ names, the SQL MUST JOIN
   dim_location and GROUP BY dl.wilaya so each wilaya gets its own row:

     SELECT dl.wilaya, SUM(g.total_revenue) AS total_revenue
     FROM global_revenue g
     JOIN dim_location dl ON g.location_id = dl.location_id
     WHERE g.location_id IN (
         SELECT location_id FROM dim_location WHERE wilaya IN ('Alger', 'Oran')
     )
     GROUP BY dl.wilaya

5. AGGREGATION — choose by the user's wording AND the column's nature
   - SUM(...)   for totals or amount/count columns: revenue, cost,
                subscribers, capex_*, opex_*, ebitda, net_income, profit,
                or whenever the user says "total", "sum", "how much".
   - AVG(...)   for rates, ratios, scores, percentages, ARPU: churn_rate,
                migration_rate, recharge_rate, ebitda_margin, arpu,
                *_ratio, *_score, *_pct, *_percentage, *_index, or when
                the user says "average", "mean".
   - COUNT(*)   for "how many <rows>" questions on dimension tables.
   When unsure, prefer SUM for amounts and AVG for rates.

6. TIME FILTER — translate routing.filters.time into a WHERE clause.
   Use the date column of the metric table (commonly `week_start`).
   The exact functions to use are listed in the TIME FILTER DIALECT block
   at the end of this instruction — use ONLY those functions, no others.

7. RANKING — superlatives and top-N
   - "highest <X>"   → ORDER BY <X> DESC LIMIT 1
   - "lowest <X>"    → ORDER BY <X> ASC  LIMIT 1
   - "top N by <X>"  → ORDER BY <X> DESC LIMIT N
   - "bottom N <X>"  → ORDER BY <X> ASC  LIMIT N
   - "best <X>"      → ORDER BY <X> DESC LIMIT 1
   - "worst <X>"     → ORDER BY <X> ASC  LIMIT 1

   For "which wilaya had the highest …", GROUP BY dl.wilaya FIRST,
   then ORDER BY the aggregated KPI DESC LIMIT 1.

8. TRENDS — grouped by time
   When the user says "trend", "over time", "by week", "weekly", "plot",
   "month-over-month", OR when `week_start` appears in MANDATORY COLUMNS:
   - SELECT week_start alongside the metric(s).
   - GROUP BY week_start, ORDER BY week_start ASC.
   - Combined with multiple wilayas: GROUP BY dl.wilaya, week_start
     ORDER BY week_start ASC — one row per (wilaya, week).

   Trend example — net income trend, two wilayas over 2025:
     SELECT dl.wilaya, week_start, SUM(f.net_income) AS net_income
     FROM fpa_profitability f
     JOIN dim_location dl ON f.location_id = dl.location_id
     WHERE f.location_id IN (
         SELECT location_id FROM dim_location
         WHERE wilaya IN ('Tlemcen', 'Sétif'))
       AND week_start >= '2025-01-01' AND week_start < '2026-01-01'
     GROUP BY dl.wilaya, week_start
     ORDER BY week_start ASC

9. BREAKDOWN
   "breakdown by X" → GROUP BY X. Include every relevant KPI column
   from routing.columns in the SELECT, each wrapped in its aggregate.

10. LITERALS — put values directly in the WHERE clause; no placeholders,
    no ?'s, no :name binding.

11. DEFAULT AGGREGATION — required unless RULE 8 (trend) applies
    If the user is NOT asking for a trend / listing ("list all" / "show all"),
    ALWAYS aggregate every metric column with SUM() or AVG(). Returning raw
    weekly rows (one per week per commune) is almost never what a business
    user wants — summarise the period instead.

    One location + one metric → one scalar:
      SELECT AVG(gross_margin) AS avg_gross_margin
      FROM fpa_profitability
      WHERE location_id IN (...)
        AND week_start >= '2026-01-01' AND week_start < '2026-04-01'

    Multiple metrics (breakdown) → aggregate each one:
      SELECT SUM(capex_it)      AS total_capex_it,
             SUM(capex_network) AS total_capex_network
      FROM opex_capex
      WHERE week_start >= '2026-01-01' AND week_start < '2027-01-01'

    Multi-wilaya (no wilaya filter) → GROUP BY dl.wilaya:
      SELECT dl.wilaya, AVG(gross_margin) AS avg_gross_margin
      FROM fpa_profitability f
      JOIN dim_location dl ON f.location_id = dl.location_id
      GROUP BY dl.wilaya ORDER BY avg_gross_margin DESC

═══════════════════════════════════════════════════════════════════════
FINAL CHECK before emitting (silently, in your head):
  ☐ Reply starts with SELECT or WITH — first character is 'S', no leading text.
  ☐ Uses only the routed tables.
  ☐ Uses only the exact column names from MANDATORY COLUMNS — no invented variants.
  ☐ All wilaya filters use the subquery pattern with canonical names.
  ☐ No wilaya filter that the user did not ask for.
  ☐ Time filter uses the pre-computed literal dates from TIME FILTER block.
  ☐ ORDER BY + LIMIT present iff the user asked for a ranking.
  ☐ GROUP BY present iff aggregating across a dimension.
  ☐ Metric wrapped in SUM()/AVG() — no bare column selects unless RULE 8 trend.
  ☐ Trend: week_start in SELECT and GROUP BY; ORDER BY week_start ASC.
═══════════════════════════════════════════════════════════════════════"""


def build_router_messages(query: str, schema_prompt: str, knowledge: str,
                          history: str = "", feedback: str = "") -> list[dict]:
    """Assemble the chat messages for the router (phase 1)."""
    parts = [schema_prompt, "", "Reference knowledge:", knowledge]
    if history:
        parts += ["", "Recent conversation (use ONLY for explicit follow-ups "
                      "per RULE 5; for any other new question apply RULE 4 "
                      "ANTI-LEAK and do NOT copy filters from history):",
                  history]
    if feedback:
        parts += ["", f"A previous attempt failed: {feedback}. Re-map the "
                      f"tables and columns more carefully this time."]
    parts += ["", f"User question: {query}"]
    return [
        {"role": "system", "content": ROUTER_SYSTEM},
        {"role": "user", "content": "\n".join(parts)},
    ]


def build_sqlgen_instruction(query: str, routing: dict, entities: dict,
                             schema) -> str:
    """SQL-gen instruction. The model owns aggregation/time/ranking
    decisions via the rules above — no Python helper picks SUM vs AVG.

    We surface three concrete inputs cleanly:
      - the time filter string (if any),
      - the canonical wilaya names (if any),
      - the mandatory table list.
    """
    del query  # the SLM sees the user query separately
    tables = [t for t in routing.get("tables", []) if schema.has_table(t)]
    metric_tables = [t for t in tables if t != "dim_location"]
    wilayas = (entities or {}).get("wilayas", []) or []
    time_phrase = (routing.get("filters") or {}).get("time")
    segment = (routing.get("filters") or {}).get("segment")

    surface_blocks: list[str] = []

    if metric_tables:
        surface_blocks.append(
            "MANDATORY TABLES: "
            + ", ".join(f"`{t}`" for t in metric_tables)
            + " (plus `dim_location` if a wilaya filter or GROUP BY "
              "dl.wilaya is needed). Do NOT use any other table.")

    columns = routing.get("columns", [])
    if columns:
        surface_blocks.append(
            "MANDATORY COLUMNS (exact names — never invent variants): "
            + ", ".join(f"`{c}`" for c in columns)
            + "\n  Use ONLY these column names. E.g. if listed as `migration_rate`, "
              "never write `pmigration_rate` or any other variant.")

    if wilayas:
        canon = ", ".join(f"'{w}'" for w in wilayas)
        surface_blocks.append(
            f"RESOLVED WILAYAS (canonical French spellings — copy these "
            f"EXACTLY into the subquery): {canon}")
    else:
        surface_blocks.append(
            "NO WILAYA was requested by the user. Per RULE 3 do NOT add "
            "any wilaya/location_id filter. If you want to break down by "
            "wilaya, JOIN dim_location and GROUP BY dl.wilaya.")

    # ── dialect-aware time filter ─────────────────────────────────────────
    _sqlite = V6Config.USE_SQLITE
    if _sqlite:
        _time_patterns = """\
  SQLite — use strftime() ONLY (no YEAR/QUARTER/MONTH/CURDATE):
    "2025"          → WHERE strftime('%Y', week_start) = '2025'
    "2024"          → WHERE strftime('%Y', week_start) = '2024'
    "Q1 <year>"     → WHERE strftime('%Y', week_start) = '<year>'
                        AND CAST(strftime('%m', week_start) AS INT) BETWEEN 1 AND 3
    "Q2 <year>"     → WHERE strftime('%Y', week_start) = '<year>'
                        AND CAST(strftime('%m', week_start) AS INT) BETWEEN 4 AND 6
    "Q3 <year>"     → WHERE strftime('%Y', week_start) = '<year>'
                        AND CAST(strftime('%m', week_start) AS INT) BETWEEN 7 AND 9
    "Q4 <year>"     → WHERE strftime('%Y', week_start) = '<year>'
                        AND CAST(strftime('%m', week_start) AS INT) BETWEEN 10 AND 12
    "this year"     → WHERE strftime('%Y', week_start) = strftime('%Y', 'now')
    "last year"     → WHERE strftime('%Y', week_start) = CAST(strftime('%Y','now') AS INT) - 1
    "this month"    → WHERE strftime('%Y-%m', week_start) = strftime('%Y-%m', 'now')
    "last month"    → WHERE strftime('%Y-%m', week_start)
                          = strftime('%Y-%m', date('now', 'start of month', '-1 month'))
    "last quarter"  → WHERE week_start >= date('now', '-3 months')
    "last week"     → WHERE week_start >= date('now', '-7 days')"""
    else:
        _time_patterns = """\
  MySQL — use YEAR/QUARTER/MONTH/CURDATE() ONLY (no strftime):
    "2025"          → WHERE YEAR(week_start) = 2025
    "2024"          → WHERE YEAR(week_start) = 2024
    "Q1 <year>"     → WHERE YEAR(week_start) = <year> AND QUARTER(week_start) = 1
    "Q2 <year>"     → WHERE YEAR(week_start) = <year> AND QUARTER(week_start) = 2
    "Q3 <year>"     → WHERE YEAR(week_start) = <year> AND QUARTER(week_start) = 3
    "Q4 <year>"     → WHERE YEAR(week_start) = <year> AND QUARTER(week_start) = 4
    "this year"     → WHERE YEAR(week_start) = YEAR(CURDATE())
    "last year"     → WHERE YEAR(week_start) = YEAR(CURDATE()) - 1
    "this month"    → WHERE YEAR(week_start)  = YEAR(CURDATE())
                        AND MONTH(week_start) = MONTH(CURDATE())
    "last month"    → WHERE week_start >= DATE_SUB(DATE_FORMAT(CURDATE(),'%Y-%m-01'), INTERVAL 1 MONTH)
                        AND week_start <  DATE_FORMAT(CURDATE(),'%Y-%m-01')
    "last quarter"  → WHERE week_start >= DATE_SUB(CURDATE(), INTERVAL 3 MONTH)
    "last week"     → WHERE week_start >= DATE_SUB(CURDATE(), INTERVAL 7 DAY)"""

    resolved_time = _resolve_time(time_phrase) if time_phrase else None
    if time_phrase:
        if resolved_time:
            surface_blocks.append(
                f"TIME FILTER (pre-computed from \"{time_phrase}\"):\n"
                f"  Apply this EXACT condition in the WHERE clause:\n"
                f"  {resolved_time}\n"
                f"  Do NOT use SQL date functions (strftime, YEAR, MONTH, etc.). "
                f"Paste these literal date strings as-is.")
        else:
            surface_blocks.append(
                f"TIME FILTER DIALECT — DATABASE: {'SQLite' if _sqlite else 'MySQL'}\n"
                f"{_time_patterns}\n\n"
                f"  The user said: \"{time_phrase}\"\n"
                f"  Match it to the pattern above and apply the correct WHERE clause.")
    else:
        surface_blocks.append("NO TIME FILTER was requested — do not invent one.")

    if segment:
        surface_blocks.append(
            f"SEGMENT: the user asked for `{segment}`. The segment is "
            f"encoded by the table name (prepaid_kpi vs postpaid_kpi); "
            f"the routing already picked the correct one — no extra "
            f"WHERE clause for segment is needed.")

    return _SQLGEN_BASE + "\n\n" + "\n\n".join(surface_blocks)


def parse_router_output(text: str) -> dict:
    """Parse the router's JSON. Always returns a dict; `_parse_ok` flags success."""
    default = {
        "intent": "", "tables": [], "columns": [],
        "filters": {"wilayas": [], "segment": None, "time": None},
        "notes": "", "_parse_ok": False,
    }
    if not text:
        return default

    clean = text.strip()
    if "```" in clean:                       # strip markdown fences if present
        clean = clean.replace("```json", "```")
        parts = [p for p in clean.split("```") if p.strip()]
        clean = max(parts, key=len) if parts else clean
    ts, te = clean.find("<think>"), clean.find("</think>")
    if ts >= 0 and te > ts:                  # strip a stray <think> block
        clean = clean[:ts] + clean[te + len("</think>"):]

    a, b = clean.find("{"), clean.rfind("}")
    if a < 0 or b <= a:
        return default
    try:
        obj = json.loads(clean[a:b + 1])
    except Exception:  # noqa: BLE001 — malformed JSON
        return default

    out = dict(default)
    out["_parse_ok"] = True
    out["intent"] = str(obj.get("intent", "")).strip().lower()
    out["tables"] = [str(x) for x in (obj.get("tables") or [])]
    out["columns"] = [str(x) for x in (obj.get("columns") or [])]
    f = obj.get("filters") or {}
    out["filters"] = {
        "wilayas": list(f.get("wilayas") or []),
        "segment": f.get("segment"),
        "time": f.get("time"),
    }
    out["notes"] = str(obj.get("notes", ""))
    return out
