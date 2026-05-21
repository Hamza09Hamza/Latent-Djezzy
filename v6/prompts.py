"""v6/prompts.py — Router and SQL-generator prompts, plus output parsing.

The router is constrained to emit a single JSON object for EVERY question,
greetings included. That is the fix for the misrouting bug: v5 told the
model to reply in free text for non-data questions and then keyword-grepped
the reply — so "Hello!", containing no literal word "greeting", fell through
to the data path and generated SQL. Here the model must emit
{"intent": "greeting"}, which parses unambiguously.

The SQL prompt carries one uniform join rule (every metric table joins
dim_location via location_id, no exceptions) and an explicit instruction to
apply only the filters in the routing object — which closes the
hallucinated-WHERE-clause bug.
"""

from __future__ import annotations
import json

VALID_INTENTS = ("data", "definition", "greeting", "meta", "unanswerable")

ROUTER_SYSTEM = """You are the router of a telecom analytics assistant.
Read the user's question and reply with ONE JSON object — nothing before it,
nothing after it.

The JSON object has exactly these fields:
  "intent"  : one of
       "data"         - needs numbers from the database
       "definition"   - asks what a term or KPI means
       "greeting"     - hello, thanks, or small talk
       "meta"         - asks what you are or what you can do
       "unanswerable" - needs a KPI or table that is not in the schema
  "tables"  : list of schema tables the query needs (data intent only)
  "columns" : list of schema columns the query needs (data intent only)
  "filters" : {"wilayas": [...], "segment": "prepaid"|"postpaid"|null,
               "time": "<relative-time phrase>"|null}
  "notes"   : one short sentence explaining your table/column mapping

Rules:
- Reply with the JSON object ONLY. No prose, no markdown fences, no <think>.
- Never invent table or column names — use only the schema given below.
- Map synonyms via the reference knowledge (sales / turnover -> total_revenue).
- A wilaya filter always needs dim_location: include it in "tables" and put
  "wilaya" in "columns".
- For greeting / meta / definition / unanswerable, leave tables, columns and
  filter lists empty.

Examples:

User question: hi there
{"intent": "greeting", "tables": [], "columns": [], "filters": {"wilayas": [], "segment": null, "time": null}, "notes": "small talk"}

User question: what does ARPU mean
{"intent": "definition", "tables": [], "columns": [], "filters": {"wilayas": [], "segment": null, "time": null}, "notes": "asks the meaning of ARPU"}

User question: what can you do
{"intent": "meta", "tables": [], "columns": [], "filters": {"wilayas": [], "segment": null, "time": null}, "notes": "asks about the assistant"}

User question: total revenue in Oran
{"intent": "data", "tables": ["global_revenue", "dim_location"], "columns": ["total_revenue", "wilaya"], "filters": {"wilayas": ["Oran"], "segment": null, "time": null}, "notes": "revenue is global_revenue.total_revenue; join dim_location to filter the wilaya Oran"}

User question: average churn for prepaid in Algiers
{"intent": "data", "tables": ["prepaid_kpi", "dim_location"], "columns": ["churn_rate", "wilaya"], "filters": {"wilayas": ["Algiers"], "segment": "prepaid", "time": null}, "notes": "churn_rate in prepaid_kpi; join dim_location for the wilaya"}

User question: show the weekly arpu trend for prepaid
{"intent": "data", "tables": ["prepaid_kpi"], "columns": ["arpu", "week_start"], "filters": {"wilayas": [], "segment": "prepaid", "time": null}, "notes": "arpu over time from prepaid_kpi grouped by week_start, no wilaya filter"}

User question: what is the fpa_quantum_score
{"intent": "unanswerable", "tables": [], "columns": [], "filters": {"wilayas": [], "segment": null, "time": null}, "notes": "fpa_quantum_score is not in the schema"}"""

_SQLGEN_BASE = """PHASE 2 - SQL GENERATION.

You now switch from routing to SQL. Write ONE read-only SQL SELECT that
answers the user's question.

JOIN RULE: every metric table (prepaid_kpi, postpaid_kpi, global_revenue,
fpa_profitability, opex_capex) keys location by `location_id`. To filter or
group by a wilaya you MUST join dim_location:
    JOIN dim_location dl ON <table>.location_id = dl.location_id
and then use dl.wilaya. No metric table has a `wilaya` column of its own.

Rules:
- Use ONLY tables and columns from the schema and the routing analysis above.
- Apply ONLY the filters from the routing object. Do NOT add a wilaya, date
  or segment the user did not ask for.
- For a "trend" or "over time" question, GROUP BY week_start and ORDER BY
  week_start; do not group by wilaya unless the user compares wilayas.
- Put literal values directly in the WHERE clause; no placeholders.
- Output ONLY the SQL statement: no JSON, no markdown, no comment.
- Start the output with SELECT or WITH."""

# Columns that describe structure, not a measured KPI.
_STRUCTURAL = {"wilaya", "location_id", "week_start", "id", "commune",
               "wilaya_code", "region", "code"}
_TREND_WORDS = ("trend", "over time", "evolution", "weekly", "week by week",
                "history", "historical", "timeline", "progression", "evolve")


def is_trend_query(query: str) -> bool:
    q = (query or "").lower()
    return any(w in q for w in _TREND_WORDS)


def build_router_messages(query: str, schema_prompt: str, knowledge: str,
                          history: str = "", feedback: str = "") -> list[dict]:
    """Assemble the chat messages for the router (phase 1)."""
    parts = [schema_prompt, "", "Reference knowledge:", knowledge]
    if history:
        parts += ["", "Recent conversation (oldest first, context only):",
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
    """SQL-gen instruction with one concrete, schema-correct example.

    The example is built from the *resolved* entities, so it never injects a
    wilaya the user did not ask for, and it matches the query shape (trend
    vs comparison) so the model does not group by wilaya for a time series.
    """
    tables = [t for t in routing.get("tables", [])
              if schema.has_table(t) and t != "dim_location"]
    cols = [c for c in routing.get("columns", []) if c not in _STRUCTURAL]
    wilayas = (entities or {}).get("wilayas", [])

    if not tables:
        return _SQLGEN_BASE

    t = tables[0]
    alias = t[0]
    kpis = cols[:2] or schema.numeric_columns(t)[:1] or ["*"]
    agg = ", ".join(f"AVG({alias}.{c}) AS {c}" for c in kpis if c != "*")
    join = (f"JOIN dim_location dl ON {alias}.location_id = dl.location_id"
            if schema.needs_location_join(t) else "")

    if is_trend_query(query):
        where = (f" WHERE dl.wilaya = '{wilayas[0]}'"
                 if wilayas and join else "")
        example = (f"SELECT {alias}.week_start, {agg or alias + '.*'} "
                   f"FROM {t} {alias} {join}{where} "
                   f"GROUP BY {alias}.week_start ORDER BY {alias}.week_start")
        shape = "a weekly trend"
    elif join:
        where = (" WHERE dl.wilaya IN ("
                 + ", ".join(f"'{w}'" for w in wilayas) + ")"
                 if wilayas else "")
        example = (f"SELECT dl.wilaya, {agg or alias + '.*'} "
                   f"FROM {t} {alias} {join}{where} GROUP BY dl.wilaya")
        shape = "an aggregate by wilaya"
    else:
        example = f"SELECT {agg or '*'} FROM {t} {alias}"
        shape = "an aggregate"

    return (_SQLGEN_BASE
            + f"\n\nThis question needs {shape}. A correct query shape is:\n"
            + f"  {example}\n"
            + "Adapt it to the user's exact question. Keep the join rule and "
            + "add only the filters listed in the routing object.")


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
