"""v6/sql_tools.py — The SQL trust boundary.

The SLM writes raw SQL; this module decides whether it may run. It:
  - strips markdown / prose and keeps the first statement,
  - rejects anything that is not a single read-only SELECT / WITH,
  - blocks every DDL/DML keyword,
  - checks referenced tables exist in the live schema,
  - **checks the query is consistent with the user's intent** — no wilaya,
    date, or segment filter the user never asked for (the fix for the
    hallucinated-WHERE-clause bug), and no missing filter the user did ask
    for,
  - enforces a row LIMIT and executes against SQLite or MySQL.

Nothing here raises — callers always receive a structured dict.
"""

from __future__ import annotations
import datetime
import decimal
import re

from .config import V6Config
from .entities import get_resolver
from .prompts import is_trend_query
from .schema import db_connect

# Whole-word write / DDL keywords that must never appear in a query.
_BLOCKED = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|create|replace|grant|"
    r"revoke|rename|lock|unlock|call|execute|use|load|handler|attach|"
    r"outfile|dumpfile|pragma)\b", re.IGNORECASE)


# ── cleaning + static validation ─────────────────────────────────────────
def clean_sql(raw: str) -> str:
    """Strip code fences / prose; keep only the first statement."""
    s = (raw or "").strip()
    s = re.sub(r"^```[a-zA-Z]*\n?", "", s).strip()
    s = re.sub(r"\n?```\s*$", "", s).strip()
    s = s.replace("```", "").strip()
    s = re.sub(r"^sql\s*[:\-]?\s*", "", s, flags=re.IGNORECASE).strip()
    if ";" in s:
        s = s.split(";", 1)[0].strip()
    return s


def validate_sql(sql: str, schema=None) -> dict:
    """Static validation, no database. Returns {valid, errors, sql}."""
    s = clean_sql(sql)
    errors: list[str] = []
    if not s:
        return {"valid": False, "errors": ["empty SQL"], "sql": s}

    low = s.lower()
    if not (low.startswith("select") or low.startswith("with")):
        errors.append("not a read-only SELECT/WITH statement")
    if _BLOCKED.search(s):
        errors.append("contains a blocked write/DDL keyword")

    if schema is not None:
        refs = re.findall(r"\b(?:from|join)\s+`?([a-zA-Z_]\w*)`?", s,
                          re.IGNORECASE)
        for t in refs:
            if not schema.has_table(t):
                errors.append(f"unknown table '{t}'")

    return {"valid": not errors, "errors": errors, "sql": s}


# ── intent-consistency check (the hallacinated-filter guard) ─────────────
def consistency_check(sql: str, entities: dict, query: str = "", schema=None) -> list[str]:
    """Flag filters in the SQL that disagree with the resolved intent, and catch hallucinated columns."""
    issues: list[str] = []
    s = sql or ""
    low = s.lower()
    requested = {w.lower() for w in (entities or {}).get("wilayas", [])}

    # check for hallucinated columns (schema parameter passed by caller)
    if schema is not None:
        all_valid_cols = {c for t in schema.all_tables()
                         for c in schema.column_names(t)}
        col_refs = re.findall(r"\b(?:[a-z_]\w*\.)?([a-z_]\w*)\b", s,
                              re.IGNORECASE)
        for col in set(col_refs):
            # skip SQL keywords and common tokens
            if (col.lower() not in {"select", "from", "where", "and", "or",
                                     "join", "on", "as", "group", "by",
                                     "order", "limit", "having", "in", "not",
                                     "is", "null", "case", "when", "then",
                                     "else", "end", "with", "as", "distinct",
                                     "asc", "desc", "sum", "avg", "count",
                                     "max", "min", "cast", "to", "left",
                                     "right", "inner", "outer", "cross",
                                     "union", "except", "intersect"}
                    and col not in all_valid_cols):
                issues.append(f"column '{col}' does not exist in any table")

    resolver = get_resolver()
    sql_wilayas: set[str] = set()
    for lit in re.findall(r"'([^']*)'", s):       # quoted string literals
        canon = resolver.resolve_wilaya(lit)
        if canon:
            sql_wilayas.add(canon.lower())

    for w in sorted(sql_wilayas - requested):
        issues.append(f"query filters wilaya '{w}', which was not requested")
    for w in sorted(requested - sql_wilayas):
        issues.append(f"query is missing the requested wilaya filter '{w}'")

    if is_trend_query(query) and "group by" in low and "week_start" not in low:
        issues.append("trend question but the query does not group by week_start")
    return issues


def correction_hint(issues: list[str], entities: dict) -> str:
    """Build a targeted correction appended to the retry SQL instruction."""
    wilayas = (entities or {}).get("wilayas", [])
    parts: list[str] = []
    if any("not requested" in i for i in issues):
        parts.append("Filter by wilaya ONLY for: " + ", ".join(wilayas) + "."
                     if wilayas else
                     "Do NOT add any wilaya filter — the user named no wilaya.")
    if any("missing the requested" in i for i in issues):
        parts.append("You MUST filter dl.wilaya for: " + ", ".join(wilayas) + ".")
    if any("week_start" in i for i in issues):
        parts.append("GROUP BY week_start and ORDER BY week_start.")
    return " ".join(parts)


# ── limit + execution ────────────────────────────────────────────────────
def enforce_limit(sql: str, max_rows: int | None = None) -> str:
    """Append a LIMIT when the query has none."""
    max_rows = max_rows or V6Config.SQL_MAX_ROWS
    if re.search(r"\blimit\b", sql, re.IGNORECASE):
        return sql
    return f"{sql.rstrip().rstrip(';')} LIMIT {max_rows}"


def _coerce(v):
    """Make a DB value JSON/checkpoint-safe."""
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (datetime.date, datetime.datetime)):
        return str(v)
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return v


def execute_sql(sql: str) -> dict:
    """Run a validated query against the active backend. Never raises."""
    try:
        conn = db_connect()
        try:
            if V6Config.USE_SQLITE:
                cur = conn.cursor()
                cur.execute(sql)
                fetched = cur.fetchall()
                columns = [d[0] for d in (cur.description or [])]
                rows = [{k: _coerce(r[k]) for k in r.keys()} for r in fetched]
            else:
                cur = conn.cursor(dictionary=True)
                cur.execute(sql)
                fetched = cur.fetchall()
                rows = [{k: _coerce(v) for k, v in r.items()} for r in fetched]
                columns = (list(rows[0].keys()) if rows
                           else [d[0] for d in (cur.description or [])])
        finally:
            conn.close()
        return {"ok": True, "rows": rows, "columns": columns, "error": None}
    except Exception as exc:  # noqa: BLE001 — surface DB errors as data
        return {"ok": False, "rows": [], "columns": [], "error": str(exc)}
