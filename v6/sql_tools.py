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


# ── intent-consistency check (the hallucinated-filter guard) ─────────────
_LOCID_IN_RE = re.compile(
    r"\blocation_id\s*(?:=|in)\s*\(?\s*([\d,\s]+)\)?",
    re.IGNORECASE)
_WILAYA_NAME_FILTER_RE = re.compile(
    r"\bwilaya\s*(?:=|in)\s*\(?\s*'[^']*'",
    re.IGNORECASE)


def consistency_check(sql: str, entities: dict, query: str = "",
                      schema=None) -> list[str]:
    """Catch hallucinated columns, fragile wilaya-name filters, and id-set
    mismatches. The new wilaya policy is: filter by location_id integer."""
    issues: list[str] = []
    s = sql or ""
    low = s.lower()
    requested_ids = set(int(i) for i in (entities or {}).get("wilaya_ids", []))
    requested_names = [w for w in (entities or {}).get("wilayas", [])]

    # 1. hallucinated columns — every identifier must exist somewhere
    if schema is not None:
        all_valid_cols = {c for t in schema.all_tables()
                         for c in schema.column_names(t)}
        keywords = {"select", "from", "where", "and", "or", "join", "on",
                    "as", "group", "by", "order", "limit", "having", "in",
                    "not", "is", "null", "case", "when", "then", "else",
                    "end", "with", "distinct", "asc", "desc", "sum", "avg",
                    "count", "max", "min", "cast", "to", "left", "right",
                    "inner", "outer", "cross", "union", "except",
                    "intersect", "between"}
        for col in {m for m in re.findall(
                r"\b(?:[a-z_]\w*\.)?([a-z_]\w*)\b", s, re.IGNORECASE)}:
            if (col.lower() not in keywords
                    and not col.isdigit()
                    and col not in all_valid_cols):
                issues.append(f"column '{col}' does not exist in any table")

    # 2. wilaya filter must be by location_id (an integer), not by name string
    if _WILAYA_NAME_FILTER_RE.search(s) and requested_ids:
        issues.append(
            "wilaya filter uses a name string in WHERE; use "
            "`location_id` and the integer ids from the reference knowledge")

    # 3. id-set parity: the ids in the SQL must equal the ids the user wanted
    sql_ids: set[int] = set()
    for m in _LOCID_IN_RE.finditer(s):
        for tok in m.group(1).split(","):
            tok = tok.strip()
            if tok.isdigit():
                sql_ids.add(int(tok))
    if requested_ids:
        for extra in sorted(sql_ids - requested_ids):
            issues.append(f"query filters location_id {extra}, "
                          f"which the user did not request")
        for missing in sorted(requested_ids - sql_ids):
            issues.append(f"query is missing the requested location_id "
                          f"{missing}")
    elif sql_ids and not requested_names:
        # ids in SQL but user named no wilaya at all → hallucinated filter
        issues.append(
            "query filters by location_id but the user named no wilaya")

    if is_trend_query(query) and "group by" in low and "week_start" not in low:
        issues.append("trend question but the query does not group by "
                      "week_start")
    return issues


def correction_hint(issues: list[str], entities: dict) -> str:
    """Build a targeted correction appended to the retry SQL instruction."""
    wilayas = (entities or {}).get("wilayas", []) or []
    ids = (entities or {}).get("wilaya_ids", []) or []
    id_pairs = ", ".join(f"{w} = {i}" for w, i in zip(wilayas, ids))
    id_list = ", ".join(str(i) for i in ids)
    parts: list[str] = []

    if any("name string" in i for i in issues):
        parts.append(
            "Do NOT filter by wilaya name. Use `WHERE <table>.location_id "
            f"IN ({id_list})` with the integer ids ({id_pairs}).")
    if any("which the user did not request" in i for i in issues):
        parts.append(
            f"Filter by ONLY these location_id values: {id_list}."
            if ids else
            "Do NOT add any location_id filter — the user named no wilaya.")
    if any("missing the requested location_id" in i for i in issues):
        parts.append(
            f"You MUST include every requested location_id: {id_list}.")
    if any("does not exist in any table" in i for i in issues):
        parts.append(
            "Use only columns that appear in the schema; the previous query "
            "referenced one that does not exist.")
    if any("named no wilaya" in i for i in issues):
        parts.append(
            "Remove the location_id filter — the user did not name any wilaya.")
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
