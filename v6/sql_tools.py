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


# ── intent-consistency check ─────────────────────────────────────────────
_FROM_OR_JOIN_RE = re.compile(
    r"\b(?:from|join)\s+`?([a-zA-Z_]\w*)`?"
    r"(?:\s+(?:as\s+)?`?([a-zA-Z_]\w*)`?)?", re.IGNORECASE)
_QUALIFIED_REF_RE = re.compile(
    r"\b([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)\b")
# Detect inline numeric id lists — a sign the model ignored the subquery rule
_INLINE_LOC_ID_RE = re.compile(
    r"\blocation_id\s*in\s*\(\s*\d", re.IGNORECASE)
# Detect the correct subquery pattern
_SUBQUERY_LOC_RE = re.compile(
    r"\blocation_id\s*in\s*\(\s*select", re.IGNORECASE)
# Extract wilaya names used in any = / IN filter
_WILAYA_EQ_RE = re.compile(
    r"\bwilaya\s*=\s*'((?:[^']|'')*)'", re.IGNORECASE)  # handles SQL '' escaping
_WILAYA_IN_CLAUSE_RE = re.compile(
    r"\bwilaya\s+in\s*\(([^)]*)\)", re.IGNORECASE)


def _build_alias_map(sql: str, schema) -> dict[str, str]:
    """Map every alias *and* table name appearing in FROM/JOIN to its real
    table. Knowing which alias points at which table is what lets us decide
    whether `f.total_revenue` is legal — it depends on which table `f` is."""
    out: dict[str, str] = {}
    if schema is None:
        return out
    sql_keywords = {"on", "where", "group", "order", "having", "limit", "as"}
    for m in _FROM_OR_JOIN_RE.finditer(sql or ""):
        table = m.group(1)
        if not schema.has_table(table):
            continue
        out[table] = table
        alias = m.group(2)
        if alias and alias.lower() not in sql_keywords:
            out[alias] = table
    return out


def consistency_check(sql: str, entities: dict, query: str = "",
                      schema=None) -> list[str]:
    """Catch hallucinated columns and wilaya filter mistakes.

    Two checks:
    1. Alias.column hallucinations — any `alias.col` reference where `col`
       does not exist in the table the alias points to.
    2. Inline id lists — if the model wrote `location_id IN (1, 2, 3, ...)`
       instead of the required subquery, flag it so the retry uses the
       correct subquery pattern.
    3. Non-canonical wilaya names — if the SQL contains `wilaya = 'X'` and X
       does not match the canonical French spelling, flag it.
    """
    issues: list[str] = []
    s = sql or ""
    requested_names = list((entities or {}).get("wilayas", []) or [])

    # 1. hallucinated columns: for every `alias.col` reference, the column
    #    must exist in the table the alias points to.
    if schema is not None:
        alias_map = _build_alias_map(s, schema)
        for m in _QUALIFIED_REF_RE.finditer(s):
            alias, col = m.group(1), m.group(2)
            if alias in alias_map:
                table = alias_map[alias]
                if col not in schema.column_names(table):
                    issues.append(
                        f"column '{col}' does not exist in table '{table}'")

    # 2. inline location_id list — model should use a subquery, not hard-coded ids
    if requested_names and _INLINE_LOC_ID_RE.search(s) and not _SUBQUERY_LOC_RE.search(s):
        canon = ", ".join(f"'{w}'" for w in requested_names)
        issues.append(
            f"wilaya filter uses an inline location_id number list; "
            f"use a subquery instead: WHERE <table>.location_id IN "
            f"(SELECT location_id FROM dim_location WHERE wilaya IN ({canon}))")

    # 2b. hallucinated wilaya filter — model added a location_id/wilaya subquery
    #     when the user named no wilaya at all
    if not requested_names and (_SUBQUERY_LOC_RE.search(s) or _INLINE_LOC_ID_RE.search(s)):
        issues.append(
            "query filters by wilaya/location_id but the user named no specific "
            "wilaya; remove the filter and use GROUP BY dl.wilaya to show all wilayas")

    # 3. non-canonical wilaya name — extract names the SQL uses in wilaya filters.
    # We unescape SQL double-single-quotes (M''Sila → M'Sila) before comparing so
    # that correctly escaped apostrophe names never trigger a false-positive.
    if requested_names:
        canonical_set = set(requested_names)
        sql_wilaya_names: list[str] = []
        for m in _WILAYA_EQ_RE.finditer(s):
            sql_wilaya_names.append(m.group(1).replace("''", "'"))
        for m in _WILAYA_IN_CLAUSE_RE.finditer(s):
            for part in m.group(1).split(","):
                name = part.strip().strip("'").strip('"').replace("''", "'")
                if name:
                    sql_wilaya_names.append(name)
        for name in sql_wilaya_names:
            if name not in canonical_set:
                issues.append(
                    f"SQL uses wilaya name '{name}' but the canonical DB "
                    f"spelling is {', '.join(repr(w) for w in requested_names)}. "
                    f"Copy the canonical name exactly into the subquery.")

    del query  # shape decisions left to the SLM
    return issues


def correction_hint(issues: list[str], entities: dict,
                    exec_error: str | None = None) -> str:
    """Build a targeted correction appended to the retry SQL instruction.

    The actual database error string, when present, IS the correction — it
    names the missing column or table precisely. We surface it verbatim so
    the SLM can reason from the truth instead of generic phrasing.
    """
    wilayas = (entities or {}).get("wilayas", []) or []
    parts: list[str] = []

    if exec_error:
        parts.append(
            f"The previous query was rejected by the database: {exec_error}. "
            f"Look up the column on its real table (the schema is above), "
            f"or split the query.")
    if any("inline location_id" in i for i in issues):
        canon = ", ".join(f"'{w}'" for w in wilayas)
        parts.append(
            f"Do NOT list location_ids by hand. Use this subquery pattern: "
            f"WHERE <table>.location_id IN "
            f"(SELECT location_id FROM dim_location WHERE wilaya IN ({canon})).")
    if any("named no specific wilaya" in i for i in issues):
        parts.append(
            "The user named NO specific wilaya. Remove the location_id/wilaya "
            "filter completely. Instead JOIN dim_location and GROUP BY dl.wilaya "
            "to return one row per wilaya.")
    if any("canonical DB spelling" in i for i in issues):
        # SQL-escape apostrophes: M'Sila → 'M''Sila' so the hint shows
        # valid SQL that the model can copy verbatim.
        def _sql_lit(w: str) -> str:
            return "'" + w.replace("'", "''") + "'"
        canon = ", ".join(_sql_lit(w) for w in wilayas)
        parts.append(
            f"Use the exact canonical French wilaya spelling(s): {canon}. "
            f"These go in: WHERE wilaya = '...' or WHERE wilaya IN (...).")
    if any("does not exist in table" in i for i in issues):
        bad = [i for i in issues if "does not exist in table" in i]
        parts.append("Fix the missing columns: " + "; ".join(bad) + ".")
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
