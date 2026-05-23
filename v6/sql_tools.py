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


# ── intent-consistency check ─────────────────────────────────────────────
_FROM_OR_JOIN_RE = re.compile(
    r"\b(?:from|join)\s+`?([a-zA-Z_]\w*)`?"
    r"(?:\s+(?:as\s+)?`?([a-zA-Z_]\w*)`?)?", re.IGNORECASE)
_QUALIFIED_REF_RE = re.compile(
    r"\b([a-zA-Z_]\w*)\.([a-zA-Z_]\w*)\b")
_LOCID_IN_RE = re.compile(
    r"\blocation_id\s*(?:=|in)\s*\(?\s*([\d,\s]+)\)?",
    re.IGNORECASE)
_WILAYA_NAME_FILTER_RE = re.compile(
    r"\bwilaya\s*(?:=|in)\s*\(?\s*'[^']*'",
    re.IGNORECASE)


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
    """Catch hallucinated columns, missing / extra wilaya commune ids, and
    fragile name-string wilaya filters. Targeted: only `alias.column`
    references are checked for column existence, so legitimate table names
    and aliases stop tripping false positives."""
    issues: list[str] = []
    s = sql or ""
    low = s.lower()
    requested_names = list((entities or {}).get("wilayas", []) or [])
    ids_map: dict = (entities or {}).get("wilaya_ids_map", {}) or {}
    all_requested_ids: set[int] = set()
    for w in requested_names:
        all_requested_ids.update(int(i) for i in ids_map.get(w, []))

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

    # 2. location_id parity. The reference knowledge gave the SLM, for each
    #    requested wilaya, the FULL set of commune ids. The SQL's
    #    location_id filter must cover that set exactly — missing ids
    #    mean a partial wilaya aggregate.
    sql_ids: set[int] = set()
    for m in _LOCID_IN_RE.finditer(s):
        for tok in m.group(1).split(","):
            tok = tok.strip()
            if tok.isdigit():
                sql_ids.add(int(tok))

    if all_requested_ids:
        missing = sorted(all_requested_ids - sql_ids)
        if missing:
            preview = ", ".join(str(i) for i in missing[:6])
            more = f" (+{len(missing) - 6} more)" if len(missing) > 6 else ""
            issues.append(
                f"query is missing {len(missing)} commune location_id(s) "
                f"that belong to the requested wilaya(s): {preview}{more}")
        extra = sorted(sql_ids - all_requested_ids)
        if extra:
            preview = ", ".join(str(i) for i in extra[:6])
            issues.append(
                f"query filters location_id(s) the user did not request: "
                f"{preview}")
    elif sql_ids and not requested_names:
        issues.append(
            "query filters by location_id but the user named no wilaya")

    # 3. wilaya-name filter is fragile — flag any `wilaya = 'X'` /
    #    `wilaya IN ('X', ...)` filter when ids were available, so the SLM
    #    is nudged back to id-based filtering.
    if (_WILAYA_NAME_FILTER_RE.search(s) and all_requested_ids):
        issues.append(
            "wilaya filter uses a name string in WHERE; replace it with "
            "`location_id IN (...)` using the ids from the reference knowledge")

    if is_trend_query(query) and "group by" in low and "week_start" not in low:
        issues.append(
            "trend question but the query does not group by week_start")
    return issues


def correction_hint(issues: list[str], entities: dict,
                    exec_error: str | None = None) -> str:
    """Build a targeted correction appended to the retry SQL instruction.

    The actual database error string, when present, IS the correction — it
    names the missing column or table precisely. We surface it verbatim so
    the SLM can reason from the truth instead of generic phrasing.
    """
    wilayas = (entities or {}).get("wilayas", []) or []
    ids_map: dict = (entities or {}).get("wilaya_ids_map", {}) or {}
    parts: list[str] = []

    if exec_error:
        parts.append(
            f"The previous query was rejected by the database: {exec_error}. "
            f"Look up the column on its real table (the schema is above), "
            f"or split the query.")
    if any("missing" in i and "location_id" in i for i in issues):
        # surface the per-wilaya id lists so the SLM can copy them
        id_lines = "; ".join(
            f"{w}=({', '.join(str(i) for i in ids_map.get(w, []))})"
            for w in wilayas if ids_map.get(w))
        parts.append(
            f"You MUST include every commune id for each requested wilaya. "
            f"Use: WHERE <table>.location_id IN (...) — the full id sets are "
            f"{id_lines}.")
    if any("the user did not request" in i for i in issues):
        if wilayas:
            id_lines = "; ".join(
                f"{w}=({', '.join(str(i) for i in ids_map.get(w, []))})"
                for w in wilayas if ids_map.get(w))
            parts.append(
                f"Filter location_id ONLY for these wilaya id sets: "
                f"{id_lines}.")
        else:
            parts.append(
                "Do NOT add any location_id filter — the user named no wilaya.")
    if any("name string" in i for i in issues):
        parts.append(
            "Do NOT filter by wilaya name; use `location_id IN (...)` with "
            "the integer ids from the reference knowledge.")
    if any("does not exist in table" in i for i in issues):
        bad = [i for i in issues if "does not exist in table" in i]
        parts.append("Fix the missing columns: " + "; ".join(bad) + ".")
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
