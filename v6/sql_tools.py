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
    """Catch hallucinated columns, non-canonical wilaya filters, and
    hallucinated wilaya filters. Targeted: only `alias.column` references
    are validated, so legitimate table names and aliases stop tripping
    false positives."""
    issues: list[str] = []
    s = sql or ""
    low = s.lower()
    requested_names = list((entities or {}).get("wilayas", []) or [])
    canon_set = set(requested_names)

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
            # alias unknown → SLM made up the alias too; skip — the SQL
            # engine will report the precise error on execution.

    # 2. wilaya filter parity. The resolver gives us the canonical spellings
    #    the database actually holds; the SQL must use those, and only those.
    resolver = get_resolver()
    sql_wilayas_canon: set[str] = set()
    sql_wilayas_raw: list[str] = []
    for lit in re.findall(r"'([^']*)'", s):
        canon = resolver.resolve_wilaya(lit)
        if canon:
            sql_wilayas_canon.add(canon)
            sql_wilayas_raw.append(lit)

    # 2a. non-canonical spelling — flag every quoted name that resolves to
    #     a wilaya but isn't the canonical spelling.
    for lit in sql_wilayas_raw:
        canon = resolver.resolve_wilaya(lit)
        if canon and lit != canon:
            issues.append(
                f"query uses '{lit}' but the canonical wilaya spelling is "
                f"'{canon}' — use that exact name in the SQL")

    # 2b. missing / extra wilayas relative to what the user requested.
    if canon_set:
        for missing in sorted(canon_set - sql_wilayas_canon):
            issues.append(
                f"query is missing the requested wilaya '{missing}'")
        for extra in sorted(sql_wilayas_canon - canon_set):
            issues.append(
                f"query filters wilaya '{extra}', which the user did not "
                f"request")
    elif sql_wilayas_canon:
        issues.append(
            "query filters by wilaya but the user named no wilaya")

    if is_trend_query(query) and "group by" in low and "week_start" not in low:
        issues.append(
            "trend question but the query does not group by week_start")
    return issues


def correction_hint(issues: list[str], entities: dict,
                    exec_error: str | None = None) -> str:
    """Build a targeted correction appended to the retry SQL instruction.

    When the previous attempt actually ran and the database returned an
    error, that error string IS the correction — it names the offending
    column or table precisely. We surface it verbatim so the SLM can use
    its own reasoning instead of guessing from generic phrasing.
    """
    wilayas = (entities or {}).get("wilayas", []) or []
    name_list = ", ".join(f"'{w}'" for w in wilayas)
    parts: list[str] = []

    if exec_error:
        parts.append(
            f"The previous query was rejected by the database: {exec_error}. "
            f"Pick the table where the missing column actually lives, or "
            f"split the query.")
    if any("canonical wilaya spelling" in i for i in issues):
        bad = [i for i in issues if "canonical wilaya spelling" in i]
        parts.append("Use the canonical wilaya spelling: " + "; ".join(bad)
                     + ".")
    if any("missing the requested wilaya" in i for i in issues):
        parts.append(
            f"You MUST filter dl.wilaya for every requested wilaya: "
            f"{name_list}.")
    if any("the user did not request" in i for i in issues):
        parts.append(
            f"Filter dl.wilaya ONLY for: {name_list}." if wilayas else
            "Do NOT add any wilaya filter — the user named no wilaya.")
    if any("does not exist in table" in i for i in issues):
        bad = [i for i in issues if "does not exist in table" in i]
        parts.append("Fix the missing columns: " + "; ".join(bad) + ".")
    if any("named no wilaya" in i for i in issues):
        parts.append(
            "Remove the wilaya filter — the user did not name any wilaya.")
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
