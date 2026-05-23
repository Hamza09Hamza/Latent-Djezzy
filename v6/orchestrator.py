"""v6/orchestrator.py — Deterministic validation + plan assembly.

The *decision* of what a query wants belongs to the policy brain
(brain.py). What stays here is deliberately deterministic — because it is
fact-checking, not guessing:

  - validate the router SLM's tables/columns against the live schema
    (a model must never decide whether a column exists; that guess was the
    `wilaya` bug),
  - inject dim_location when a wilaya filter needs a join,
  - inherit tables from the last data turn for a follow-up,
  - judge confidence and send a hopeless query to `clarify`,
  - assemble the ordered plan.

Dynamic decision (the brain) + deterministic fact-check (here) is the core.
This module runs inside the `sql` action, which the brain selects only for
data queries; greeting / meta / definition / unanswerable never reach it.
"""

from __future__ import annotations
import re

from .config import V6Config


def _last_data_turn(turns: list[dict]) -> dict | None:
    for turn in reversed(turns or []):
        if turn.get("intent") == "data" and turn.get("tables"):
            return turn
    return None


def _is_implicit_followup(query: str, valid_cols: list[str],
                          valid_tables: list[str], schema) -> bool:
    """True when the query is so short and KPI-free that it's almost certainly
    a follow-up like 'and for Constantine?' or 'what about Oran?'.

    "KPI-free" is decided from the schema: a real metric column is one that
    appears as numeric in some metric table. No hand-curated structural list.
    """
    words = re.sub(r"[^\w\s]", " ", query or "").split()
    numeric_in_scope: set[str] = set()
    for t in valid_tables:
        if schema is not None and schema.has_table(t):
            numeric_in_scope.update(schema.numeric_columns(t))
    metric_cols = [c for c in valid_cols if c in numeric_in_scope]
    return len(words) <= 6 and not metric_cols


def assemble(query: str, routing: dict, capabilities: list[str],
             followup: bool, grounding: float, turns: list[dict],
             schema) -> dict:
    """Validate the router's schema mapping and build the final data plan."""
    trace: list[str] = []
    caps = list(capabilities or [])

    # validate the router's tables / columns against the live schema
    valid_tables = [t for t in routing.get("tables", []) if schema.has_table(t)]
    dropped = [t for t in routing.get("tables", []) if not schema.has_table(t)]
    if dropped:
        trace.append(f"dropped unknown tables {dropped}")
    all_cols = {c for t in schema.all_tables() for c in schema.column_names(t)}
    valid_cols = [c for c in routing.get("columns", []) if c in all_cols]
    invalid_cols = [c for c in routing.get("columns", []) if c not in all_cols]
    if invalid_cols:
        trace.append(f"dropped unknown columns {invalid_cols}")

    # follow-up: inherit tables / columns from the last data turn.
    # Fire when (a) no valid tables at all, OR (b) the query is short and
    # the router produced only non-metric columns — a sign it grabbed the
    # wrong table to fill in a follow-up like "and for Constantine?".
    inherited = False
    force_inherit = _is_implicit_followup(query, valid_cols,
                                          valid_tables, schema)
    if not valid_tables or force_inherit:
        last = _last_data_turn(turns)
        if last:
            inherited_tables = [t for t in last.get("tables", [])
                                if schema.has_table(t)]
            if inherited_tables:
                valid_tables = inherited_tables
                valid_cols = list(last.get("columns", []))
                inherited = True
                trace.append(f"inherited tables from memory {valid_tables}"
                             + (" (implicit follow-up)" if force_inherit else ""))
    elif followup:
        trace.append("follow-up detected")

    # a wilaya filter needs dim_location alongside any metric table
    filters = routing.get("filters", {}) or {}
    if filters.get("wilayas") and valid_tables:
        if (any(schema.needs_location_join(t) for t in valid_tables)
                and "dim_location" not in valid_tables):
            valid_tables.append("dim_location")
            trace.append("added dim_location for the wilaya filter")

    routing_v = dict(routing)
    routing_v["tables"] = valid_tables
    routing_v["columns"] = valid_cols

    # confidence — does the KPI evidence support a data query?
    # `grounding` is now the max cosine among non-wilaya chunks (see
    # knowledge_block). A router-hallucinated table with only a location
    # hit (grounding near 0) must not be treated as answerable.
    metric_tables = [t for t in valid_tables if t != "dim_location"]
    _floor = V6Config.RAG_LOW_CONF * 0.7   # 0.315 at default threshold
    if metric_tables and grounding >= V6Config.RAG_LOW_CONF:
        confidence = "high"
    elif grounding >= V6Config.RAG_LOW_CONF or inherited:
        confidence = "medium"
    elif metric_tables and grounding >= _floor:
        # Router found a table but KPI grounding is weak — let it try,
        # but don't mark it high so the brain can bail after one attempt.
        confidence = "medium"
    else:
        confidence = "low"
    trace.append(f"confidence={confidence} (grounding={grounding:.3f})")

    # plan
    if confidence == "low":
        plan = ["clarify"]
        caps = []
        trace.append("→ clarify: no resolvable metric table")
    else:
        plan = ["sql"] + caps

    return {
        "routing": routing_v,
        "capabilities": caps,
        "plan": plan,
        "confidence": confidence,
        "trace": trace,
    }
