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

from .config import V6Config


def _last_data_turn(turns: list[dict]) -> dict | None:
    for turn in reversed(turns or []):
        if turn.get("intent") == "data" and turn.get("tables"):
            return turn
    return None


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

    # follow-up: inherit tables / columns from the last data turn
    inherited = False
    if not valid_tables:
        last = _last_data_turn(turns)
        if last:
            valid_tables = [t for t in last.get("tables", [])
                            if schema.has_table(t)]
            valid_cols = valid_cols or list(last.get("columns", []))
            inherited = bool(valid_tables)
            if inherited:
                trace.append(f"inherited tables from memory {valid_tables}")
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

    # confidence — does the evidence support a data query?
    metric_tables = [t for t in valid_tables if t != "dim_location"]
    if metric_tables and grounding >= V6Config.RAG_LOW_CONF:
        confidence = "high"
    elif metric_tables or grounding >= V6Config.RAG_LOW_CONF or inherited:
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
