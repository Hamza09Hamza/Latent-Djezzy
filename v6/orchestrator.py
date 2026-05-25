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
    """Validate the router's schema mapping and build the final data plan.

    Follow-up detection is the router's responsibility (RULE 5 in its
    system prompt). Here we only fall back to inheriting tables/columns
    when the router returned nothing schema-valid — a structural rescue,
    not a heuristic on the query text.
    """
    del query  # kept for API stability; not inspected here anymore
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

    # Structural rescue: if the router gave us no valid tables at all
    # (or only non-metric ones), pull tables+columns from the last data
    # turn. We do NOT inspect the query text — the router's prompt is
    # the only place follow-ups are decided.
    inherited = False
    metric_valid = [t for t in valid_tables if t != "dim_location"]
    if not metric_valid:
        last = _last_data_turn(turns)
        if last:
            inherited_tables = [t for t in last.get("tables", [])
                                if schema.has_table(t)]
            if inherited_tables:
                valid_tables = inherited_tables
                valid_cols = list(last.get("columns", []))
                inherited = True
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

    # confidence — does the KPI evidence support a data query?
    # `grounding` is the max cosine among non-wilaya RAG chunks. Short follow-
    # ups ("and for Tiaret?") carry no KPI keywords, so grounding is near-zero
    # even when the router correctly inherited the KPI from context. If the
    # router produced BOTH a metric table AND an actual KPI column (not just
    # dimension columns like wilaya / week_start), trust that mapping: the
    # router's RULE 5 did the follow-up inheritance work, not us.
    metric_tables = [t for t in valid_tables if t != "dim_location"]
    _dim_only = {"wilaya", "location_id", "week_start", "month_start",
                 "commune_id", "commune", "region"}
    _kpi_cols = [c for c in valid_cols if c not in _dim_only]
    _floor = V6Config.RAG_LOW_CONF * 0.7   # 0.315 at default threshold
    if metric_tables and _kpi_cols:
        # Router gave us a metric table + a real KPI column → trust the mapping.
        # High when RAG also confirms it; medium otherwise (handles follow-ups).
        confidence = "high" if grounding >= V6Config.RAG_LOW_CONF else "medium"
    elif grounding >= V6Config.RAG_LOW_CONF or inherited:
        confidence = "medium"
    elif metric_tables and grounding >= _floor:
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
