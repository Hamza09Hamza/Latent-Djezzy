"""v6/state.py — The LangGraph state object.

Every node receives this dict and returns a partial update. Fields are
grouped by pipeline stage. `total=False` means a node only writes the keys
it owns. List fields are never mutated in place — a node returns a fresh
list — so LangGraph's replace-on-update semantics stay predictable.

`turns` is the cross-turn memory: the checkpointer persists it per
`thread_id`, so a follow-up like "and for Oran?" can see what came before.
"""

from __future__ import annotations
from typing import TypedDict


class AgentState(TypedDict, total=False):
    # ── input ────────────────────────────────────────────────────────────
    query: str
    thread_id: str

    # ── conversation memory (persisted by the checkpointer) ──────────────
    turns: list[dict]            # [{query, intent, answer, sql}, ...]
    last_rows: list[dict]        # rows from the previous data turn
    last_columns: list[str]
    carried_entities: dict       # entities resolved on the previous turn

    # ── retrieval ────────────────────────────────────────────────────────
    knowledge: str               # formatted RAG context block
    grounding: float             # top cosine score [0, 1]

    # ── planning (latent planner + router SLM → orchestrator) ────────────
    router_raw: str              # raw router model output
    routing: dict                # parsed + schema-validated routing object
    intent: str                  # data|definition|greeting|meta|unanswerable
    capabilities: list[str]      # subset of {"viz", "email", "template"}
    exec_plan: list[str]         # ordered node names chosen by the orchestrator
    confidence: str              # high|medium|low
    plan_scores: dict            # latent-planner intent/capability scores
    feedback: str                # failure note carried into a re-plan
    replan_count: int            # how many times the graph has re-planned

    # ── resolved entities (deterministic) ────────────────────────────────
    entities: dict               # {wilayas, segment, time_range, recipients}

    # ── sql ──────────────────────────────────────────────────────────────
    sql: str
    sql_attempts: int
    sql_valid: bool
    sql_issues: list[str]
    rows: list[dict]
    columns: list[str]
    exec_ok: bool

    # ── capability artifacts ─────────────────────────────────────────────
    chart_path: str
    email_draft: dict            # {to, subject, body, status: "draft"|"sent"}
    document_path: str

    # ── output ────────────────────────────────────────────────────────────
    answer: str
    errors: list[str]
    trace: list[str]             # human-readable step log for this turn
    timings: dict                # {node_name_ms: float}


def initial_state(query: str, thread_id: str = "default") -> dict:
    """A fresh per-turn state.

    Every per-turn field is reset explicitly. The checkpointer persists state
    across turns on a `thread_id`, so without this reset a stale `chart_path`
    or `email_draft` from the previous turn would leak into this answer. The
    cross-turn memory fields — `turns`, `last_rows`, `last_columns` — are
    deliberately NOT listed here, so they survive.
    """
    return {
        "query": query,
        "thread_id": thread_id,
        # retrieval / planning
        "knowledge": "", "grounding": 0.0,
        "router_raw": "", "routing": {}, "intent": "",
        "capabilities": [], "exec_plan": [], "confidence": "",
        "plan_scores": {}, "feedback": "", "replan_count": 0,
        "entities": {},
        # sql
        "sql": "", "sql_attempts": 0, "sql_valid": False, "sql_issues": [],
        "rows": [], "columns": [], "exec_ok": False,
        # capability artifacts
        "chart_path": "", "document_path": "", "email_draft": None,
        # output
        "answer": "", "errors": [], "trace": [], "timings": {},
    }
