"""v6/state.py — The LangGraph state object.

Every node receives this dict and returns a partial update. Fields are
grouped by pipeline stage. `total=False` means a node only writes the keys
it owns. List fields are never mutated in place — a node returns a fresh
list — so LangGraph's replace-on-update semantics stay predictable.

`turns`, `memory_summary` and `last_rows/last_columns` are the cross-turn
memory: the checkpointer persists them per `thread_id`, so a follow-up like
"and for Oran?" can see what came before.
"""

from __future__ import annotations
from typing import TypedDict


class AgentState(TypedDict, total=False):
    # ── input ────────────────────────────────────────────────────────────
    query: str
    thread_id: str

    # ── conversation memory (persisted by the checkpointer) ──────────────
    turns: list[dict]            # last 2 raw turns (older ones compacted)
    memory_summary: str          # compacted summary of older turns
    last_rows: list[dict]        # rows from the previous data turn
    last_columns: list[str]
    carried_entities: dict       # entities resolved on the previous turn

    # ── the brain (the policy loop) ──────────────────────────────────────
    brain_step: int              # loop iteration counter
    step_log: list[dict]         # one outcome dict per executed action
    intent: str                  # set once, at brain step 0
    next_action: str             # action the brain chose this tick
    continue_score: float        # the seuil signal [0, 1]
    brain_scores: dict           # intent/action/continue scores (debug)

    # ── retrieval ────────────────────────────────────────────────────────
    knowledge: str               # formatted RAG context block
    grounding: float             # top cosine score [0, 1]

    # ── sql (produced by the `sql` action) ───────────────────────────────
    router_raw: str              # raw router model output
    routing: dict                # parsed + schema-validated routing object
    feedback: str                # failure note carried into an SQL retry
    entities: dict               # {wilayas, segment, time_range, recipients}
    sql: str
    sql_valid: bool
    sql_issues: list[str]
    rows: list[dict]
    columns: list[str]
    exec_ok: bool

    # ── capability artifacts ─────────────────────────────────────────────
    chart_path: str
    email_draft: dict            # {to, subject, body, status: "draft"|...}
    document_path: str

    # ── output ───────────────────────────────────────────────────────────
    thoughts: list[dict]         # streamed UI feed: {kind, text}
    final_answer: str
    errors: list[str]
    trace: list[str]             # human-readable step log for this turn
    timings: dict                # {node_name_ms: float}


def initial_state(query: str, thread_id: str = "default") -> dict:
    """A fresh per-turn state.

    Every per-turn field is reset explicitly. The checkpointer persists
    state across turns on a `thread_id`, so without this reset a stale
    `chart_path` or `email_draft` would leak into the next answer. The
    cross-turn memory fields — `turns`, `memory_summary`, `last_rows`,
    `last_columns`, `carried_entities` — are deliberately NOT listed here,
    so they survive.
    """
    return {
        "query": query,
        "thread_id": thread_id,
        # brain loop
        "brain_step": 0, "step_log": [], "intent": "",
        "next_action": "", "continue_score": 0.0, "brain_scores": {},
        # retrieval
        "knowledge": "", "grounding": 0.0,
        # sql
        "router_raw": "", "routing": {}, "feedback": "", "entities": {},
        "sql": "", "sql_valid": False, "sql_issues": [],
        "rows": [], "columns": [], "exec_ok": False,
        # capability artifacts
        "chart_path": "", "document_path": "", "email_draft": None,
        # output
        "thoughts": [], "final_answer": "", "errors": [],
        "trace": [], "timings": {},
    }
