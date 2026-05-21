"""v6/graph.py — The LangGraph state machine.

Wires the nodes into a graph and compiles it with a checkpointer, so
conversation state survives across turns. `LatentMindV6.ask()` is the single
public entry point; each conversation is a `thread_id`.

Shape:

    START → plan ─┬─(greeting/meta/def/unanswerable)→ direct_answer ─┐
                  └─(data)→ retrieve → router → orchestrator         │
                                                   │                 │
                                          ┌────────┴────────┐        │
                                       clarify        resolve_entities│
                                          │                 │        │
                                          │      sql_generate ⇄ sql_validate
                                          │                 │        │
                                          │         sql_execute → answer
                                          │                 │        │
                                          │      (fail) replan ──→ router
                                          │                 │        │
                                          │   visualize → template → email
                                          └────────┬────────┴─────────┘
                                               finalize → END
"""

from __future__ import annotations
import time

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from . import nodes
from .state import AgentState, initial_state


def build_graph():
    """Assemble and compile the agent graph with an in-memory checkpointer."""
    g = StateGraph(AgentState)

    g.add_node("plan", nodes.plan_node)
    g.add_node("retrieve", nodes.retrieve_node)
    g.add_node("router", nodes.router_node)
    g.add_node("orchestrator", nodes.orchestrator_node)
    g.add_node("resolve_entities", nodes.resolve_entities_node)
    g.add_node("sql_generate", nodes.sql_generate_node)
    g.add_node("sql_validate", nodes.sql_validate_node)
    g.add_node("sql_execute", nodes.sql_execute_node)
    g.add_node("answer", nodes.answer_node)
    g.add_node("replan", nodes.replan_node)
    g.add_node("visualize", nodes.visualize_node)
    g.add_node("template", nodes.template_node)
    g.add_node("email", nodes.email_node)
    g.add_node("direct_answer", nodes.direct_answer_node)
    g.add_node("clarify", nodes.clarify_node)
    g.add_node("finalize", nodes.finalize_node)

    # the latent planner decides the first branch
    g.add_edge(START, "plan")
    g.add_conditional_edges(
        "plan", nodes.route_after_plan,
        {"direct_answer": "direct_answer", "retrieve": "retrieve"})

    # data path: RAG → schema mapping → deterministic validation
    g.add_edge("retrieve", "router")
    g.add_edge("router", "orchestrator")
    g.add_conditional_edges(
        "orchestrator", nodes.route_after_orchestrator,
        {"clarify": "clarify", "resolve_entities": "resolve_entities"})

    # SQL: generate ⇄ validate (bounded retry) → execute
    g.add_edge("resolve_entities", "sql_generate")
    g.add_edge("sql_generate", "sql_validate")
    g.add_conditional_edges(
        "sql_validate", nodes.route_after_validate,
        {"sql_generate": "sql_generate", "sql_execute": "sql_execute",
         "answer": "answer"})
    g.add_edge("sql_execute", "answer")

    # re-plan loop: a failed execution routes back through the router
    g.add_conditional_edges(
        "answer", nodes.route_after_answer,
        {"replan": "replan", "visualize": "visualize"})
    g.add_edge("replan", "router")

    # capability chain — each node self-skips unless its tag is in `plan`
    g.add_edge("visualize", "template")
    g.add_edge("template", "email")
    g.add_edge("email", "finalize")

    g.add_edge("direct_answer", "finalize")
    g.add_edge("clarify", "finalize")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=MemorySaver())


class LatentMindV6:
    """The agent: one compiled graph, conversation memory per `thread_id`."""

    def __init__(self):
        self.graph = build_graph()

    def ask(self, query: str, thread_id: str = "default",
            verbose: bool = False) -> dict:
        """Run one turn. State persists under `thread_id` for follow-ups."""
        t0 = time.time()
        config = {"configurable": {"thread_id": thread_id},
                  "recursion_limit": 60}
        result = self.graph.invoke(initial_state(query, thread_id), config)
        timings = dict(result.get("timings", {}))
        timings["total_ms"] = round((time.time() - t0) * 1000, 1)
        result["timings"] = timings
        if verbose:
            for line in result.get("trace", []):
                print("   ·", line)
        return result

    def reset(self, thread_id: str | None = None) -> None:
        """Forget conversation memory (and any held KV caches)."""
        from . import slm as _slm
        if _slm._slm is not None:
            if thread_id:
                _slm._slm.clear_thread(thread_id)
            else:
                _slm._slm._store.clear()
        self.graph = build_graph()          # fresh checkpointer = blank memory


_agent: LatentMindV6 | None = None


def get_agent() -> LatentMindV6:
    global _agent
    if _agent is None:
        _agent = LatentMindV6()
    return _agent
