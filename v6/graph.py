"""v6/graph.py — The LangGraph policy loop.

The graph is a star. The brain is the hub: it decides one action, that
action runs and returns to the brain, which re-decides with the new
outcome in hand. The loop ends when the brain's continue score drops below
the seuil (or it is confused, or BRAIN_MAX_STEPS is reached) — then the
turn routes once to the communicator and ends.

    START → brain ─┬─ rag      → brain
                   ├─ sql      → brain
                   ├─ chart    → brain
                   ├─ email    → brain
                   ├─ template → brain
                   └─ communicator → END

The compiled graph carries a MemorySaver checkpointer, so conversation
memory (turns, memory_summary) survives across turns on a `thread_id`.
`LatentMindV6.ask()` is the single public entry point.
"""

from __future__ import annotations
import time

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from . import nodes
from .state import AgentState, initial_state

_ACTIONS = ("rag", "sql", "chart", "email", "template")


def build_graph():
    """Assemble and compile the policy-loop graph."""
    g = StateGraph(AgentState)

    g.add_node("brain", nodes.brain_node)
    g.add_node("rag", nodes.rag_node)
    g.add_node("sql", nodes.sql_node)
    g.add_node("chart", nodes.chart_node)
    g.add_node("email", nodes.email_node)
    g.add_node("template", nodes.template_node)
    g.add_node("communicator", nodes.communicator_node)

    # the brain is the hub — its decision routes to one action or the end
    g.add_edge(START, "brain")
    g.add_conditional_edges("brain", nodes.route_after_brain, {
        "rag": "rag", "sql": "sql", "chart": "chart", "email": "email",
        "template": "template", "communicator": "communicator"})
    for action in _ACTIONS:
        g.add_edge(action, "brain")          # every action returns to the brain
    g.add_edge("communicator", END)

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
        """Forget memory — KV caches, the brain's embed cache, checkpoints."""
        from . import slm as _slm
        from . import brain as _brain
        if _slm._slm is not None:
            if thread_id:
                _slm._slm.clear_thread(thread_id)
            else:
                _slm._slm._store.clear()
        if _brain._brain is not None:
            _brain._brain.clear_cache(thread_id)
        self.graph = build_graph()          # fresh checkpointer = blank memory


_agent: LatentMindV6 | None = None


def get_agent() -> LatentMindV6:
    global _agent
    if _agent is None:
        _agent = LatentMindV6()
    return _agent
