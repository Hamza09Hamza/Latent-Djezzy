"""v6 — LatentMind agentic analytics pipeline.

A LangGraph state machine over an Algerian-telecom database. One small
language model plays two roles (router + SQL generator) and hands off its
KV cache between them — the "latent communication" of LatentMAS. Around
that model sits a *deterministic* orchestrator: it, not the model, decides
the route. The graph can answer data questions, explain KPIs, draw charts,
draft emails, and fill report templates, and it remembers the conversation
across turns via a LangGraph checkpointer.

Entry point:

    from v6.graph import LatentMindV6
    agent = LatentMindV6()
    print(agent.ask("total revenue in Oran", thread_id="sess1")["answer"])
"""

__version__ = "6.0.0"
