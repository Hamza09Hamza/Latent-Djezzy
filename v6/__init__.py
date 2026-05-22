"""v6 — LatentMind agentic analytics pipeline.

A LangGraph policy loop over an Algerian-telecom database. A trained MLP —
the brain — picks one action at a time (rag · sql · chart · email ·
template), watches it finish, and re-decides with the outcome in hand,
until its continue score drops below the seuil. One small language model
plays two roles inside the `sql` action (router + SQL generator) and hands
off its KV cache between them — the "latent communication" of LatentMAS.
The graph answers data questions, explains KPIs, draws charts, drafts
emails and fills report templates, and remembers the conversation across
turns via a LangGraph checkpointer.

Entry point:

    from v6.graph import LatentMindV6
    agent = LatentMindV6()
    print(agent.ask("total revenue in Oran", thread_id="sess1")["final_answer"])
"""

__version__ = "6.0.0"
