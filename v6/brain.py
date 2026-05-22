"""v6/brain.py — The policy brain.

The brain is the dynamic decision the old one-shot planner never made. It
is called once per loop step. Each call it reads three things:

  - the query,
  - the conversation memory,
  - the outcome of every action executed so far this turn,

and predicts three things through a small trained MLP:

  - intent     — greeting | meta | definition | data | unanswerable
                 (decided at step 0, then reused for the whole turn),
  - action     — the next step: rag | sql | chart | email | template,
  - continue   — a [0, 1] score, the *seuil*: below it the loop ends and
                 the turn routes to the communicator.

Because it re-decides after every step, it reacts to what actually
happened — a failed SQL query, an email with no recipient, a 0-row
result — instead of committing to a fixed plan up front.

It is a *trained* MLP. There is no untrained fallback: build it once with

    python3 -m v6.brain_data    # synthesize agentic traces
    python3 -m v6.train_brain   # train models/brain_head.pt

`encode_outcome` is the single source of truth for the situation vector —
both this module and the trace synthesizer brain_data.py import it, so the
features the head trains on are exactly the features it sees live.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .config import V6Config
from .knowledge import get_encoder

# ── vocabularies ─────────────────────────────────────────────────────────
INTENTS = ["greeting", "meta", "definition", "data", "unanswerable"]
ACTIONS = ["rag", "sql", "chart", "email", "template"]
ERRORS = ["none", "sql_error", "sql_no_rows", "sql_no_query",
          "email_no_recipient", "artifact_failed", "rag_weak"]
ROW_BUCKETS = ["none", "zero", "one", "many"]

# outcome_vec layout (25-d): last-action one-hot over [none]+ACTIONS (6),
# last-ok (1), error one-hot (7), row-bucket one-hot (4), attempt norm (1),
# grounding (1), done-actions multi-hot over ACTIONS (5).
OUTCOME_DIM = (1 + len(ACTIONS)) + 1 + len(ERRORS) + len(ROW_BUCKETS) + 1 + 1 + len(ACTIONS)
SITUATION_DIM = 2 * V6Config.EMBED_DIM + OUTCOME_DIM


def row_bucket(n: int) -> str:
    """Bucket a row count — the brain reasons over scale, not exact counts."""
    if n <= 0:
        return "zero"
    if n == 1:
        return "one"
    return "many"


def encode_outcome(step_log: list[dict], grounding: float = 0.0) -> torch.Tensor:
    """Engineer the 25-d outcome vector from the executed-step history.

    Only the *last* step drives the action/error/row features (that is what
    the brain reacts to); the full history drives the done-actions set. An
    empty `step_log` (loop just started) is the all-"none" situation.
    """
    v = torch.zeros(OUTCOME_DIM)
    last = step_log[-1] if step_log else None
    off = 0

    # last action one-hot — slot 0 is "none" (the loop has not acted yet)
    la = last.get("action") if last else None
    v[off + (0 if la not in ACTIONS else 1 + ACTIONS.index(la))] = 1.0
    off += 1 + len(ACTIONS)

    # last action succeeded
    v[off] = 1.0 if (last and last.get("ok")) else 0.0
    off += 1

    # error-type one-hot
    err = (last or {}).get("error_type", "none")
    v[off + (ERRORS.index(err) if err in ERRORS else ERRORS.index("artifact_failed"))] = 1.0
    off += len(ERRORS)

    # row-count bucket one-hot
    rb = (last or {}).get("row_bucket", "none")
    v[off + (ROW_BUCKETS.index(rb) if rb in ROW_BUCKETS else 0)] = 1.0
    off += len(ROW_BUCKETS)

    # attempt count of the last action, normalized into [0, 1]
    v[off] = min(int((last or {}).get("attempt", 0)), 3) / 3.0
    off += 1

    # RAG grounding score
    v[off] = max(0.0, min(1.0, float(grounding)))
    off += 1

    # done-actions multi-hot
    done = {s.get("action") for s in step_log}
    for i, a in enumerate(ACTIONS):
        if a in done:
            v[off + i] = 1.0

    return v


# ── the network ──────────────────────────────────────────────────────────
class BrainHead(nn.Module):
    """3-head policy MLP: situation vector → (intent, action, continue)."""

    def __init__(self, dim: int = SITUATION_DIM, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.1))
        self.intent = nn.Linear(hidden, len(INTENTS))
        self.action = nn.Linear(hidden, len(ACTIONS))
        self.cont = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.intent(h), self.action(h), self.cont(h).squeeze(-1)


@dataclass
class BrainDecision:
    """One tick of the brain — the next action and whether to keep going."""
    intent: str
    action: str
    action_conf: float
    continue_score: float
    intent_scores: dict = field(default_factory=dict)
    action_scores: dict = field(default_factory=dict)


# ── the brain ────────────────────────────────────────────────────────────
class Brain:
    """The policy loop's decision-maker — a trained 3-head MLP."""

    def __init__(self):
        self.encoder = get_encoder()
        self.head = BrainHead()
        if not os.path.isfile(V6Config.BRAIN_HEAD_PATH):
            raise FileNotFoundError(
                f"brain head not found at {V6Config.BRAIN_HEAD_PATH}\n"
                "The brain is a trained MLP — build it once before use:\n"
                "    python3 -m v6.brain_data    # synthesize agentic traces\n"
                "    python3 -m v6.train_brain   # train the head")
        self.head.load_state_dict(
            torch.load(V6Config.BRAIN_HEAD_PATH, map_location="cpu"))
        self.head.eval()
        # thread_id → (query, query_emb, memory_emb) — query/memory are
        # constant within a turn, so encode them once and reuse every tick.
        self._cache: dict = {}

    def _embed(self, thread_id: str, query: str, memory: str):
        cached = self._cache.get(thread_id)
        if cached and cached[0] == query:
            return cached[1], cached[2]
        qv = self.encoder.encode(query)
        mv = (self.encoder.encode(memory) if memory
              else torch.zeros(V6Config.EMBED_DIM))
        self._cache[thread_id] = (query, qv, mv)
        return qv, mv

    def clear_cache(self, thread_id: str | None = None) -> None:
        if thread_id is None:
            self._cache.clear()
        else:
            self._cache.pop(thread_id, None)

    @torch.no_grad()
    def decide(self, query: str, memory: str, step_log: list[dict],
               grounding: float = 0.0,
               thread_id: str = "default") -> BrainDecision:
        """One brain tick: the situation in, the next decision out."""
        qv, mv = self._embed(thread_id, query, memory)
        ov = encode_outcome(step_log, grounding)
        feat = torch.cat([qv, mv, ov]).unsqueeze(0)

        intent_logits, action_logits, cont_logit = self.head(feat)
        iprob = torch.softmax(intent_logits[0], dim=-1)
        aprob = torch.softmax(action_logits[0], dim=-1)
        cont = float(torch.sigmoid(cont_logit.reshape(-1)[0]))

        ii, ai = int(iprob.argmax()), int(aprob.argmax())
        return BrainDecision(
            intent=INTENTS[ii], action=ACTIONS[ai],
            action_conf=round(float(aprob[ai]), 3),
            continue_score=round(cont, 3),
            intent_scores={INTENTS[i]: round(float(iprob[i]), 3)
                           for i in range(len(INTENTS))},
            action_scores={ACTIONS[i]: round(float(aprob[i]), 3)
                           for i in range(len(ACTIONS))})


_brain: Brain | None = None


def get_brain() -> Brain:
    global _brain
    if _brain is None:
        _brain = Brain()
    return _brain
