"""v6/planner.py — The latent planner.

This is the dynamic decision the regex orchestrator was missing. Instead of
hand-written patterns, the query (and recent history) is classified in
BGE-M3 embedding space — the same encoder already loaded for RAG, so the
cost is one extra matmul. Two modes:

  "prototype" (default) — nearest-prototype: cosine of the query against a
                          few example phrases per class. No training; add a
                          phrasing to data/planner_prototypes.json and the
                          planner learns it.
  "mlp"                 — a trained head (query⊕history embedding → 256 →
                          intent + capabilities). See planner_data.py and
                          train_planner.py; set V6_PLANNER=mlp to use it.

Both return a PlanDecision — a discrete, inspectable plan, decoded once.
Dynamic decision, predictable execution.
"""

from __future__ import annotations
import json
import os
import re
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .config import V6Config
from .knowledge import get_encoder

_FOLLOWUP_RE = re.compile(
    r"^\s*(and|what about|how about|now|also|then|same for|ok now|what of)\b",
    re.I)


def is_followup(query: str) -> bool:
    """A terse, context-dependent query such as 'and for Oran?'."""
    q = (query or "").strip()
    if _FOLLOWUP_RE.search(q):
        return True
    return len(q.split()) <= 2


@dataclass
class PlanDecision:
    """The planner's output — intent, capabilities, and the scores behind them."""
    intent: str
    intent_score: float
    intent_margin: float
    capabilities: list[str]
    cap_scores: dict
    followup: bool
    mode: str
    intent_scores: dict = field(default_factory=dict)

    @property
    def low_confidence(self) -> bool:
        return self.intent_score < V6Config.PLANNER_LOW_CONF


class PlannerHead(nn.Module):
    """Trained head: query⊕history embedding → intent logits + capability logits."""

    def __init__(self, dim: int = 2 * V6Config.EMBED_DIM, hidden: int = 256,
                 n_intents: int = 5, n_caps: int = 3):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(0.1))
        self.intent = nn.Linear(hidden, n_intents)
        self.caps = nn.Linear(hidden, n_caps)

    def forward(self, x):
        h = self.trunk(x)
        return self.intent(h), self.caps(h)


class LatentPlanner:
    """Classifies a query into an intent + capability set in embedding space."""

    INTENTS = ["greeting", "data", "definition", "meta", "unanswerable"]
    CAPS = ["viz", "email", "template"]

    def __init__(self):
        self.encoder = get_encoder()
        with open(V6Config.PLANNER_PROTOTYPES, encoding="utf-8") as f:
            protos = json.load(f)
        self.intent_protos = {
            it: self._encode(protos["intents"].get(it, []))
            for it in self.INTENTS}
        self.cap_protos = {
            cp: self._encode(protos["capabilities"].get(cp, []))
            for cp in self.CAPS}
        self.mode = "prototype"
        self.head: PlannerHead | None = None
        if V6Config.PLANNER_MODE == "mlp":
            self._load_head()

    def _encode(self, phrases: list[str]):
        if not phrases:
            return torch.empty(0, V6Config.EMBED_DIM)
        return self.encoder.encode_batch(phrases)

    def _load_head(self) -> None:
        if not os.path.isfile(V6Config.PLANNER_HEAD_PATH):
            print(f"[planner] V6_PLANNER=mlp but no head at "
                  f"{V6Config.PLANNER_HEAD_PATH} — staying in prototype mode")
            return
        self.head = PlannerHead()
        self.head.load_state_dict(
            torch.load(V6Config.PLANNER_HEAD_PATH, map_location="cpu"))
        self.head.eval()
        self.mode = "mlp"

    # ── public API ───────────────────────────────────────────────────────
    @torch.no_grad()
    def plan(self, query: str, history: str = "") -> PlanDecision:
        qv = self.encoder.encode(query)               # [1024], L2-normalized
        if self.mode == "mlp" and self.head is not None:
            return self._plan_mlp(query, history, qv)
        return self._plan_prototype(query, qv)

    # ── prototype mode ───────────────────────────────────────────────────
    @staticmethod
    def _score(qv, protos) -> float:
        """Mean of the top-2 cosine similarities to a class's prototypes."""
        if protos is None or len(protos) == 0:
            return 0.0
        sims = protos @ qv
        return float(torch.topk(sims, min(2, len(sims))).values.mean())

    def _plan_prototype(self, query: str, qv) -> PlanDecision:
        intent_scores = {it: self._score(qv, p)
                         for it, p in self.intent_protos.items()}
        ranked = sorted(intent_scores.items(), key=lambda x: -x[1])
        intent, score = ranked[0]
        margin = score - (ranked[1][1] if len(ranked) > 1 else 0.0)

        caps, cap_scores = [], {}
        for cp, protos in self.cap_protos.items():
            s = float((protos @ qv).max()) if len(protos) else 0.0
            cap_scores[cp] = round(s, 3)
            if s >= V6Config.CAP_THRESHOLD:
                caps.append(cp)

        return PlanDecision(
            intent=intent, intent_score=round(score, 3),
            intent_margin=round(margin, 3), capabilities=caps,
            cap_scores=cap_scores, followup=is_followup(query),
            mode="prototype",
            intent_scores={k: round(v, 3) for k, v in intent_scores.items()})

    # ── mlp mode ─────────────────────────────────────────────────────────
    def _plan_mlp(self, query: str, history: str, qv) -> PlanDecision:
        pv = self.encoder.encode(
            (history + "\n" + query).strip() if history else query)
        feat = torch.cat([qv, pv]).unsqueeze(0)
        intent_logits, cap_logits = self.head(feat)
        iprob = torch.softmax(intent_logits[0], dim=-1)
        cprob = torch.sigmoid(cap_logits[0])

        idx = int(iprob.argmax())
        srt = torch.sort(iprob, descending=True).values
        margin = float(srt[0] - srt[1]) if len(srt) > 1 else float(srt[0])

        caps, cap_scores = [], {}
        for j, cp in enumerate(self.CAPS):
            s = float(cprob[j])
            cap_scores[cp] = round(s, 3)
            if s >= 0.5:
                caps.append(cp)

        return PlanDecision(
            intent=self.INTENTS[idx], intent_score=round(float(iprob[idx]), 3),
            intent_margin=round(margin, 3), capabilities=caps,
            cap_scores=cap_scores, followup=is_followup(query), mode="mlp",
            intent_scores={self.INTENTS[i]: round(float(iprob[i]), 3)
                           for i in range(len(self.INTENTS))})


_planner: LatentPlanner | None = None


def get_planner() -> LatentPlanner:
    global _planner
    if _planner is None:
        _planner = LatentPlanner()
    return _planner
