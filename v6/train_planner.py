"""v6/train_planner.py — Train the MLP planner head.

Pipeline:
    python3 -m v6.planner_data         # 1. generate data/planner_train.jsonl
    python3 -m v6.train_planner        # 2. train → models/planner_head.pt
    V6_PLANNER=mlp python3 -m v6.cli   # 3. the agent uses the trained head

The head maps a 2048-d feature — BGE-M3 embedding of the query concatenated
with the embedding of (history + query) — to an intent class and three
capability multi-labels. It is tiny (~0.6M params) and trains in seconds on
CPU once the embeddings are computed.
"""

from __future__ import annotations
import json
import os
import random

import torch
import torch.nn as nn

from .config import V6Config
from .knowledge import get_encoder
from .planner import LatentPlanner, PlannerHead

INTENTS = LatentPlanner.INTENTS
CAPS = LatentPlanner.CAPS
DATA_PATH = os.path.join(V6Config.DATA_DIR, "planner_train.jsonl")


def _featurize(rows: list[dict], encoder):
    """Embed each example into the 2048-d query⊕(history+query) feature."""
    queries = [r["query"] for r in rows]
    pairs = [((r.get("history", "") + "\n" + r["query"]).strip() or r["query"])
             for r in rows]
    qv = encoder.encode_batch(queries)
    pv = encoder.encode_batch(pairs)
    feats = torch.cat([qv, pv], dim=1)

    y_intent = torch.tensor([INTENTS.index(r["intent"]) for r in rows])
    y_caps = torch.zeros(len(rows), len(CAPS))
    for i, r in enumerate(rows):
        for c in r.get("caps", []):
            if c in CAPS:
                y_caps[i, CAPS.index(c)] = 1.0
    return feats, y_intent, y_caps


def main(epochs: int = 80, lr: float = 1e-3, val_frac: float = 0.15) -> None:
    if not os.path.isfile(DATA_PATH):
        raise SystemExit(
            f"missing {DATA_PATH}\nrun first:  python3 -m v6.planner_data")

    rows = [json.loads(line) for line in open(DATA_PATH, encoding="utf-8")
            if line.strip()]
    random.Random(0).shuffle(rows)
    print(f"loaded {len(rows)} examples; encoding with BGE-M3 "
          f"(one-time, ~{len(rows) // 100}s)...")

    encoder = get_encoder()
    feats, y_intent, y_caps = _featurize(rows, encoder)

    n_val = max(1, int(len(rows) * val_frac))
    x_tr, x_va = feats[n_val:], feats[:n_val]
    yi_tr, yi_va = y_intent[n_val:], y_intent[:n_val]
    yc_tr, yc_va = y_caps[n_val:], y_caps[:n_val]

    head = PlannerHead()
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    ce, bce = nn.CrossEntropyLoss(), nn.BCEWithLogitsLoss()

    for ep in range(1, epochs + 1):
        head.train()
        opt.zero_grad()
        intent_logits, cap_logits = head(x_tr)
        loss = ce(intent_logits, yi_tr) + bce(cap_logits, yc_tr)
        loss.backward()
        opt.step()

        if ep % 20 == 0 or ep == epochs:
            head.eval()
            with torch.no_grad():
                il, cl = head(x_va)
                intent_acc = (il.argmax(1) == yi_va).float().mean().item()
                cap_acc = ((cl > 0).float() == yc_va).float().mean().item()
            print(f"  epoch {ep:3d} | loss {loss.item():.4f} "
                  f"| val intent acc {intent_acc:.3f} "
                  f"| val cap acc {cap_acc:.3f}")

    os.makedirs(V6Config.MODELS_DIR, exist_ok=True)
    torch.save(head.state_dict(), V6Config.PLANNER_HEAD_PATH)
    print(f"\nsaved → {V6Config.PLANNER_HEAD_PATH}")
    print("use it:  V6_PLANNER=mlp python3 -m v6.cli")


if __name__ == "__main__":
    main()
