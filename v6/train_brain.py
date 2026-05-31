"""v6/train_brain.py — Train the policy brain's MLP head.

Pipeline:
    python3 -m v6.brain_data     # 1. synthesize data/brain_train.jsonl
    python3 -m v6.train_brain    # 2. train → models/brain_head.pt
    python3 -m v6.cli            # 3. the agent loads the trained brain

The head maps the 2073-d situation vector — BGE-M3(query) ⊕ BGE-M3(memory)
⊕ the 25-d outcome vector — to three predictions: intent, next action, and
the continue score (the seuil). Three losses:

  - intent   — cross-entropy on every row (intent is outcome-invariant),
  - action   — cross-entropy on rows where the loop continues (the last
               tick of a trace has no action label),
  - continue — binary cross-entropy on every row.

Intent and action are class-weighted and continue is pos-weighted, so the
rare classes (chart/email/template, greeting/meta) are not drowned out.
Trains in seconds on CPU once the embeddings are cached.
"""

from __future__ import annotations
import json
import os
import random

import torch
import torch.nn as nn

from .brain import ACTIONS, INTENTS, BrainHead, encode_outcome
from .config import V6Config
from .knowledge import get_encoder

DATA_PATH = V6Config.BRAIN_TRAIN_PATH


def _featurize(rows: list[dict], encoder):
    """Build the situation matrix + the three label tensors.

    Queries and memory strings repeat across the ticks of a trace, so each
    unique string is embedded once and reused.
    """
    queries = [r["query"] for r in rows]
    memories = [r.get("history", "") for r in rows]

    uniq_q = sorted(set(queries))
    uniq_m = sorted({m for m in memories if m})
    q_emb = dict(zip(uniq_q, encoder.encode_batch(uniq_q)))
    m_emb = dict(zip(uniq_m, encoder.encode_batch(uniq_m))) if uniq_m else {}
    zero = torch.zeros(V6Config.EMBED_DIM)

    feats = []
    for r, q, m in zip(rows, queries, memories):
        ov = encode_outcome(r.get("step_log", []), r.get("grounding", 0.0))
        feats.append(torch.cat([q_emb[q], m_emb.get(m, zero), ov]))
    X = torch.stack(feats)

    y_intent = torch.tensor([INTENTS.index(r["intent"]) for r in rows])
    y_action = torch.tensor([
        ACTIONS.index(r["label_action"]) if r["label_action"] else -1
        for r in rows])
    y_cont = torch.tensor([float(r["label_continue"]) for r in rows])
    # Intent is decided at step 0 (empty outcome) and reused for the rest of
    # the turn, so it must be TRAINED only on step-0 rows — the ticks whose
    # situation matches inference. Training it on later ticks both teaches a
    # signal that's never read AND inflates multi-tick classes (data), which
    # the inverse-freq weighting then over-corrects (the "count → unanswerable"
    # basin). A step-0 row is one with no executed steps yet.
    is_step0 = torch.tensor([len(r.get("step_log", [])) == 0 for r in rows])
    return X, y_intent, y_action, y_cont, is_step0


def _class_weights(y: torch.Tensor, n: int) -> torch.Tensor:
    """Inverse-frequency weights over the valid (non -1) labels in `y`."""
    counts = torch.bincount(y[y >= 0], minlength=n).float().clamp(min=1.0)
    return counts.sum() / (n * counts)


def main(epochs: int = 160, lr: float = 1e-3, val_frac: float = 0.15) -> None:
    if not os.path.isfile(DATA_PATH):
        raise SystemExit(
            f"missing {DATA_PATH}\nrun first:  python3 -m v6.brain_data")

    with open(DATA_PATH, encoding="utf-8") as f:
        rows = [json.loads(ln) for ln in f if ln.strip()]
    random.Random(0).shuffle(rows)
    print(f"loaded {len(rows)} rows; encoding with BGE-M3 (one-time)...")

    encoder = get_encoder()
    X, y_intent, y_action, y_cont, is_step0 = _featurize(rows, encoder)

    n_val = max(1, int(len(rows) * val_frac))
    xtr, xva = X[n_val:], X[:n_val]
    itr, iva = y_intent[n_val:], y_intent[:n_val]
    atr, ava = y_action[n_val:], y_action[:n_val]
    ctr, cva = y_cont[n_val:], y_cont[:n_val]
    s0tr, s0va = is_step0[n_val:], is_step0[:n_val]

    head = BrainHead()
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    # Intent weights + loss use ONLY step-0 rows (see _featurize).
    ce_i = nn.CrossEntropyLoss(weight=_class_weights(itr[s0tr], len(INTENTS)))
    ce_a = nn.CrossEntropyLoss(weight=_class_weights(atr, len(ACTIONS)))
    pos = ((ctr == 0).sum().clamp(min=1).float()
           / (ctr == 1).sum().clamp(min=1).float())
    bce = nn.BCEWithLogitsLoss(pos_weight=pos)

    for ep in range(1, epochs + 1):
        head.train()
        opt.zero_grad()
        il, al, cl = head(xtr)
        amask = atr >= 0
        loss = bce(cl, ctr)
        if s0tr.any():
            loss = loss + ce_i(il[s0tr], itr[s0tr])
        if amask.any():
            loss = loss + ce_a(al[amask], atr[amask])
        loss.backward()
        opt.step()

        if ep % 40 == 0 or ep == epochs:
            head.eval()
            with torch.no_grad():
                il, al, cl = head(xva)
                im = s0va if s0va.any() else torch.ones_like(iva, dtype=torch.bool)
                iacc = (il[im].argmax(1) == iva[im]).float().mean().item()
                vm = ava >= 0
                aacc = ((al[vm].argmax(1) == ava[vm]).float().mean().item()
                        if vm.any() else 0.0)
                cacc = (((torch.sigmoid(cl) >= 0.5).float() == cva)
                        .float().mean().item())
            print(f"  epoch {ep:3d} | loss {loss.item():.4f} "
                  f"| val intent {iacc:.3f} | action {aacc:.3f} "
                  f"| continue {cacc:.3f}")

    os.makedirs(V6Config.MODELS_DIR, exist_ok=True)
    torch.save(head.state_dict(), V6Config.BRAIN_HEAD_PATH)
    print(f"\nsaved → {V6Config.BRAIN_HEAD_PATH}")

    # Held-out eval — the number to ACTUALLY trust. The val accuracy printed
    # above is an IID split of the same synthetic templates, so it runs hot
    # (95 %+) and says little about generalization. brain_eval.jsonl is a
    # hand-written, never-templated set; this is the honest score.
    try:
        from .eval_brain import EVAL_PATH, evaluate
        if os.path.isfile(EVAL_PATH):
            import v6.brain as _b
            _b._brain = None                      # reload the just-saved head
            res = evaluate()
            ck = res["checks"]

            def _p(k: str) -> str:
                a, n = ck[k]
                return f"{100 * a / n:.1f}% ({a}/{n})" if n else "n/a"

            print("\nheld-out eval (TRUST THIS, not the val above):")
            print(f"  intent {_p('intent')} | action {_p('action')} "
                  f"| continue {_p('continue')}")
            print("  full per-tag report + misses:  python3 -m v6.eval_brain")
    except Exception as e:  # noqa: BLE001 — eval is advisory, never fatal
        print(f"(held-out eval skipped: {e})")

    print("\nuse it:  python3 -m v6.cli")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=160)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    main(epochs=args.epochs, lr=args.lr)
