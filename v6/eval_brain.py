"""v6/eval_brain.py — Trustworthy, held-out evaluation of the policy brain.

WHY THIS EXISTS
---------------
train_brain.py reports a validation accuracy, but its val split is drawn from
the SAME synthetic templates as training (an IID split of brain_train.jsonl).
A model only has to recognize the template distribution it memorized to score
95 %+ there — that number says almost nothing about how the brain handles a
real, naturally-phrased, never-seen query.

This module evaluates against data/brain_eval.jsonl: a small, HAND-WRITTEN,
held-out set whose phrasings deliberately do NOT come from the trace generator.
It is the honest measure of generalization, and it is interpretable: every miss
is printed with the human note explaining what the case is testing, and results
break down per tag (cross-turn vs fresh-chart vs definition vs unanswerable …)
so you see WHERE the brain is weak, not just an aggregate.

Each eval case (one JSON object per line) may assert any of three things:

    {
      "id":   "fresh-chart-prefix-1",
      "query": "now show me the churn rate of Alger vs Oran for 2025",
      "memory": "Q: revenue trend in Bejaia\nA: line chart rendered",
      "step_log": [ {action, ok, error_type, row_bucket, attempt}, ... ],
      "grounding": 0.5,
      "expect_intent":   "data",     # checked against decision.intent
      "expect_action":   "rag",      # checked against decision.action
      "expect_continue": true,       # checked against (continue >= BRAIN_SEUIL)
      "tags": ["route", "fresh-chart"],
      "note": "starts like a follow-up but names a new metric -> fresh query"
    }

Run:  python3 -m v6.eval_brain          # human report
      python3 -m v6.eval_brain --json   # machine-readable summary
"""

from __future__ import annotations
import json
import os
from collections import defaultdict

from .config import V6Config

EVAL_PATH = os.path.join(V6Config.DATA_DIR, "brain_eval.jsonl")


def _load_cases(path: str) -> list[dict]:
    if not os.path.isfile(path):
        raise SystemExit(f"missing eval set: {path}")
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()
                and not ln.lstrip().startswith("//")]


def evaluate(path: str = EVAL_PATH) -> dict:
    """Run the brain on every eval case and return a structured result dict."""
    from .brain import get_brain
    brain = get_brain()
    seuil = V6Config.BRAIN_SEUIL
    cases = _load_cases(path)

    # per-check tallies + per-tag tallies (correct, total)
    checks = {"intent": [0, 0], "action": [0, 0], "continue": [0, 0]}
    by_tag: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    confusion_intent: dict[tuple[str, str], int] = defaultdict(int)
    confusion_action: dict[tuple[str, str], int] = defaultdict(int)
    misses: list[dict] = []

    for c in cases:
        d = brain.decide(
            c["query"], c.get("memory", ""), c.get("step_log", []),
            grounding=float(c.get("grounding", 0.5)),
            thread_id=f"eval-{c.get('id', '')}")
        tags = c.get("tags", []) or ["untagged"]
        case_ok = True
        problems: list[str] = []

        if "expect_intent" in c:
            ok = d.intent == c["expect_intent"]
            checks["intent"][1] += 1
            checks["intent"][0] += ok
            confusion_intent[(c["expect_intent"], d.intent)] += 1
            if not ok:
                case_ok = False
                problems.append(f"intent: want {c['expect_intent']}, got {d.intent}")

        if "expect_action" in c:
            ok = d.action == c["expect_action"]
            checks["action"][1] += 1
            checks["action"][0] += ok
            confusion_action[(c["expect_action"], d.action)] += 1
            if not ok:
                case_ok = False
                problems.append(
                    f"action: want {c['expect_action']}, got {d.action} "
                    f"({d.action_conf:.2f})")

        if "expect_continue" in c:
            went = d.continue_score >= seuil
            ok = went == bool(c["expect_continue"])
            checks["continue"][1] += 1
            checks["continue"][0] += ok
            if not ok:
                case_ok = False
                problems.append(
                    f"continue: want {'go' if c['expect_continue'] else 'stop'}, "
                    f"got {'go' if went else 'stop'} (score {d.continue_score:.2f})")

        for t in tags:
            by_tag[t][1] += 1
            by_tag[t][0] += case_ok
        if not case_ok:
            misses.append({"id": c.get("id", "?"), "query": c["query"],
                           "problems": problems, "note": c.get("note", "")})

    return {
        "n_cases": len(cases),
        "checks": checks,
        "by_tag": dict(by_tag),
        "confusion_intent": confusion_intent,
        "confusion_action": confusion_action,
        "misses": misses,
    }


def _pct(c: int, n: int) -> str:
    return f"{(100.0 * c / n):5.1f}%  ({c}/{n})" if n else "   n/a"


def print_report(res: dict) -> None:
    print(f"\n=== Brain held-out eval — {res['n_cases']} hand-written cases ===\n")
    print("Per-check accuracy (this is the number to trust):")
    for name in ("intent", "action", "continue"):
        c, n = res["checks"][name]
        print(f"  {name:9} {_pct(c, n)}")

    overall_c = sum(v[0] for v in res["checks"].values())
    overall_n = sum(v[1] for v in res["checks"].values())
    print(f"  {'TOTAL':9} {_pct(overall_c, overall_n)}")

    print("\nPer-tag accuracy (case fully correct):")
    for tag in sorted(res["by_tag"], key=lambda t: res["by_tag"][t][0] / max(1, res["by_tag"][t][1])):
        c, n = res["by_tag"][tag]
        print(f"  {tag:20} {_pct(c, n)}")

    if res["misses"]:
        print(f"\nMisses ({len(res['misses'])}):")
        for m in res["misses"]:
            print(f"  [{m['id']}] {m['query']!r}")
            for p in m["problems"]:
                print(f"       ✗ {p}")
            if m["note"]:
                print(f"       · {m['note']}")
    else:
        print("\nNo misses — every assertion passed.")
    print()


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--path", default=EVAL_PATH)
    args = ap.parse_args()

    res = evaluate(args.path)
    if args.json:
        # confusion dicts have tuple keys — stringify for JSON
        out = {**res,
               "confusion_intent": {f"{k[0]}->{k[1]}": v
                                    for k, v in res["confusion_intent"].items()},
               "confusion_action": {f"{k[0]}->{k[1]}": v
                                    for k, v in res["confusion_action"].items()}}
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print_report(res)


if __name__ == "__main__":
    main()
