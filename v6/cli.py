"""v6/cli.py — Interactive REPL and demo runner for the V6 agent.

    python3 -m v6.cli                 # run the demo queries
    python3 -m v6.cli -i              # interactive REPL
    python3 -m v6.cli -t "question"   # one question, with the node trace

REPL commands:  :trace  :reset  :thread <id>  :send  :quit
"""

from __future__ import annotations
import os
import sys

# ── resolve the SQLite backend before any v6 module reads the env ────────
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.environ.get("V6_USE_SQLITE") and not os.environ.get("V5_USE_SQLITE"):
    for _p in ("/content/interndb.sqlite", os.path.join(_REPO, "interndb.sqlite")):
        if os.path.isfile(_p):
            os.environ["V6_USE_SQLITE"] = "1"
            os.environ["V6_SQLITE_PATH"] = _p
            break

import argparse  # noqa: E402

from .capabilities import send_email  # noqa: E402
from .config import V6Config  # noqa: E402
from .graph import LatentMindV6  # noqa: E402

DIVIDER = "─" * 68


def _fmt_timings(t: dict) -> str:
    parts = []
    for key, label in (("retrieve_ms", "rag"), ("router_ms", "router"),
                        ("sqlgen_ms", "sqlgen"), ("exec_ms", "exec")):
        if t.get(key):
            parts.append(f"{label} {t[key] / 1000:.2f}s")
    total = t.get("total_ms", 0) / 1000
    return " + ".join(parts) + f" = {total:.2f}s total"


def print_result(res: dict, show_trace: bool = False) -> None:
    intent = res.get("intent", "?")
    plan = "→".join(res.get("plan", []))
    conf = res.get("confidence", "?")
    grounding = res.get("grounding", 0.0)
    print(f"\n[{intent} | plan: {plan} | confidence: {conf} | "
          f"grounding: {grounding:.3f}]")

    if res.get("sql"):
        print(f"  SQL: {res['sql']}")

    print()
    print(res.get("answer", "(no answer)"))

    if show_trace:
        print("\n  trace:")
        for line in res.get("trace", []):
            print(f"    · {line}")

    print(f"\n⏱  {_fmt_timings(res.get('timings', {}))}")


def banner() -> None:
    backend = (f"SQLite ({V6Config.sqlite_path()})" if V6Config.USE_SQLITE
               else f"MySQL ({V6Config.MYSQL_DB})")
    print(DIVIDER)
    print("  LatentMind V6 — LangGraph agentic analytics")
    print(f"  model:   {V6Config.slm_id()}")
    print(f"  backend: {backend}")
    print(f"  planner: {V6Config.PLANNER_MODE}")
    print(f"  4-bit:   {V6Config.USE_4BIT}   flash-attn: {V6Config.USE_FLASH_ATTN}")
    print(DIVIDER)


DEMO_QUERIES = [
    "hello",
    "what does ARPU mean",
    "what was the total revenue in Oran",
    "compare churn rate between Algiers and Constantine",
    "show the weekly arpu trend for prepaid",
    "chart the average revenue by wilaya",
    "email the average prepaid churn by wilaya to the finance director",
    "and what about Oran?",
]


def run_demo(agent: LatentMindV6, show_trace: bool) -> None:
    for q in DEMO_QUERIES:
        print(f"\n{DIVIDER}\nQ: {q}\n{DIVIDER}")
        res = agent.ask(q, thread_id="demo", verbose=False)
        print_result(res, show_trace)


def run_repl(agent: LatentMindV6, show_trace: bool) -> None:
    print("\nInteractive mode. Type a question, or :quit to exit.\n")
    thread = "session"
    last_draft = None
    while True:
        try:
            line = input("you ▸ ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in (":quit", ":q", ":exit"):
            break
        if line == ":trace":
            show_trace = not show_trace
            print(f"  trace display {'on' if show_trace else 'off'}")
            continue
        if line == ":reset":
            agent.reset()
            last_draft = None
            print("  conversation memory cleared")
            continue
        if line.startswith(":thread"):
            parts = line.split(maxsplit=1)
            thread = parts[1] if len(parts) > 1 else thread
            print(f"  thread = {thread}")
            continue
        if line == ":send":
            if not last_draft or not last_draft.get("to"):
                print("  no email draft with a recipient to send")
                continue
            print(f"  sending to {last_draft['to']} ...")
            r = send_email(last_draft)
            print("  sent ✓" if r["ok"] else f"  failed: {r['error']}")
            continue

        res = agent.ask(line, thread_id=thread)
        print_result(res, show_trace)
        last_draft = res.get("email_draft")
        if last_draft and last_draft.get("to"):
            print("  (type :send to send this drafted email)")


def main() -> None:
    ap = argparse.ArgumentParser(description="LatentMind V6 agent")
    ap.add_argument("query", nargs="?", help="a single question to run")
    ap.add_argument("-i", "--interactive", action="store_true",
                    help="interactive REPL")
    ap.add_argument("-t", "--trace", action="store_true",
                    help="show the node trace")
    args = ap.parse_args()

    banner()
    print("\nloading pipeline (BGE-M3 + dual-role SLM)...")
    agent = LatentMindV6()
    print("ready.\n")

    if args.query:
        res = agent.ask(args.query, thread_id="cli")
        print_result(res, args.trace)
    elif args.interactive:
        run_repl(agent, args.trace)
    else:
        run_demo(agent, args.trace)


if __name__ == "__main__":
    sys.exit(main())
