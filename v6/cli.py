"""v6/cli.py — Interactive REPL for the V6 agent.

    python3 -m v6.cli                  # demo queries
    python3 -m v6.cli -i               # interactive REPL
    python3 -m v6.cli "your question"  # one-shot question
    python3 -m v6.cli -t "question"    # one-shot with node trace

REPL commands:  :trace  :reset  :thread <id>  :send  :quit
"""

from __future__ import annotations
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.environ.get("V6_USE_SQLITE") and not os.environ.get("V5_USE_SQLITE"):
    for _p in ("/content/interndb.sqlite", os.path.join(_REPO, "interndb.sqlite")):
        if os.path.isfile(_p):
            os.environ["V6_USE_SQLITE"] = "1"
            os.environ["V6_SQLITE_PATH"] = _p
            break

import argparse

from .capabilities import send_email
from .config import V6Config
from .graph import LatentMindV6
from .slm import get_slm

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
    ps = res.get("plan_scores") or {}
    intent = res.get("intent", "?")
    caps   = res.get("capabilities", [])
    score  = ps.get("score", 0.0)
    print(f"\n[intent={intent}  conf={score:.2f}  caps={caps}]")

    sql = res.get("sql", "")
    if sql:
        short = sql.replace("\n", " ")[:100]
        print(f"  SQL: {short}{'…' if len(sql) > 100 else ''}")

    rows = res.get("rows", [])
    if rows:
        print(f"  rows: {len(rows)}")

    print()
    print(res.get("final_answer", "(no answer)"))

    chart = res.get("chart_path", "")
    if chart:
        print(f"\n  Chart saved → {chart}")

    draft = res.get("email_draft")
    if draft:
        to = draft.get("to_name", "?")
        subj = draft.get("subject", "")
        print(f"\n  Email draft → {to} | {subj}")
        print(f"  (type :send to send)")

    doc = res.get("document_path", "")
    if doc:
        print(f"\n  Report saved → {doc}")

    if show_trace:
        print("\n  trace:")
        for line in res.get("trace", []):
            print(f"    · {line}")

    print(f"\n  {_fmt_timings(res.get('timings', {}))}")


def _clear_cache(thread: str) -> None:
    try:
        get_slm().clear_thread(thread)
    except Exception:
        pass


def banner() -> None:
    backend = (f"SQLite ({V6Config.sqlite_path()})" if V6Config.USE_SQLITE
               else f"MySQL ({V6Config.MYSQL_DB})")
    print(DIVIDER)
    print("  LatentMind V6 — LangGraph agentic analytics")
    print(f"  model:   {V6Config.slm_id()}")
    print(f"  backend: {backend}")
    print(f"  planner: {V6Config.PLANNER_MODE}")
    print(f"  4-bit:   {V6Config.USE_4BIT}   speculative: {V6Config.USE_SPECULATIVE}")
    print(f"  output:  {V6Config.OUTPUT_DIR}")
    print(DIVIDER)


DEMO_QUERIES = [
    "Hello, what can you do?",
    "What does ARPU mean?",
    "What is the total revenue for Oran in 2024?",
    "Compare churn rates between Algiers and Constantine last quarter",
    "Show me a bar chart of total revenue by wilaya for 2024",
    "Which wilaya had the highest churn rate?",
]


def run_demo(agent: LatentMindV6, show_trace: bool) -> None:
    for q in DEMO_QUERIES:
        print(f"\n{DIVIDER}\nQ: {q}\n{DIVIDER}")
        res = agent.ask(q, thread_id="demo", verbose=False)
        print_result(res, show_trace)
        _clear_cache("demo")


def run_repl(agent: LatentMindV6, show_trace: bool) -> None:
    print("\nInteractive mode. Commands: :trace  :reset  :thread <id>  :send  :quit\n")
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
            print(f"  trace {'on' if show_trace else 'off'}")
            continue
        if line == ":reset":
            agent.reset()
            last_draft = None
            print("  conversation reset")
            continue
        if line.startswith(":thread"):
            parts = line.split(maxsplit=1)
            thread = parts[1] if len(parts) > 1 else thread
            print(f"  thread = {thread}")
            continue
        if line == ":send":
            if not last_draft or not last_draft.get("to"):
                print("  no email draft with a recipient")
                continue
            print(f"  sending to {last_draft['to']} ...")
            r = send_email(last_draft)
            print("  sent" if r["ok"] else f"  failed: {r['error']}")
            continue

        res = agent.ask(line, thread_id=thread)
        print_result(res, show_trace)
        _clear_cache(thread)
        last_draft = res.get("email_draft")


def main() -> None:
    ap = argparse.ArgumentParser(description="LatentMind V6 agent")
    ap.add_argument("query", nargs="?", help="a single question")
    ap.add_argument("-i", "--interactive", action="store_true",
                    help="interactive REPL")
    ap.add_argument("-t", "--trace", action="store_true",
                    help="show node trace")
    args = ap.parse_args()

    banner()
    print("\nloading models (BGE-M3 + SLM)...")
    agent = LatentMindV6()
    print("ready.\n")

    if args.query:
        res = agent.ask(args.query, thread_id="cli")
        print_result(res, args.trace)
        _clear_cache("cli")
    elif args.interactive:
        run_repl(agent, args.trace)
    else:
        run_demo(agent, args.trace)


if __name__ == "__main__":
    sys.exit(main())
