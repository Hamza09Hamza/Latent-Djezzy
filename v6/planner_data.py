"""v6/planner_data.py — Synthetic training data for the MLP planner head.

Fills slot templates with real KPI names (data/kpi_catalog.json) and wilaya
names to produce labeled (query, history, intent, capabilities) examples.
The prototype phrases from data/planner_prototypes.json are included as-is so
the trained head never regresses below prototype mode.

    python3 -m v6.planner_data        # → data/planner_train.jsonl

This is scaffolding: it lets you train the head whenever you want without
hand-labelling. Replace or extend it with real logged queries when you have
them — that is what makes the planner genuinely "dynamic" over time.
"""

from __future__ import annotations
import json
import os
import random

from .config import V6Config

OUT_PATH = os.path.join(V6Config.DATA_DIR, "planner_train.jsonl")

_WILAYAS = ["Oran", "Alger", "Constantine", "Annaba", "Setif", "Blida",
            "Batna", "Tlemcen", "Bejaia", "Tizi Ouzou", "Ouargla", "Biskra",
            "Tiaret", "Mostaganem", "Skikda", "Djelfa", "Adrar", "Mascara"]
_TIMES = ["", "", "", "last month", "last quarter", "this year",
          "last week", "last 4 weeks", "recently"]
_SEGMENTS = ["", "", "prepaid", "postpaid"]

_DATA_TEMPLATES = [
    "what is the {kpi} in {w}", "what was the {kpi} in {w} {t}",
    "show the {kpi} for {seg} subscribers", "{kpi} trend for {seg}",
    "compare {kpi} between {w} and {w2}", "average {kpi} {t}",
    "how much {kpi} did {w} record {t}", "which wilaya has the highest {kpi}",
    "total {kpi} {t}", "{kpi} by wilaya", "give me the {kpi} for {w}",
    "{kpi} in {w} {t}", "top 5 wilayas by {kpi}", "{seg} {kpi} {t}",
]
_DEF_TEMPLATES = [
    "what does {kpi} mean", "define {kpi}", "explain {kpi}",
    "what is {kpi}", "tell me what {kpi} means",
    "what does {kpi} stand for", "what is the meaning of {kpi}",
]
_GREETINGS = [
    "hello", "hi", "hey there", "good morning", "good afternoon",
    "thanks", "thank you so much", "thanks a lot", "bye", "goodbye",
    "salam", "bonjour", "hi, how are you", "ok thanks", "great, thank you",
    "nice work", "cheers", "hello there",
]
_META = [
    "what can you do", "who are you", "what kind of questions can i ask",
    "how do you work", "what is this", "tell me about yourself",
    "what are your capabilities", "help me get started", "what can this do",
    "how does this assistant work", "what should i ask you",
]
_FAKE_KPIS = ["quantum score", "blockchain ratio", "customer happiness index",
              "satellite uptime", "employee morale rate", "stock price",
              "carbon footprint", "brand sentiment score", "nps trend",
              "website traffic"]
_UNANSWERABLE_TEMPLATES = [
    "what is the {fake}", "show me the {fake}", "give me the {fake} for {w}",
    "what was the {fake} {t}",
]
_VIZ_WRAP = ["{q} as a chart", "chart {q}", "plot {q}", "visualize {q}",
             "{q}, show me a graph", "{q} in a bar chart", "draw {q}"]
_EMAIL_WRAP = ["email {q} to the finance director", "send {q} to Sarah",
               "{q} and mail it to the team", "send {q} to operations",
               "forward {q} to the manager"]
_TEMPLATE_WRAP = ["{q} and put it in a report", "generate a report of {q}",
                  "fill the weekly report with {q}", "{q} as a report document"]
_FOLLOWUPS = ["and for {w}?", "what about {w}", "how about {w}",
              "now {seg}", "same for {w}", "and {w}"]


def _kpi_terms() -> list[str]:
    terms: set[str] = set()
    if os.path.isfile(V6Config.KPI_CATALOG_PATH):
        with open(V6Config.KPI_CATALOG_PATH, encoding="utf-8") as f:
            for e in json.load(f):
                terms.add(e["column"].replace("_", " "))
                for s in e.get("synonyms", [])[:3]:
                    if s.isascii() and 2 < len(s) < 28:
                        terms.add(s.lower())
    return sorted(terms) or ["revenue", "churn rate", "arpu", "subscribers"]


def _fill(template: str, kpis: list[str], rng: random.Random) -> str:
    return template.format(
        kpi=rng.choice(kpis), w=rng.choice(_WILAYAS), w2=rng.choice(_WILAYAS),
        t=rng.choice(_TIMES), seg=rng.choice(_SEGMENTS),
        fake=rng.choice(_FAKE_KPIS)).replace("  ", " ").strip()


def build_dataset(seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    kpis = _kpi_terms()
    rows: list[dict] = []

    def add(query, intent, caps=None, history=""):
        q = " ".join(query.split())
        if q:
            rows.append({"query": q, "history": history,
                         "intent": intent, "caps": caps or []})

    # greeting / meta
    for g in _GREETINGS:
        add(g, "greeting")
    for m in _META:
        add(m, "meta")

    # definitions
    for tmpl in _DEF_TEMPLATES:
        for _ in range(14):
            add(tmpl.format(kpi=rng.choice(kpis)), "definition")

    # plain data
    base_queries: list[str] = []
    for tmpl in _DATA_TEMPLATES:
        for _ in range(22):
            q = _fill(tmpl, kpis, rng)
            base_queries.append(q)
            add(q, "data")

    # unanswerable
    for tmpl in _UNANSWERABLE_TEMPLATES:
        for _ in range(14):
            add(_fill(tmpl, kpis, rng), "unanswerable")

    # capability-wrapped data
    for wrap, cap in ((_VIZ_WRAP, "viz"), (_EMAIL_WRAP, "email"),
                      (_TEMPLATE_WRAP, "template")):
        for _ in range(150):
            base = rng.choice(base_queries)
            add(rng.choice(wrap).format(q=base), "data", [cap])
    # multi-capability
    for _ in range(80):
        base = rng.choice(base_queries)
        viz = rng.choice(_VIZ_WRAP).format(q=base)
        add(rng.choice(_EMAIL_WRAP).format(q=viz), "data", ["viz", "email"])

    # follow-ups (history-dependent data turns)
    for _ in range(160):
        prev = rng.choice(base_queries)
        fu = _fill(rng.choice(_FOLLOWUPS), kpis, rng)
        add(fu, "data", [], history=prev)

    # seed with the prototype phrases so the head matches prototype mode
    if os.path.isfile(V6Config.PLANNER_PROTOTYPES):
        with open(V6Config.PLANNER_PROTOTYPES, encoding="utf-8") as f:
            protos = json.load(f)
        for intent, phrases in protos.get("intents", {}).items():
            for p in phrases:
                add(p, intent)
        for cap, phrases in protos.get("capabilities", {}).items():
            for p in phrases:
                add(p, "data", [cap])

    rng.shuffle(rows)
    return rows


def main() -> None:
    rows = build_dataset()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    by_intent: dict = {}
    for r in rows:
        by_intent[r["intent"]] = by_intent.get(r["intent"], 0) + 1
    print(f"wrote {len(rows)} examples → {OUT_PATH}")
    print("  by intent:", by_intent)
    print("  next: python3 -m v6.train_planner")


if __name__ == "__main__":
    main()
