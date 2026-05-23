"""v6/brain_data.py — Synthetic agentic traces for the policy brain.

The brain is trained by behaviour cloning: this module writes out the
*policy we want* as traces, and train_brain.py fits the MLP to imitate it.

A trace is one simulated turn — an intent, a query, optional history, and
the gold sequence of (action, simulated-outcome) the brain should take.
Each trace expands into one training row per brain tick:

    {query, history, intent, step_log, grounding,
     label_action, label_continue}

`step_log` is the outcomes of the steps *before* this tick — exactly what
brain.encode_outcome() turns into the situation vector at run time. The
final tick of every trace has label_continue=0: the seuil fires and the
turn routes to the communicator.

Key design:
  - `_expand` generates ALL ticks for a trace (k=0..len(gold)).
  - `_terminal` adds ONLY the final stopping-state row for a trace.
    Used to add extra concentrated signal that the brain should STOP
    once a terminal action (chart/email/template) has succeeded.
    These rows have no corresponding continue=1 siblings, so they
    directly increase the "stop" signal density for those patterns.

This file IS the editable policy spec. Add a trace shape, retrain, and the
brain learns the new behaviour. Swap the templates for real logged turns
when you have them — that is what makes the policy genuinely dynamic.

    python3 -m v6.brain_data        # → data/brain_train.jsonl
"""

from __future__ import annotations
import json
import os
import random

from .config import V6Config

OUT_PATH = V6Config.BRAIN_TRAIN_PATH

# ── query templates (shared with the retired planner_data scaffolding) ────
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
    "nice work", "cheers", "hello there", "hey", "good evening",
    "howdy", "greetings", "thank you", "much appreciated", "appreciate it",
    "perfect, thanks", "awesome thanks", "see you later", "take care",
    "have a good day", "good day", "hiya", "ok thank you", "thanks again",
]
_META = [
    "what can you do", "who are you", "what kind of questions can i ask",
    "how do you work", "what is this", "tell me about yourself",
    "what are your capabilities", "help me get started", "what can this do",
    "how does this assistant work", "what should i ask you",
    "what are you", "explain what you do", "what features do you have",
    "give me an overview of what you can do", "what is latentmind",
    "are you an ai assistant", "what data can you access",
    "how can you help me", "what questions work best here",
    "describe yourself", "what is your purpose", "what can i ask you about",
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
_EMAIL_WRAP_NORECIP = ["email {q}", "send {q} by email", "{q} — email this out",
                       "mail {q}", "email out {q}", "{q}, send it as an email"]
_TEMPLATE_WRAP = ["{q} and put it in a report", "generate a report of {q}",
                  "fill the weekly report with {q}", "{q} as a report document"]
_FOLLOWUPS = ["and for {w}?", "what about {w}", "how about {w}",
              "now {seg}", "same for {w}", "and {w}"]

# Performance / executive-report queries — map to fpa_profitability + global_revenue
_PERFORMANCE_QUERIES = [
    "Q4 {t} performance summary", "Q3 {t} performance review",
    "Q1 {t} results", "Q2 {t} business results",
    "executive performance report {t}", "quarterly business results {t}",
    "how did we do in Q3 {t}", "annual performance review {t}",
    "H1 {t} financial results", "H2 {t} performance",
    "company performance {t}", "give me the quarterly KPI summary {t}",
    "financial summary {t}", "business performance overview {t}",
    "year-to-date financial results", "half-year performance report",
    "what were our results {t}", "overall company results {t}",
    "key financial metrics {t}", "show me the financial highlights {t}",
]


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


def _norm(q: str) -> str:
    return " ".join((q or "").split())


def _fill(template: str, kpis: list[str], rng: random.Random) -> str:
    return _norm(template.format(
        kpi=rng.choice(kpis), w=rng.choice(_WILAYAS), w2=rng.choice(_WILAYAS),
        t=rng.choice(_TIMES), seg=rng.choice(_SEGMENTS),
        fake=rng.choice(_FAKE_KPIS)))


def _proto(intent: str) -> list[str]:
    """Intent prototype phrases — hand-written gold examples, if present."""
    if not os.path.isfile(V6Config.BRAIN_PROTOTYPES):
        return []
    with open(V6Config.BRAIN_PROTOTYPES, encoding="utf-8") as f:
        return list(json.load(f).get("intents", {}).get(intent, []))


# ── simulated action outcomes ────────────────────────────────────────────
def _rag(weak: bool = False) -> dict:
    return {"action": "rag", "ok": True,
            "error_type": "rag_weak" if weak else "none",
            "row_bucket": "none", "attempt": 1}


def _sql_ok(bucket: str = "many", attempt: int = 1) -> dict:
    return {"action": "sql", "ok": True, "error_type": "none",
            "row_bucket": bucket, "attempt": attempt}


def _sql_norows() -> dict:
    return {"action": "sql", "ok": True, "error_type": "sql_no_rows",
            "row_bucket": "zero", "attempt": 1}


def _sql_fail(attempt: int) -> dict:
    return {"action": "sql", "ok": False, "error_type": "sql_error",
            "row_bucket": "none", "attempt": attempt}


def _sql_noquery() -> dict:
    return {"action": "sql", "ok": False, "error_type": "sql_no_query",
            "row_bucket": "none", "attempt": 1}


def _chart(ok: bool = True) -> dict:
    return {"action": "chart", "ok": ok,
            "error_type": "none" if ok else "artifact_failed",
            "row_bucket": "none", "attempt": 1}


def _email(ok: bool = True) -> dict:
    return {"action": "email", "ok": ok,
            "error_type": "none" if ok else "email_no_recipient",
            "row_bucket": "none", "attempt": 1}


def _template(ok: bool = True) -> dict:
    return {"action": "template", "ok": ok,
            "error_type": "none" if ok else "artifact_failed",
            "row_bucket": "none", "attempt": 1}


def _grounding_at(prior: list[dict]) -> float:
    """The grounding score known at a tick — 0 until `rag` has executed."""
    for s in prior:
        if s["action"] == "rag":
            return 0.32 if s["error_type"] == "rag_weak" else 0.72
    return 0.0


def _expand(out: list[dict], intent: str, query: str, history: str,
            gold: list[dict]) -> None:
    """Turn one trace into one training row per brain tick."""
    query = _norm(query)
    if not query:
        return
    for k in range(len(gold) + 1):
        prior = gold[:k]
        row = {
            "query": query, "history": history, "intent": intent,
            "step_log": prior, "grounding": round(_grounding_at(prior), 3),
        }
        if k < len(gold):
            row["label_action"] = gold[k]["action"]
            row["label_continue"] = 1
        else:
            row["label_action"] = None
            row["label_continue"] = 0
        out.append(row)


def _terminal(out: list[dict], intent: str, query: str, history: str,
              done_seq: list[dict]) -> None:
    """Add ONLY the terminal stopping-state row for a completed sequence.

    Unlike _expand (which generates all k=0..len(gold) ticks), this adds
    just the k=len(done_seq) row with label_continue=0. These concentrated
    stop-signal rows teach the brain to halt after a terminal action
    succeeds, without adding extra 'go' rows for the intermediate steps.
    """
    query = _norm(query)
    if not query:
        return
    out.append({
        "query": query, "history": history, "intent": intent,
        "step_log": done_seq,
        "grounding": round(_grounding_at(done_seq), 3),
        "label_action": None,
        "label_continue": 0,
    })


# ── dataset ──────────────────────────────────────────────────────────────
def build_dataset(seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    kpis = _kpi_terms()
    rows: list[dict] = []

    # greeting / meta — gold=[] → the brain stops at step 0 → communicator
    for g in sorted(set(_GREETINGS) | set(_proto("greeting"))):
        _expand(rows, "greeting", g, "", [])
    for m in sorted(set(_META) | set(_proto("meta"))):
        _expand(rows, "meta", m, "", [])

    # definition — gold=[rag], then communicator
    for _ in range(112):
        _expand(rows, "definition",
                _DEF_TEMPLATES[rng.randrange(len(_DEF_TEMPLATES))]
                .format(kpi=rng.choice(kpis)), "", [_rag()])
    for p in _proto("definition"):
        _expand(rows, "definition", p, "", [_rag()])

    # unanswerable — gold=[] → communicator says it can't be answered
    for _ in range(64):
        _expand(rows, "unanswerable",
                _fill(rng.choice(_UNANSWERABLE_TEMPLATES), kpis, rng), "", [])
    for p in _proto("unanswerable"):
        _expand(rows, "unanswerable", p, "", [])

    # a pool of plain data queries reused by every data trace shape
    base = [_fill(rng.choice(_DATA_TEMPLATES), kpis, rng) for _ in range(400)]
    base += [b for b in _proto("data") if b]

    def pick() -> str:
        return rng.choice(base)

    # plain data — rag → sql → communicator
    for _ in range(168):
        _expand(rows, "data", pick(), "", [_rag(), _sql_ok()])
    # weak grounding — the brain proceeds to sql anyway
    for _ in range(60):
        _expand(rows, "data", pick(), "", [_rag(weak=True), _sql_ok()])
    # single-row result
    for _ in range(50):
        _expand(rows, "data", pick(), "", [_rag(), _sql_ok("one")])
    # zero rows — the brain stops (a chart of nothing helps no one)
    for _ in range(90):
        _expand(rows, "data", pick(), "", [_rag(), _sql_norows()])
    # data + chart
    for _ in range(200):
        _expand(rows, "data", rng.choice(_VIZ_WRAP).format(q=pick()), "",
                [_rag(), _sql_ok(), _chart()])
    # data + email
    for _ in range(150):
        _expand(rows, "data", rng.choice(_EMAIL_WRAP).format(q=pick()), "",
                [_rag(), _sql_ok(), _email()])
    # data + report
    for _ in range(200):
        _expand(rows, "data", rng.choice(_TEMPLATE_WRAP).format(q=pick()), "",
                [_rag(), _sql_ok(), _template()])
    # data + chart + email — multi-capability via the loop
    for _ in range(90):
        q = rng.choice(_EMAIL_WRAP).format(
            q=rng.choice(_VIZ_WRAP).format(q=pick()))
        _expand(rows, "data", q, "", [_rag(), _sql_ok(), _chart(), _email()])
    # SQL fails once → retry → succeeds
    for _ in range(100):
        _expand(rows, "data", pick(), "",
                [_rag(), _sql_fail(1), _sql_ok("many", attempt=2)])
    # SQL fails twice → give up (one retry only)
    for _ in range(70):
        _expand(rows, "data", pick(), "",
                [_rag(), _sql_fail(1), _sql_fail(2)])
    # no query can be formed at all (too vague) → stop, do not retry
    for _ in range(60):
        _expand(rows, "data", pick(), "", [_rag(), _sql_noquery()])
    # email with no recipient named → cannot retry → stop
    for _ in range(90):
        _expand(rows, "data", rng.choice(_EMAIL_WRAP_NORECIP).format(q=pick()),
                "", [_rag(), _sql_ok(), _email(ok=False)])
    # follow-ups — a terse query with prior context in memory
    for _ in range(120):
        prev = pick()
        _expand(rows, "data", _fill(rng.choice(_FOLLOWUPS), kpis, rng),
                f"Earlier question: {prev}", [_rag(), _sql_ok()])

    # ── performance / executive-report traces ────────────────────────────
    # These map broad "Q4 performance", "quarterly results" etc. to the
    # fpa_profitability + global_revenue tables. The gold sequence is
    # rag → sql → template (a performance question deserves a report).
    for _ in range(180):
        pt = rng.choice(_PERFORMANCE_QUERIES)
        q = _norm(pt.format(t=rng.choice(_TIMES), w=rng.choice(_WILAYAS),
                            kpi="performance", seg="", fake=""))
        _expand(rows, "data", q, "", [_rag(), _sql_ok(), _template()])
    # some performance queries just want the numbers (no report)
    for _ in range(80):
        pt = rng.choice(_PERFORMANCE_QUERIES)
        q = _norm(pt.format(t=rng.choice(_TIMES), w=rng.choice(_WILAYAS),
                            kpi="performance", seg="", fake=""))
        _expand(rows, "data", q, "", [_rag(), _sql_ok()])

    # ── terminal-stop augmentation ───────────────────────────────────────
    # These are ONLY the final stopping-state rows — no intermediate ticks.
    # They directly increase the density of "I'm done, stop here" training
    # signal for the three terminal actions, correcting the repeat-action bug
    # where the brain kept picking chart/email/template a second time.

    # chart done → stop
    for _ in range(400):
        q = rng.choice(_VIZ_WRAP).format(q=pick())
        _terminal(rows, "data", q, "", [_rag(), _sql_ok(), _chart(ok=True)])

    # email done → stop
    for _ in range(300):
        q = rng.choice(_EMAIL_WRAP).format(q=pick())
        _terminal(rows, "data", q, "", [_rag(), _sql_ok(), _email(ok=True)])

    # template done → stop
    for _ in range(300):
        q = rng.choice(_TEMPLATE_WRAP).format(q=pick())
        _terminal(rows, "data", q, "", [_rag(), _sql_ok(), _template(ok=True)])

    # chart + email done (multi-capability terminal) → stop
    for _ in range(150):
        q = rng.choice(_EMAIL_WRAP).format(
            q=rng.choice(_VIZ_WRAP).format(q=pick()))
        _terminal(rows, "data", q, "",
                  [_rag(), _sql_ok(), _chart(ok=True), _email(ok=True)])

    # performance report done → stop
    for _ in range(200):
        pt = rng.choice(_PERFORMANCE_QUERIES)
        q = _norm(pt.format(t=rng.choice(_TIMES), w=rng.choice(_WILAYAS),
                            kpi="performance", seg="", fake=""))
        _terminal(rows, "data", q, "",
                  [_rag(), _sql_ok(), _template(ok=True)])

    rng.shuffle(rows)
    return rows


def main() -> None:
    rows = build_dataset()
    os.makedirs(V6Config.DATA_DIR, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by_intent: dict = {}
    by_action: dict = {}
    cont = [0, 0]
    for r in rows:
        by_intent[r["intent"]] = by_intent.get(r["intent"], 0) + 1
        if r["label_action"]:
            by_action[r["label_action"]] = by_action.get(r["label_action"], 0) + 1
        cont[r["label_continue"]] += 1
    print(f"wrote {len(rows)} training rows → {OUT_PATH}")
    print("  by intent       :", by_intent)
    print("  by action label :", by_action)
    print(f"  continue 0/1    : {cont[0]} stop / {cont[1]} go")
    print("  next: python3 -m v6.train_brain")


if __name__ == "__main__":
    main()
