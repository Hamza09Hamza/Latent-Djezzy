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
import re

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
# off-topic / out-of-scope — code, trivia, world facts, translation, math,
# recommendations, recipes, creative writing, general advice. These are NOT
# telecom analytics and must NEVER reach the SQL pipeline. We label them
# `meta` because `meta` already routes to the warm chat persona, whose SCOPE
# GUARD deflects off-topic and steers back to what the assistant can do. The
# brain only needs to learn "this is not data/definition" → chat. Bilingual,
# because real users (and the benchmark) mix French and English freely.
_OFFTOPIC = [
    # code / programming
    "write me a python function to sort a list", "write me a python script",
    "can you write some code", "write a function to reverse a string",
    "help me debug my javascript", "how do i write a for loop in python",
    "fix this code for me", "what's the syntax for a sql join in mysql",
    "build me a react component", "explain how recursion works",
    # world facts / trivia
    "what's the weather today", "what's the weather in algiers",
    "who won the world cup", "who is the president of france",
    "what's the capital of japan", "how tall is mount everest",
    "what year did world war two end", "who painted the mona lisa",
    # translation / language
    "translate hello to spanish", "translate this sentence to arabic",
    "how do you say thank you in italian", "what does bonjour mean in english",
    # math / general computation
    "what's 15 times 23", "solve this math equation for me",
    "what's the square root of 144", "convert 100 dollars to euros",
    # recommendations / advice / lifestyle
    "recommend a good restaurant in oran", "give me a recipe for couscous",
    "what movie should i watch tonight", "how do i cook pasta",
    "plan my vacation to morocco", "what stocks should i buy",
    "give me some workout tips", "how do i lose weight",
    # creative / personal
    "write me a poem", "tell me a story", "sing me a song",
    "explain quantum physics", "help me write an essay",
    "what's the meaning of life", "set an alarm for 7am",
    "summarize this article for me",
]
_OFFTOPIC_FR = [
    # code
    "écris-moi une fonction python", "écris-moi du code",
    "peux-tu écrire un script python", "corrige ce code pour moi",
    "comment écrire une boucle for en python",
    # faits / culture générale
    "quel temps fait-il aujourd'hui", "qui a gagné la coupe du monde",
    "qui est le président de la france", "quelle est la capitale du japon",
    "en quelle année s'est terminée la seconde guerre mondiale",
    # traduction
    "traduis bonjour en anglais", "comment dit-on merci en italien",
    "que veut dire hello en français",
    # maths
    "combien font 15 fois 23", "quelle est la racine carrée de 144",
    "convertis 100 dollars en euros",
    # recommandations / conseils
    "recommande-moi un bon restaurant", "donne-moi une recette de couscous",
    "quel film regarder ce soir", "comment cuisiner des pâtes",
    "planifie mes vacances", "donne-moi des conseils sportifs",
    # créatif / personnel
    "écris-moi un poème", "raconte-moi une histoire",
    "explique-moi la physique quantique", "aide-moi à écrire un essai",
    "résume cet article", "quelle heure est-il à tokyo",
]
_FAKE_KPIS = ["quantum score", "blockchain ratio", "customer happiness index",
              "satellite uptime", "employee morale rate", "stock price",
              "carbon footprint", "brand sentiment score", "nps trend",
              "website traffic",
              # out-of-schema metrics that LOOK like real telecom KPIs — the
              # hardest unanswerables (the b16 "network satisfaction" miss).
              "network satisfaction score", "satisfaction score",
              "customer satisfaction", "net promoter score",
              "service quality index", "network coverage rating",
              "call drop rate", "complaint resolution time"]
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

# Cross-turn "just put the previous result in a report" follow-ups.
# These are short standalone queries — no SQL intent — where prior data
# is already in the conversation memory. The gold action is template only.
_CROSST_TEMPLATE_QS = [
    "put it in a report", "put that in a report", "make a report",
    "generate a report", "save it as a report", "create a report",
    "put this in a report", "turn it into a report", "build a report",
    "mets-le dans un rapport", "mets ça dans un rapport",
    "génère un rapport", "crée un rapport", "fais un rapport",
    "dans un rapport s'il te plaît", "rapport", "un rapport",
    "mets les résultats dans un rapport", "exporte en rapport",
]

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


# ── French templates ──────────────────────────────────────────────────────
# The benchmark and real users mix French freely, but the brain's traces were
# almost all English — so French capability VERBS (Trace / Envoie / Mets dans
# un rapport) were under-trained and STT noise tipped them (b10 chart lost,
# b12 email→report). These mirror the English shapes in French.
_TIMES_FR = ["", "", "", "le mois dernier", "le trimestre dernier",
             "cette année", "la semaine dernière", "récemment"]
_SEGMENTS_FR = ["", "", "prépayés", "postpayés"]
_KPI_FR = ["revenu", "revenu net", "revenu total", "marge brute", "ARPU",
           "taux de désabonnement", "abonnés", "EBITDA", "OPEX", "CAPEX",
           "part de marché", "taux de migration", "taux de recharge",
           "bénéfice net", "chiffre d'affaires"]
_DATA_TEMPLATES_FR = [
    "quel est le {kpi} à {w}", "quel était le {kpi} à {w} {tf}",
    "montre le {kpi} pour les abonnés {segf}", "tendance du {kpi} pour {segf}",
    "compare le {kpi} entre {w} et {w2}", "{kpi} moyen {tf}",
    "combien de {kpi} {w} a enregistré {tf}", "quelle wilaya a le plus de {kpi}",
    "{kpi} total {tf}", "{kpi} par wilaya", "donne-moi le {kpi} pour {w}",
    "{kpi} à {w} {tf}", "top 5 des wilayas par {kpi}", "{kpi} {segf} {tf}",
]
_DEF_TEMPLATES_FR = [
    "c'est quoi le {kpi}", "c'est quoi exactement le {kpi}",
    "que veut dire {kpi}", "définis le {kpi}", "explique le {kpi}",
    "qu'est-ce que le {kpi}", "que signifie {kpi}",
]
_VIZ_WRAP_FR = ["trace {q}", "trace la tendance de {q}", "affiche {q}",
                "montre-moi {q} sous forme de graphique", "{q} en graphique",
                "trace l'évolution de {q}", "fais un graphique de {q}"]
_EMAIL_WRAP_FR = ["envoie {q} au directeur financier",
                  "envoie {q} au responsable des opérations",
                  "envoie {q} à Sarah", "{q} et envoie-le à l'équipe",
                  "transfère {q} au manager"]
_EMAIL_WRAP_NORECIP_FR = ["envoie {q} par email", "envoie {q}",
                          "{q} — envoie ça par mail", "envoie {q} par mail"]
_TEMPLATE_WRAP_FR = ["{q} et mets-le dans un rapport",
                     "génère un rapport de {q}", "mets {q} dans un rapport",
                     "{q} sous forme de rapport"]
_FOLLOWUPS_FR = ["et pour {w} ?", "et à {w}", "pareil pour {w}",
                 "et {w} ?", "maintenant {segf}", "et le {segf} ?"]
_UNANSWERABLE_TEMPLATES_FR = [
    "quel est le {fakef}", "montre-moi le {fakef}",
    "donne-moi le {fakef} pour {w}", "quel était le {fakef} {tf}",
]
_FAKE_KPIS_FR = ["score de satisfaction réseau", "indice de satisfaction client",
                 "score de satisfaction", "cours de l'action",
                 "empreinte carbone", "note de qualité de service",
                 "taux de couverture réseau", "score NPS",
                 "temps de résolution des plaintes"]

# ── noise augmentation ──────────────────────────────────────────────────────
# Meaning-preserving STT-style perturbations. We NEVER touch the action verb
# (it carries the capability signal); only entities, the word "wilaya", and
# punctuation/casing are perturbed — exactly the noise that flipped borderline
# decisions between the clean-text and voice runs.
_WILAYA_MISSPELL = {
    "Bejaia": ["Vijaya", "Bejaya", "Bijaya"], "Setif": ["Sétif", "Setiff"],
    "Tlemcen": ["Tlemcem", "Tlemsen"], "Ouargla": ["Wargla", "Ouergla"],
    "Tizi Ouzou": ["Tizi Ouzu", "Tizi Wuzu"], "Annaba": ["Anaba"],
    "Constantine": ["Constantin"], "Mostaganem": ["Mostaganem"],
}


def _noise(q: str, rng: random.Random) -> str:
    """A meaning-preserving perturbation of a query (label unchanged)."""
    out = q
    for canon, variants in _WILAYA_MISSPELL.items():
        if canon.lower() in out.lower() and rng.random() < 0.5:
            out = re.sub(re.escape(canon), rng.choice(variants), out,
                         flags=re.IGNORECASE)
    if "wilaya" in out and rng.random() < 0.4:
        out = out.replace("wilaya", rng.choice(["willaya", "walaya"]))
    if rng.random() < 0.5:
        out = out + "?"
    if rng.random() < 0.25:
        out = out.capitalize()
    return _norm(out)


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
    # All slot kinds are always supplied; .format() consumes only the ones the
    # template references, so English and French templates share this helper.
    return _norm(template.format(
        kpi=rng.choice(kpis), w=rng.choice(_WILAYAS), w2=rng.choice(_WILAYAS),
        t=rng.choice(_TIMES), seg=rng.choice(_SEGMENTS),
        tf=rng.choice(_TIMES_FR), segf=rng.choice(_SEGMENTS_FR),
        fake=rng.choice(_FAKE_KPIS), fakef=rng.choice(_FAKE_KPIS_FR)))


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

    # off-topic / out-of-scope — its OWN intent so the communicator can give a
    # deterministic canned deflection (never routed through the polisher, so a
    # small model can't be coaxed into writing the code/translation). gold=[]
    # → stop at step 0 → communicator. The uniques anchor the region; a noised
    # batch broadens it so casing / punctuation / a stray wilaya name can't flip
    # the decision.
    offtopic = sorted(set(_OFFTOPIC) | set(_OFFTOPIC_FR) | set(_proto("off_topic")))
    for q in offtopic:
        _expand(rows, "off_topic", q, "", [])
    for _ in range(140):
        _expand(rows, "off_topic", _noise(rng.choice(offtopic), rng), "", [])

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

    # ── cross-turn template follow-ups ───────────────────────────────────
    # User ran SQL in a previous turn; now just asks to put the result into
    # a report. No rag or sql needed — data is already in memory.
    # Gold: [template] immediately (continue=1 for the single step, then stop).
    _data_memory = [
        "Q: {q}\nA: gross_margin: 42.39% | 1 row",
        "Q: {q}\nA: 24 rows returned: wilaya | avg_arpu | ...",
        "Q: {q}\nA: total_revenue: 1,234,567.00 | 1 row",
        "Q: {q}\nA: churn_rate: 0.0412 | avg_arpu: 3.21 | 1 row",
        "Q: {q}\nA: 58 rows returned: wilaya | net_adds | ...",
    ]
    for _ in range(350):
        mem = rng.choice(_data_memory).format(q=pick())
        q = rng.choice(_CROSST_TEMPLATE_QS)
        _expand(rows, "data", q, mem, [_template()])
    # terminal stop: after template succeeds, definitely stop
    for _ in range(200):
        mem = rng.choice(_data_memory).format(q=pick())
        q = rng.choice(_CROSST_TEMPLATE_QS)
        _terminal(rows, "data", q, mem, [_template(ok=True)])

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

    # ── French traces ────────────────────────────────────────────────────
    # Mirror the English shapes so French capability verbs are well-trained.
    kpis_fr = sorted(set(kpis) | set(_KPI_FR))
    base_fr = [_fill(rng.choice(_DATA_TEMPLATES_FR), kpis_fr, rng)
               for _ in range(260)]

    def pick_fr() -> str:
        return rng.choice(base_fr)

    # French definitions
    for _ in range(90):
        _expand(rows, "definition",
                _norm(rng.choice(_DEF_TEMPLATES_FR).format(
                    kpi=rng.choice(kpis_fr))), "", [_rag()])
    # French plain data
    for _ in range(150):
        _expand(rows, "data", pick_fr(), "", [_rag(), _sql_ok()])
    for _ in range(40):
        _expand(rows, "data", pick_fr(), "", [_rag(), _sql_norows()])
    # French data + chart
    for _ in range(130):
        _expand(rows, "data", rng.choice(_VIZ_WRAP_FR).format(q=pick_fr()),
                "", [_rag(), _sql_ok(), _chart()])
    for _ in range(150):
        q = rng.choice(_VIZ_WRAP_FR).format(q=pick_fr())
        _terminal(rows, "data", q, "", [_rag(), _sql_ok(), _chart(ok=True)])
    # French data + email
    for _ in range(110):
        _expand(rows, "data", rng.choice(_EMAIL_WRAP_FR).format(q=pick_fr()),
                "", [_rag(), _sql_ok(), _email()])
    for _ in range(120):
        q = rng.choice(_EMAIL_WRAP_FR).format(q=pick_fr())
        _terminal(rows, "data", q, "", [_rag(), _sql_ok(), _email(ok=True)])
    for _ in range(60):
        _expand(rows, "data",
                rng.choice(_EMAIL_WRAP_NORECIP_FR).format(q=pick_fr()),
                "", [_rag(), _sql_ok(), _email(ok=False)])
    # French data + report
    for _ in range(130):
        _expand(rows, "data", rng.choice(_TEMPLATE_WRAP_FR).format(q=pick_fr()),
                "", [_rag(), _sql_ok(), _template()])
    for _ in range(150):
        q = rng.choice(_TEMPLATE_WRAP_FR).format(q=pick_fr())
        _terminal(rows, "data", q, "", [_rag(), _sql_ok(), _template(ok=True)])
    # French follow-ups
    for _ in range(100):
        prev = pick_fr()
        _expand(rows, "data", _fill(rng.choice(_FOLLOWUPS_FR), kpis_fr, rng),
                f"Question précédente : {prev}", [_rag(), _sql_ok()])
    # French unanswerable (out-of-schema KPIs that look real)
    for _ in range(70):
        _expand(rows, "unanswerable",
                _fill(rng.choice(_UNANSWERABLE_TEMPLATES_FR), kpis_fr, rng),
                "", [])
    # extra English unanswerable concentrating on the new look-real fakes
    for _ in range(40):
        _expand(rows, "unanswerable",
                _fill(rng.choice(_UNANSWERABLE_TEMPLATES), kpis, rng), "", [])

    # ── noise augmentation ───────────────────────────────────────────────
    # Same gold labels, STT-perturbed queries (EN + FR). Broadens embedding
    # coverage so a misspelled wilaya or trailing "?" doesn't flip a decision.
    noisy_pool = base + base_fr
    for _ in range(240):
        _expand(rows, "data", _noise(rng.choice(noisy_pool), rng), "",
                [_rag(), _sql_ok()])
    for _ in range(120):
        src = rng.choice(_VIZ_WRAP + _VIZ_WRAP_FR).format(
            q=rng.choice(noisy_pool))
        _expand(rows, "data", _noise(src, rng), "",
                [_rag(), _sql_ok(), _chart()])
    for _ in range(120):
        src = rng.choice(_EMAIL_WRAP + _EMAIL_WRAP_FR).format(
            q=rng.choice(noisy_pool))
        _expand(rows, "data", _noise(src, rng), "",
                [_rag(), _sql_ok(), _email()])
    for _ in range(120):
        src = rng.choice(_TEMPLATE_WRAP + _TEMPLATE_WRAP_FR).format(
            q=rng.choice(noisy_pool))
        _expand(rows, "data", _noise(src, rng), "",
                [_rag(), _sql_ok(), _template()])

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
