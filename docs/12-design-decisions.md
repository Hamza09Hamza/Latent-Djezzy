# 12 — Design Decisions & Lessons

The cross-cutting "why" of V6 in one place. Each entry is a decision, the reasoning,
and — where it applies — the failure mode it was built to prevent.

---

## The decisions

### 1. Routing is latent, never regex

**Decision.** Intent and action are classified in a learned BGE-M3 embedding space
by the trained brain. A fully deterministic regex orchestrator was explicitly
rejected — *"static stuff that's not gonna be acceptable."*

**Why.** Regex classifiers don't generalize: every new phrasing needs a new rule,
and they collapse on Darija, code-switching, and ASR noise. An MLP over an encoder
generalizes across all of them.

**The line that keeps it honest.** Deterministic code is allowed **only** for
fact-checks against the live schema and for assembling a validated plan — that is
*validation*, not *classification*. `route_after_brain` doesn't *make* the decision;
it *gates* the brain's learned one. **Dynamic decision, predictable execution.**

### 2. The brain decides, everything else executes

**Decision.** No routing `if`-statements in the graph. The trained MLP picks each
action and judges when to stop; the policy lives in editable training traces
(`brain_data.py`).

**Why.** Heuristics that mimic the brain's job ("if intent==data and no rag yet, go
to rag") are *policy*. If they live in Python, the policy splits across code and
data and drifts. Keeping it all in traces means there's one place that defines what
the agent does.

### 3. A self-correcting loop, not a one-shot plan

**Decision.** The brain re-decides after **every** action instead of committing to a
plan up front.

**Why — the failure mode.** A one-shot planner can't react to a failed SQL query, a
zero-row result, or an email with no recipient. The brain's 25-d outcome vector
exposes exactly those signals, and the trained policy reacts: retry SQL, skip the
chart, stop and clarify.

### 4. The seuil gates everything, including terminals

**Decision.** The continue score gates *all* actions, even chart/email/template.

**Why.** A brain that exempts terminals can fire a report on every SQL result — the
action argmax might be "template" even when the brain is done (continue ≈ 0.01).
Treating the seuil as "do I want *another* step, regardless of which" makes a
deliberate "put it in a report" (trained to continue ≈ 1.0) different from an
incidental argmax (continue ≈ 0.01).

### 5. Determinism at the trust boundary

**Decision.** Every model output that touches the DB or the user's numbers is
post-processed by deterministic validators first: static SQL validation, live schema
introspection, entity resolution, consistency checks, frozen number formatting.

**Why.** A probabilistic model *will* eventually hallucinate a column, a wilaya, a
join, or a number. The answer is a verification cage, not a bigger model. Four
independent layers each catch a different class of mistake.

### 6. Numbers are frozen in Python, never formatted by a model

**Decision.** `numfmt.py` rounds, scales, and unit-tags every figure
deterministically *before* the polisher, the chart, or the TTS sees it. The model
may only **copy** a frozen figure.

**Why — the bug.** A 1.5B polisher turned `1,087,355,290.78` into `52,590,189,81`
while "rounding." For an analytics tool, a confidently wrong number is the worst
possible failure. The raw 12-digit value that invited corruption no longer exists in
the model's input. The same frozen string flows to prose, chart, and voice — and a
`speakable` net collapses any long number that still slips through.

### 7. Think small, verify hard

**Decision.** Small, local, private models (a 1024-d encoder, a 4B SLM, a ~534K
MLP) wrapped in verification — not a giant hosted LLM.

**Why.** Small models are cheap, fast, and keep the operator's data on-prem. They
are also unreliable — and that is an engineering problem with an engineering answer
(the cage), not a reason to reach for a bigger model. Most of the codebase is the
cage.

### 8. Train only the policy; rent everything else

**Decision.** V6 trains exactly one network (the ~534K brain). SQL, embeddings,
STT, and TTS are pretrained and used as-is.

**Why.** An instruction-tuned SLM already knows SQL; BGE-M3 already embeds
multilingual text; XTTS already does voice. The leverage is in *routing* and
*caging* those models. The only thing worth training is the thing nothing
off-the-shelf provides: the decision of what to do.

### 9. One model, two roles, shared KV cache (LatentMAS)

**Decision.** The router and SQL generator are the same Qwen weights; phase 2
inherits phase 1's KV cache instead of re-encoding the shared context.

**Why — and the caveat.** It saves ~1.5 s/query. But it does **not** make the model
smarter — it's a latency win only. So the *intelligence* (the decision) lives in the
separately-trained brain, and the hand-off is kept purely for speed. Knowing the
limit of the idea is part of using it well.

### 10. One source of truth for language

**Decision.** `slm.lang_code` is the single function that decides `ar`/`fr`/`en`;
the written answer, the persona, the off-topic deflection, and the spoken voice all
delegate to it.

**Why.** Otherwise a small model mirrors the (possibly French) conversation memory
instead of answering the current (English) question, and the voice speaks the wrong
language. Centralizing it — plus an explicit "Reply ONLY in <lang>" line — locks
written and spoken language together.

### 11. Never let a small model attempt an off-topic task

**Decision.** `off_topic` and `unanswerable` return a fixed, language-matched
deflection and are **never** sent to the polisher.

**Why.** A free rewrite leaks scope ("to find the stock price, check financial
news…") and a small model can be coaxed into *attempting* the off-topic request. A
deterministic deflection can't be coaxed.

### 12. Grade the learned router on held-out data

**Decision.** The brain has a hand-written held-out eval (`brain_eval.jsonl`) whose
phrasings are not drawn from the training templates.

**Why.** High accuracy on the synthetic split can be template memorization. The
held-out eval exposed that the brain was at ~86%, and targeted data raised it to
~97%. Never grade a router on the data that trained it.

---

## The lessons, distilled

1. **Reliability comes from the cage, not the model.** V6 trusts the model as
   little as possible and verifies as much as possible.
2. **Train the policy, rent the capabilities.** Pretrained models for
   SQL/embeddings/voice; train only the ~534K decision policy.
3. **A learned stop signal makes a loop safe.** The seuil is what turns "re-decide
   forever" into "re-decide until done."
4. **Know the limit of your clever idea.** KV hand-off is fast, not smart — so the
   intelligence lives elsewhere.
5. **A wrong number is the only unforgivable bug** in an analytics tool — so the
   number boundary is deterministic, redundant (Python freeze + voice safety net),
   and verified figure-by-figure.

---

## The takeaway

V6 is not "an LLM wrapper." It is a demonstration that **a trustworthy analytics
agent can be built from small, local models** — *if* you make the decisions learned
and the execution verified. **Decide in latent space, execute deterministically,
and never trust a small model with a raw number.**

← Back to the [index](README.md).
