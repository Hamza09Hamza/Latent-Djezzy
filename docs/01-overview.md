# 01 — Overview

## What V6 is

LatentMind V6 is an **agentic analytics assistant** for the Algerian telecom market
(built around Djezzy). A user asks a question — in **French, English, or
Arabic/Darija**, by **typing or speaking** — and the system:

1. classifies the *intent* of the question (a trained brain, in embedding space),
2. retrieves the relevant domain knowledge (RAG over BGE-M3),
3. translates the question into **read-only SQL** over the operator's KPI database,
4. validates and executes that SQL and reads back the result,
5. optionally **charts** it, **drafts an email** about it, or **fills a report**,
6. writes a natural-language answer in the same language the question was asked,
7. and — on the voice channel — **speaks** that answer back in a cloned voice.

It remembers the conversation across turns, so "and for Tiaret?" after "show me
churn in Oran" inherits the right KPI.

The product persona is the **"Djezzy Voice Assistant"**: a measured, professional
analyst that only talks about telecom KPIs (revenue, ARPU, churn, subscribers,
profitability, OPEX/CAPEX) per *wilaya* (Algerian province), and politely deflects
anything off-topic.

## What makes V6 different: a trained policy loop

V6 is built as a **LangGraph state machine** in which a small trained **MLP "brain"**
picks **one action at a time**, watches what happened, and **re-decides** — until it
is confident the turn is done. There is no fixed plan and no regex routing:

```
START → brain ⇄ {rag, sql, chart, email, template} → communicator → END
```

Every action loops back to the brain. Because the brain re-decides after every
step, it **reacts** to what actually happened: it can retry a failed SQL query,
skip a chart when no rows came back, stop early when it has the answer, or stop and
clarify. This is the central design — a *learned, self-correcting policy* rather
than a committed sequence.

## The business domain

The data is **weekly KPI snapshots** at the *commune* level for an Algerian mobile
operator. Two facts shape almost every design decision:

- **Geography is Algerian wilayas.** There are **58 wilayas** (after Algeria's 2019
  reform split 48 → 58). Names are stored in **French** ("Alger", not "Algiers";
  "Béjaïa" with accents). Users type them in every spelling and script. Data is
  stored per commune, so a wilaya filter is really a filter over many `location_id`s.
- **Numbers must be exactly right.** A revenue figure off by a digit is worse than
  no figure — it is a *confidently wrong* answer a human will act on. This single
  requirement is why V6 refuses to let any language model format a raw number.

See [Data, Schema & Numbers](06-data-schema-numbers.md) for the full schema,
glossary, and KPI catalog.

## The core principles

### 1. The brain decides, everything else executes

No hardcoded routing logic in the graph. The trained MLP picks each action and
judges when to stop. Policy that belongs to "what should the system do" lives in
**training traces**, not in Python `if` statements. (See [The Brain](04-the-brain.md).)

### 2. Routing is latent, never regex

Intent and action are classified in a learned embedding space (BGE-M3), so the
system generalizes across phrasings, languages, and ASR noise. A regex classifier
would need a new rule for every phrasing and would collapse on Darija. (See
[The Latent Idea](02-the-latent-idea.md).)

### 3. Determinism at the trust boundary

Every SLM output that touches the DB or the user's numbers is post-processed by
deterministic validators before it is allowed through: static SQL validation,
schema introspection, entity resolution, consistency checks, and — crucially —
**number formatting**. Figures are rounded and unit-tagged in plain Python
(`numfmt`) *before* the model sees them; the model only copies a frozen figure.
(See [Data, Schema & Numbers](06-data-schema-numbers.md).)

### 4. One model, two roles

The Qwen SLM plays both the **router** (phase 1) and the **SQL generator** (phase
2) in the same conversation, sharing a KV cache so phase 2 inherits phase 1's
context instead of re-encoding it. (See [Architecture](03-architecture.md).)

### 5. Think small, verify hard

V6 uses **small, local, private models** (a 1024-d encoder, a 4B SLM, a
sub-million-parameter MLP) rather than a giant hosted LLM. Small models are cheap
and private but unreliable — and the answer to that is a verification cage, not a
bigger model. Most of the engineering is that cage.

## The system at a glance

```
   speech ─► STT ─┐
   text ──────────┤
                  ▼
            THE BRAIN  (intent · next-action · continue-score)
                  │  picks one action, sees the outcome, re-decides
     ┌────────────┼───────────────────────────────┐
     ▼            ▼            ▼          ▼         ▼
    rag          sql        chart      email    template
                  │
                  ▼  (loop back to the brain after every action)
            communicator ──► polished answer (numbers frozen) ──► TTS
```

Read [Architecture](03-architecture.md) next for how each box works.
