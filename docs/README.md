# LatentMind V6 — Documentation

> An agentic, voice-capable analytics assistant for an Algerian telecom operator
> (Djezzy), built as a **LangGraph policy loop** driven by a trained MLP "brain."

This `docs/` folder documents **V6 — the current system** — end to end: the
architecture, the trained brain, the RAG/knowledge layer, the SQL pipeline, the
number-formatting trust boundary, the voice layer, the serving stack, the training
recipe, the exact neural-network specs, the benchmarks, and the design decisions
behind all of it.

> **Note on numbers.** Benchmark figures (accuracy, latency, WER, exec rates) in
> these docs are illustrative placeholders meant to be replaced with measured
> values. The *methodology* behind each metric is accurate; the values are not.
> The few figures with documented provenance (e.g. the brain's 86%→97% held-out
> jump) are marked as such.

---

## The one-sentence thesis

**The decision of what to do lives in a learned latent space; the execution that
touches the database or the user's numbers is deterministic and verified.** V6
makes routing/intent/control-flow *learned* (a trained brain over BGE-M3
embeddings) while caging everything that reaches SQL or a figure behind
deterministic validators. *Dynamic decision, predictable execution.*

---

## How to read this folder

| # | Document | What it covers |
|---|---|---|
| — | **README.md** (this file) | Index + thesis |
| 01 | [Overview](01-overview.md) | What V6 does, the domain, the core principles |
| 02 | [The Latent Idea](02-the-latent-idea.md) | The encoder+MLP-in-latent-space theory V6 is built on |
| 03 | [Architecture](03-architecture.md) | The LangGraph star-topology policy loop, nodes, the SLM engine |
| 04 | [The Brain](04-the-brain.md) | The trained 3-head policy MLP — inputs, outputs, the seuil |
| 05 | [RAG, Knowledge & Entities](05-rag-knowledge-entities.md) | BGE-M3 retrieval, the four knowledge sources, wilaya resolution, live schema |
| 06 | [Data, Schema & Numbers](06-data-schema-numbers.md) | The database, glossary/catalog, and the number trust boundary (`numfmt`) |
| 07 | [Voice Layer](07-voice-layer.md) | STT (faster-whisper), TTS (XTTS-v2), the `speakable` normalizer |
| 08 | [Serving & Frontend](08-serving-frontend.md) | FastAPI + WebSocket server, the Next.js web client |
| 09 | [Training](09-training.md) | How the brain is built: data synthesis + the training recipe |
| 10 | [Neural Networks Reference](10-neural-networks-reference.md) | The `BrainHead` MLP exactly + the frozen pretrained models V6 uses |
| 11 | [Benchmarks](11-benchmarks.md) | The benchmark harness, metrics, and result tables |
| 12 | [Design Decisions](12-design-decisions.md) | The cross-cutting "why" of every major call |

The in-package reference [`v6/docs/architecture.md`](../v6/docs/architecture.md)
documents the code module-by-module; this folder is the explanatory companion.

---

## V6 at a glance

```
   speech ─► STT (faster-whisper large-v3)
                     │
   text ─────────────┤
                     ▼
              ┌───────────────┐
              │   THE BRAIN   │  trained 3-head MLP:
              │  (policy MLP) │  intent · next-action · continue-score (seuil)
              └──────┬────────┘
                     │  picks ONE action, watches the outcome, re-decides
       ┌─────────────┼───────────────────────────────┐
       ▼             ▼            ▼          ▼         ▼
      rag           sql        chart      email    template
   (BGE-M3)   (router→validate→  (typed   (draft,   (Jinja2
              generate→execute)  spec)   never send)  report)
       └─────────────┴───────────────────────────────┘
                     │ (every action loops back to the brain)
                     ▼
              communicator ──► polished answer (numbers already frozen)
                     │
                     ▼
                TTS (XTTS-v2, cloned voice) ─► spoken answer
```

Every action returns to the brain, which re-decides with the new outcome. When the
continue score drops below the seuil (default 0.5), the loop ends at the
communicator. Numbers are frozen in plain Python (`numfmt`) before any model sees
them, so the figure the user reads/hears is exactly the database value.
