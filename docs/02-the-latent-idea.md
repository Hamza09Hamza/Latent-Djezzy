# 02 — The Latent Idea

> The concept the project is named after, as it lives in **V6**: decisions are made
> in a learned latent space by an **encoder + MLP**; execution stays symbolic.

---

## 1. The thesis

A machine should **make its decisions in a latent space.** Take the input — typed
text or a transcript, in French, English, Arabic, or Darija — push it through an
**encoder** that maps it to a fixed-length vector where *meaning becomes geometry*
(things that mean the same land near each other), then let a small trainable **MLP**
read that vector and decide. **Encoder + MLP.** In V6 this is exactly the brain:
BGE-M3 turns the situation into vectors, and a 3-head MLP turns those vectors into
a decision.

---

## 2. What "latent space" means here

A **latent space** is the vector space an encoder produces, whose defining property
is **semantic similarity = geometric proximity**. "What's the churn in Oran?",
"désabonnement à Oran ?", and "وش راهو الانقطاع في وهران؟" are three different
strings that a good multilingual encoder maps to nearly the **same point**. Surface
form is gone; only the meaning's coordinates remain.

Once you are in that space, the hard problems become easy ones:

- **Intent classification** → "which region of space is this point in?" (softmax MLP)
- **Action / routing** → "given this point + context, which next step?" (MLP)
- **Knowledge retrieval** → "which catalog point is nearest?" (cosine)
- **Stopping** → "is the situation point in the done region?" (a sigmoid gate)

V6's bet: do the work to get into a good latent space, and the decisions on top can
be tiny.

---

## 3. The pattern: a frozen encoder + a small trainable MLP

The encoder is large, pretrained, and **frozen**; the head is a small MLP trained
for the specific decision. The encoder supplies *understanding*; the MLP supplies
*judgment*.

```
        situation (query + memory + outcome)
                       │
                       ▼
        ┌────────────────────────────┐   BGE-M3, FROZEN.
        │          ENCODER           │   text → a 1024-d point in latent space.
        └────────────────────────────┘
                       │
                       ▼
        latent vectors  (query_emb ⊕ memory_emb)  ⊕  outcome_vec (25-d)
                       │
                       ▼
        ┌────────────────────────────┐   the BRAIN, TRAINABLE (~534K params).
        │   MLP: Linear→ReLU→3 heads  │   intent · action · continue.
        └────────────────────────────┘
                       │
                       ▼
                   a decision
```

In V6 this is the brain (see [The Brain](04-the-brain.md)): a frozen BGE-M3 encoder
produces the latent vectors, and a sub-million-parameter MLP makes the policy
decision. The retriever uses the same encoder for cosine-nearest-neighbour
knowledge lookup — so the brain and the retriever literally share one semantic
space.

---

## 4. Why it works

**A pretrained encoder has already done the expensive part.** It has organized
meaning into a geometry where a churn question and its French paraphrase are
already neighbours — even though the brain's MLP has never seen either phrasing.
So:

1. **Little data needed.** The MLP learns *boundaries* in an already-organized
   space, not meaning from scratch — V6's brain trains on ~8K synthetic rows.
2. **Few parameters.** ~534K. The decision is cheap because the understanding is
   borrowed.
3. **It generalizes across phrasing, language, and ASR noise.** This is the reason
   V6 rejects regex routing: regex matches *surface forms* and needs a new rule per
   phrasing; an MLP over an encoder matches *positions in meaning space*, which
   generalize automatically.

| Approach | Generalizes? | Data | Params | V6 verdict |
|---|---|---|---|---|
| Regex / keyword rules | ❌ (surface only) | none | none | rejected — *"static, not acceptable"* |
| Full fine-tune of a big model | ✅ | a lot | all | overkill for routing |
| **Frozen encoder + small MLP** | ✅ | little | ~534K | **the choice** |

---

## 5. Latent communication (LatentMAS)

The latent idea applies to *communication* too. V6's router and SQL generator are
the **same** Qwen weights, so they pass information as **latent attention state (the
KV cache)** rather than re-serialized text: phase 2 inherits phase 1's internal
state and only processes the new tokens. This is the "LatentMAS" core. The honest
caveat: it is a **latency** win (skipping a re-encode of the shared context, ~1.5 s),
**not** an intelligence win — which is exactly why V6 keeps the hand-off for speed
but puts the *intelligence* (the decision) in the separately-trained brain. (See
[Architecture](03-architecture.md).)

---

## 6. Where V6 is deliberately *not* latent

The latent idea governs **decisions** — intent, routing, retrieval, the stop
signal. It **stops** at the database boundary. SQL validation, schema
introspection, entity resolution, and number formatting are **symbolic and
deterministic**. This is the dialectic the whole system balances on:

> **Dynamic decision (latent), predictable execution (symbolic).**

You decide *what to do* in latent space because that generalizes; you *do it* with
deterministic code because a query and a revenue figure must be exactly right, not
"probably right." The one place deterministic control flow is allowed —
`route_after_brain` — does not *make* the decision; it *gates* the brain's learned
decision (clamps to valid actions, enforces the step cap, blocks already-attempted
terminals).

```
   ┌──────────────── LATENT (learned, generalizes) ─────────────────┐
   │ situation ─► BGE-M3 ─► MLP brain ─► DECISION (intent / action)  │
   └─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
   ┌──────────── SYMBOLIC (deterministic, exact) ───────────────────┐
   │ validate schema · resolve entities · execute SQL · freeze nums  │
   └─────────────────────────────────────────────────────────────────┘
```

**LatentMind = "decide with an encoder + MLP in latent space; execute with code."**

→ Next: [Architecture](03-architecture.md) · the brain in full: [The Brain](04-the-brain.md).
