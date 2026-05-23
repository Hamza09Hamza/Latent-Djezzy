"""v6/knowledge.py — BGE-M3 retrieval over the KPI + wilaya knowledge base.

A small in-memory vector store grounds the router SLM. Chunks come from
four sources:
  - the database's own `data_catalog` table (authoritative column docs),
  - data/kpi_catalog.json (multilingual synonyms per KPI),
  - data/glossary.json (definitions, business context, join rules),
  - the live `dim_location` table joined with data/wilaya_aliases.json
    (every wilaya as `name → location_id`, with English/Arabic/historical
    aliases). The SLM uses these chunks to write `WHERE location_id = N`
    instead of fragile `WHERE wilaya = '<name>'`, so misspellings and
    French/English variants stop silently dropping rows.

At query time the top-k chunks and a grounding score (the top cosine) are
returned. That score is one of the deterministic signals the orchestrator
uses to judge whether a query is answerable.
"""

from __future__ import annotations
import json
import os

import torch
import torch.nn.functional as F
from torch import Tensor

from .config import V6Config
from .schema import db_connect


# ── encoder ──────────────────────────────────────────────────────────────
class BGEM3Encoder:
    """Frozen BGE-M3 → CLS token → L2-normalized 1024-d vector."""

    def __init__(self, device: torch.device | None = None):
        from transformers import AutoModel, AutoTokenizer

        self.device = device or V6Config.encoder_device()
        enc_id = V6Config.bge_m3_id()
        self.tokenizer = AutoTokenizer.from_pretrained(enc_id)
        self.model = AutoModel.from_pretrained(enc_id).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, text: str) -> Tensor:
        return self.encode_batch([text])[0]

    @torch.no_grad()
    def encode_batch(self, texts: list[str], batch_size: int = 16) -> Tensor:
        if not texts:
            return torch.empty(0, V6Config.EMBED_DIM)
        out: list[Tensor] = []
        for i in range(0, len(texts), batch_size):
            sub = texts[i:i + batch_size]
            tok = self.tokenizer(
                sub, return_tensors="pt", max_length=512,
                truncation=True, padding=True).to(self.device)
            emb = self.model(**tok).last_hidden_state[:, 0, :]
            emb = F.normalize(emb, dim=-1)
            out.append(emb.cpu())
        return torch.cat(out, dim=0)


_encoder: BGEM3Encoder | None = None


def get_encoder() -> BGEM3Encoder:
    global _encoder
    if _encoder is None:
        _encoder = BGEM3Encoder()
    return _encoder


# ── chunk building ───────────────────────────────────────────────────────
def _load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_wilaya_aliases() -> dict:
    """Read data/wilaya_aliases.json — {canonical: [aliases]}. Optional."""
    if not os.path.isfile(V6Config.WILAYA_ALIASES_PATH):
        return {}
    try:
        with open(V6Config.WILAYA_ALIASES_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        return {k: v for k, v in raw.items()
                if not k.startswith("_") and isinstance(v, list)}
    except Exception:  # noqa: BLE001 — broken file degrades to no aliases
        return {}


def _build_wilaya_chunks() -> list[dict]:
    """One chunk per wilaya: 'Wilaya Alger (location_id=16). Also called: ...'

    The SLM is told to write `WHERE location_id = N` using the integer it sees
    in this chunk. That removes the whole name-spelling fragility ('Algiers'
    vs 'Alger') from the SQL path.
    """
    chunks: list[dict] = []
    aliases = _load_wilaya_aliases()
    try:
        conn = db_connect()
        cur = conn.cursor()
        cur.execute("SELECT location_id, wilaya FROM dim_location "
                    "WHERE wilaya IS NOT NULL ORDER BY location_id")
        rows = cur.fetchall()
        conn.close()
    except Exception:  # noqa: BLE001 — db unavailable degrades to empty
        return chunks
    for loc_id, wilaya in rows:
        alts = aliases.get(wilaya, [])
        alt_str = (f" Also known as: {', '.join(alts)}." if alts else "")
        chunks.append({
            "kind": "wilaya",
            "wilaya": wilaya,
            "location_id": int(loc_id),
            "text": (f"Wilaya '{wilaya}' has location_id={int(loc_id)}.{alt_str} "
                     f"To filter SQL by this wilaya use "
                     f"`location_id = {int(loc_id)}` — not the wilaya name."),
        })
    return chunks


def build_chunks() -> list[dict]:
    """Flatten the four knowledge sources into retrievable text chunks."""
    chunks: list[dict] = []

    # data_catalog — the database documenting its own columns
    try:
        conn = db_connect()
        cur = conn.cursor()
        cur.execute("SELECT table_name, column_name, description "
                    "FROM data_catalog")
        for row in cur.fetchall():
            chunks.append({
                "kind": "column", "table": row[0], "column": row[1],
                "text": f"Column {row[0]}.{row[1]}: {row[2]}",
            })
        conn.close()
    except Exception:  # noqa: BLE001 — data_catalog optional
        pass

    # wilaya identities — one chunk per wilaya with location_id
    chunks.extend(_build_wilaya_chunks())

    # kpi_catalog.json — multilingual synonyms
    if os.path.isfile(V6Config.KPI_CATALOG_PATH):
        for e in _load_json(V6Config.KPI_CATALOG_PATH):
            syn = ", ".join(e.get("synonyms", []))
            chunks.append({
                "kind": "kpi", "table": e["table"], "column": e["column"],
                "text": (f"KPI '{e['column']}' in table {e['table']} "
                         f"(segment {e.get('segment', '-')}, unit "
                         f"{e.get('unit', '-')}). "
                         f"{e.get('description', '').rstrip('.')}. "
                         f"Also called: {syn}."),
            })

    # glossary.json — definitions, business context, relationships
    if os.path.isfile(V6Config.GLOSSARY_PATH):
        g = _load_json(V6Config.GLOSSARY_PATH)
        sysd = g.get("system", {})
        if sysd:
            chunks.append({
                "kind": "system",
                "text": f"{sysd.get('name', '')}: {sysd.get('description', '')}"})
        for item in g.get("business_context", []):
            chunks.append({
                "kind": "context",
                "text": f"[{item.get('topic', '')}] {item.get('text', '')}"})
        for item in g.get("definitions", []):
            chunks.append({
                "kind": "definition", "term": item.get("term", ""),
                "text": (f"Definition of {item.get('term', '')}: "
                         f"{item.get('text', '')}")})
        for rel in g.get("table_relationships", []):
            chunks.append({"kind": "relationship",
                           "text": f"Table relationship: {rel}"})
    return chunks


# ── retriever ────────────────────────────────────────────────────────────
class Retriever:
    """Cosine top-k retrieval over the knowledge chunks."""

    def __init__(self):
        self.chunks = build_chunks()
        self.encoder = get_encoder()
        texts = [c["text"] for c in self.chunks]
        self.embeddings = (
            self.encoder.encode_batch(texts) if texts
            else torch.empty(0, V6Config.EMBED_DIM))

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        k = k or V6Config.RAG_TOP_K
        if not self.chunks:
            return []
        q = self.encoder.encode(query)                  # [1024], normalized
        sims = self.embeddings @ q                       # cosine similarity
        topk = torch.topk(sims, min(k, len(self.chunks)))
        hits: list[dict] = []
        for score, idx in zip(topk.values.tolist(), topk.indices.tolist()):
            c = dict(self.chunks[idx])
            c["score"] = round(float(score), 4)
            hits.append(c)
        return hits

    def knowledge_block(self, query: str, k: int | None = None) -> tuple[str, float]:
        """Return (formatted knowledge text, top cosine score)."""
        hits = self.retrieve(query, k)
        if not hits:
            return "(no reference knowledge available)", 0.0
        return "\n".join(f"- {h['text']}" for h in hits), hits[0]["score"]

    def definition_for(self, query: str) -> str | None:
        """Best definition/KPI chunk for a 'what does X mean' question."""
        for h in self.retrieve(query, k=5):
            if h.get("kind") in ("definition", "kpi"):
                return h["text"]
        return None


_retriever: Retriever | None = None


def get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever
