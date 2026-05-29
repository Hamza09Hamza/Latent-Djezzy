"""v6/slm.py — Qwen2.5-Coder dual-role engine (LatentMAS core).

One model instance plays both agents:

    Phase 1 (Router)        : query + schema + knowledge → routing JSON.
    Phase 2 (SQL generator) : continues the SAME conversation, reusing the
                              phase-1 KV cache → SQL.

The KV-cache hand-off is the "latent communication": phase 2 does not
re-encode phase 1's context, it inherits phase 1's attention state. Because
LangGraph state cannot carry torch tensors, the cache lives in an in-process
store keyed by `thread_id`; the graph state carries only the id. If the
store misses (e.g. a checkpoint resumed in a fresh process) phase 2 falls
back to a plain re-encode — slower, identical output.

Colab knobs (env, see config.py): V6_4BIT, V6_FLASH_ATTN, V6_SLM_OVERRIDE.
"""

from __future__ import annotations
import threading
import time

import torch

from .config import V6Config

# ── constrained SQL decoding helpers ────────────────────────────────────────

_SQL_REGEX = (
    r"(?s)"
    r"SELECT\s+.+?"
    r"\s+FROM\s+\w[\w.]*(?:\s+(?:AS\s+)?\w+)?"
    r"(?:\s+(?:LEFT\s+|RIGHT\s+|INNER\s+|OUTER\s+)?"
    r"JOIN\s+\w[\w.]*(?:\s+(?:AS\s+)?\w+)?"
    r"(?:\s+ON\s+.+?))*"
    r"(?:\s+WHERE\s+.+?)?"
    r"(?:\s+GROUP\s+BY\s+.+?)?"
    r"(?:\s+HAVING\s+.+?)?"
    r"(?:\s+ORDER\s+BY\s+.+?)?"
    r"(?:\s+LIMIT\s+\d+)?"
    r"\s*;?"
)


def _build_sql_logits_processor(tokenizer):
    """Return a lm-format-enforcer LogitsProcessor for SQL, or None if the
    library is not installed. Failures are silent so unconstrained fallback
    kicks in automatically."""
    try:
        from lmformatenforcer import RegexParser
        from lmformatenforcer.integrations.transformers import (
            build_transformers_prefix_allowed_tokens_fn,
        )

        parser = RegexParser(_SQL_REGEX)
        prefix_fn = build_transformers_prefix_allowed_tokens_fn(tokenizer, parser)

        class _RegexLogitsProcessor:
            def __call__(self, input_ids, scores):
                import torch
                allowed = prefix_fn(0, input_ids[0])
                mask = torch.full_like(scores, fill_value=float("-inf"))
                if allowed:
                    mask[0, list(allowed)] = 0.0
                else:
                    mask[0, tokenizer.eos_token_id] = 0.0
                return scores + mask

        return _RegexLogitsProcessor()
    except Exception:  # noqa: BLE001
        return None


def _strip_to_sql(text: str) -> str:
    """Find the SELECT…; block — safety net for any stray leading whitespace."""
    import re
    m = re.search(r"(SELECT\b.*)", text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip() if m else text.strip()


class DualRoleSLM:
    def __init__(self, model_id: str | None = None,
                 device: torch.device | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.device = device or V6Config.device()
        model_id = model_id or V6Config.slm_id()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        load_kwargs: dict = {}
        if V6Config.USE_4BIT:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = (
                torch.float16 if self.device.type == "cuda" else torch.float32)

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
        if not V6Config.USE_4BIT:
            self.model.to(self.device)
        self.model.eval()

        # ── Speculative decoding: load the 0.5B drafter if main model is bigger
        # The drafter generates γ candidate tokens per step; the main model
        # validates them all in one forward pass — free speed from the KV cache.
        self._draft: AutoModelForCausalLM | None = None
        self._draft_tokenizer: AutoTokenizer | None = None
        draft_id = V6Config.draft_slm_id(model_id) if V6Config.USE_SPECULATIVE else None
        if draft_id and draft_id != model_id:
            try:
                draft_kwargs = {"torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32}
                self._draft = AutoModelForCausalLM.from_pretrained(draft_id, **draft_kwargs)
                self._draft.to(self.device)
                self._draft.eval()
                # Universal assisted decoding: only pass tokenizer args when the
                # two models have *different* vocabularies (different vocab_size).
                # Passing assistant_tokenizer when they share a vocab raises ValueError.
                draft_tok = AutoTokenizer.from_pretrained(draft_id)
                main_vocab  = self.tokenizer.vocab_size
                draft_vocab = draft_tok.vocab_size
                self._draft_tokenizer = draft_tok if draft_vocab != main_vocab else None
            except Exception:
                self._draft = None
                self._draft_tokenizer = None

        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self._im_end = (im_end if isinstance(im_end, int) and im_end >= 0
                        else self.tokenizer.eos_token_id)

        # thread_id → phase-1 result (sequences + KV cache) for the hand-off
        self._store: dict[str, dict] = {}
        self.model_id = model_id
        # Qwen3 models have thinking mode on by default — disable it everywhere
        # by passing enable_thinking=False to apply_chat_template.
        self._qwen3 = "qwen3" in model_id.lower()

    # ── chat-template helper ─────────────────────────────────────────────
    def _tmpl(self, messages: list[dict], **kw) -> str:
        """Apply the chat template. For Qwen3 disables thinking mode so the
        model outputs directly instead of reasoning aloud first."""
        if self._qwen3:
            kw.setdefault("enable_thinking", False)
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kw)

    # ── single-turn chat (used by the direct-answer node) ────────────────
    @torch.no_grad()
    def chat(self, messages: list[dict], max_new_tokens: int = 256) -> str:
        text = self._tmpl(messages)
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        gen_kw = dict(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                      pad_token_id=self.tokenizer.eos_token_id)
        if self._draft is not None:
            gen_kw["assistant_model"] = self._draft
            if self._draft_tokenizer is not None:   # different vocab → need both
                gen_kw["tokenizer"] = self.tokenizer
                gen_kw["assistant_tokenizer"] = self._draft_tokenizer
        out = self.model.generate(**gen_kw)
        return self.tokenizer.decode(
            out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()

    # ── Phase 1: router ──────────────────────────────────────────────────
    @torch.no_grad()
    def run_router(self, messages: list[dict], thread_id: str = "default",
                   max_new: int | None = None) -> dict:
        """Run the router and stash its KV cache for the phase-2 hand-off."""
        max_new = max_new or V6Config.ROUTER_MAX_NEW_TOKENS
        t0 = time.time()
        text = self._tmpl(messages)
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        gen_kw: dict = dict(
            **enc, max_new_tokens=max_new, do_sample=False,
            return_dict_in_generate=True, use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id)
        if self._draft is not None:
            gen_kw["assistant_model"] = self._draft  # speculative decoding
            if self._draft_tokenizer is not None:   # different vocab → need both
                gen_kw["tokenizer"] = self.tokenizer
                gen_kw["assistant_tokenizer"] = self._draft_tokenizer
        out1 = self.model.generate(**gen_kw)
        router_out = self.tokenizer.decode(
            out1.sequences[0, enc.input_ids.shape[1]:],
            skip_special_tokens=True).strip()

        self._store[thread_id] = {
            "router_output": router_out,
            "_seq1": out1.sequences,
            "_cache": getattr(out1, "past_key_values", None),
            "_messages": messages,
        }
        return {"router_output": router_out,
                "router_ms": (time.time() - t0) * 1000.0}

    # ── Phase 2: SQL generator ───────────────────────────────────────────
    @torch.no_grad()
    def run_sqlgen(self, thread_id: str = "default", instruction: str = "",
                   max_new: int | None = None, kv_reuse: bool = True) -> dict:
        """Generate SQL, reusing the router's KV cache when available.

        When V6Config.USE_CONSTRAINED_SQL is True the generation is grammar-
        constrained via lm-format-enforcer: every token step is masked so the
        model can only emit valid SQL tokens — no preamble, no comments, no
        trailing explanation. max_new drops to SQLGEN_CONSTRAINED_MAX_NEW_TOKENS
        (~150) because there is no wasted prose to budget for. If the library
        is unavailable or the processor fails, falls back to unconstrained.
        """
        constrained = V6Config.USE_CONSTRAINED_SQL
        if max_new is None:
            max_new = (V6Config.SQLGEN_CONSTRAINED_MAX_NEW_TOKENS
                       if constrained else V6Config.SQLGEN_MAX_NEW_TOKENS)

        rr = self._store.get(thread_id)
        if rr is None:
            return {"sql_output": "", "kv_reused": False, "sqlgen_ms": 0.0,
                    "error": "no router state for thread"}

        logits_processor = None
        if constrained:
            logits_processor = _build_sql_logits_processor(self.tokenizer)
            if logits_processor is None:
                constrained = False  # library missing — silent fallback

        t0 = time.time()
        sql_out, kv_used = None, False
        if kv_reuse and rr.get("_cache") is not None:
            try:
                sql_out = self._sqlgen_kv(
                    rr["_seq1"], rr["_cache"], instruction, max_new,
                    logits_processor=logits_processor)
                kv_used = True
            except Exception:  # noqa: BLE001 — fall back to re-encode
                sql_out = None
        if sql_out is None:
            sql_out = self._sqlgen_plain(
                rr["_messages"], rr["router_output"], instruction, max_new,
                logits_processor=logits_processor)
        return {"sql_output": sql_out, "kv_reused": kv_used,
                "constrained": constrained,
                "sqlgen_ms": (time.time() - t0) * 1000.0}

    def clear_thread(self, thread_id: str) -> None:
        """Drop a thread's stored KV cache to free GPU memory."""
        entry = self._store.pop(thread_id, None)
        if entry is not None:
            entry.clear()  # release past_key_values tensor references
        torch.cuda.empty_cache()

    # ── streaming generation ─────────────────────────────────────────────
    def stream_generate(self, messages: list[dict], max_new_tokens: int = 512):
        """Yield decoded tokens one by one as the model generates them."""
        from transformers import TextIteratorStreamer
        text = self._tmpl(messages)
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            streamer=streamer,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        t = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        t.start()
        for token in streamer:
            yield token
        t.join()

    # ── phase-2 implementations ──────────────────────────────────────────
    def _sqlgen_kv(self, seq1, cache, instruction: str, max_new: int,
                   logits_processor=None) -> str:
        if not hasattr(cache, "get_seq_length"):
            from transformers import DynamicCache
            cache = DynamicCache.from_legacy_cache(cache)
        if cache.get_seq_length() == 0:
            raise RuntimeError("empty KV cache")

        seq = seq1.to(self.device)
        if seq[0, -1].item() != self._im_end:
            end = torch.tensor([[self._im_end]], device=self.device)
            seq = torch.cat([seq, end], dim=1)

        followup = (f"\n<|im_start|>user\n{instruction}<|im_end|>\n"
                    f"<|im_start|>assistant\n")
        fu_ids = self.tokenizer(
            followup, return_tensors="pt",
            add_special_tokens=False).input_ids.to(self.device)

        full_ids = torch.cat([seq, fu_ids], dim=1)
        attn = torch.ones((1, full_ids.shape[1]), dtype=torch.long,
                          device=self.device)
        gen_kw: dict = dict(
            input_ids=full_ids, attention_mask=attn, past_key_values=cache,
            max_new_tokens=max_new, do_sample=False,
            return_dict_in_generate=True, use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id)
        if logits_processor is not None:
            from transformers import LogitsProcessorList
            gen_kw["logits_processor"] = LogitsProcessorList([logits_processor])
        out2 = self.model.generate(**gen_kw)
        raw = self.tokenizer.decode(
            out2.sequences[0, full_ids.shape[1]:],
            skip_special_tokens=True).strip()
        return _strip_to_sql(raw)

    def _sqlgen_plain(self, router_messages, router_out, instruction,
                      max_new, logits_processor=None) -> str:
        messages = list(router_messages) + [
            {"role": "assistant", "content": router_out},
            {"role": "user", "content": instruction},
        ]
        text = self._tmpl(messages)
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        gen_kw: dict = dict(
            **enc, max_new_tokens=max_new, do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id)
        if logits_processor is not None:
            from transformers import LogitsProcessorList
            gen_kw["logits_processor"] = LogitsProcessorList([logits_processor])
        out = self.model.generate(**gen_kw)
        raw = self.tokenizer.decode(
            out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
        return _strip_to_sql(raw)


_slm: DualRoleSLM | None = None


def get_slm() -> DualRoleSLM:
    global _slm
    if _slm is None:
        _slm = DualRoleSLM()
    return _slm


def detect_lang(text: str) -> str:
    """Detect query language: Arabic script → Darija, French vocab → French, else English."""
    if any('؀' <= c <= 'ۿ' for c in text):
        return "Algerian Darija (Arabic)"
    french = {
        'est', 'que', 'qui', 'les', 'des', 'pour', 'dans', 'avec', 'sur', 'par',
        'du', 'la', 'le', 'un', 'une', 'comment', 'quoi', 'quel', 'quelle',
        'montre', 'affiche', 'peux', 'faire', 'pouvez', 'montrez', 'donne',
        'combien', 'quels', 'moi', 'mois', 'année', 'quel', 'trimestre',
    }
    words = set(text.lower().replace("'", " ").replace("?", " ").split())
    if len(words & french) >= 2 or 'salam' in text.lower():
        return "French"
    return "English"


# ── Analyst: turns raw SQL rows into an analytical paragraph ─────────────────
_ANALYST_SYSTEM = """You are a senior telecom data analyst writing the
final answer for a business user. Your job is to turn the raw data block
(rows from a SQL query, with column names and numeric values) into a
short, insightful paragraph that the user can read at a glance.

ABSOLUTE RULES — break one and the answer is wrong:
1. NEVER invent a number, a wilaya, a date, a KPI, or any fact not in
   the raw data block. If a value is missing, say so plainly.
2. SPOKEN NUMBERS — this answer is READ ALOUD, so round to a clean,
   human-speakable figure and ALWAYS keep the scale word attached.
   A large raw number must become "<rounded> <scale> <unit>", never a
   bare truncated decimal.
     1,000,000,000+  → "<X.X> billion DZD"   /  "<X,X> milliards DA"
     1,000,000+      → "<X.X> million DZD"    /  "<X,X> millions DA"
     1,000+          → "<X.X> thousand DZD"   /  "<X,X> milliers DA"
                        (or round to the nearest thousand)
     rates / %       → keep 1–2 decimals, e.g. "42.39%" / "42,39 %"
   Examples of the SAME value done right vs wrong:
     raw 24,546,455 DZD → GOOD "24.5 million DZD" / "24,5 millions DA"
                          BAD  "24,5"  ·  BAD "24,546,455"  ·  BAD "24.5"
   Never output a number like "24,5" or "24.5" on its own — the listener
   cannot tell millions from a fraction. The magnitude word is mandatory.
3. Keep the units the data uses (DZD, %, etc.). Never add units not in
   the data unless the column name explicitly states them.
4. Stay grounded in the question — lead with the direct answer.

STYLE — write like a smart colleague briefing the user:
- Direct: lead with the insight, then add supporting numbers.
- Warm but professional: no "Based on the data", no "I have analyzed".
- Reply in the SAME LANGUAGE the user used (French → French, Arabic → Arabic).
- Max 3 short sentences or a tight bullet list for multi-wilaya comparisons.
- Never mention SQL, "the query", or "the data block".

EXAMPLES:
Raw: "1 row | avg_gross_margin: 40.12"
Q: "Show me the gross margin for Batna last quarter"
GOOD: "Batna's average gross margin last quarter was 40.12%."
BAD : "The avg_gross_margin value is 40.12."

Raw: "1 row | net_income: 24546455.0"
Q: "Quel est le revenu net pour Oran ?"
GOOD: "Le revenu net d'Oran est d'environ 24,5 millions de DA."
BAD : "Le revenu net est de 24,5."   (no scale word — unspeakable)
BAD : "Le revenu net est de 24 546 455 DA."   (too long to read aloud)

Raw: "2 rows | Sétif net_income 491,926,011 | Tlemcen net_income 432,742,268"
Q: "Compare net income for Tlemcen and Setif"
GOOD: "Sétif leads with about 491.9 million DZD; Tlemcen follows at roughly 432.7 million — a gap of around 59 million."
BAD : "Row 1 shows Sétif: 491926011, Row 2 shows Tlemcen: 432742268."

Now write the answer."""


# ── Polisher: rewrites RAG / definition text into natural prose ───────────────
_POLISHER_SYSTEM = """You are a telecom knowledge assistant. Rewrite the
reference text below into a clear, natural response in the SAME LANGUAGE
as the user's question.

RULES:
1. Stay faithful to the source — never invent facts.
2. Write in the same language the user used (French → French, Arabic → Arabic, English → English).
3. 2–4 natural sentences. No bullet lists unless comparing multiple items.
4. Drop jargon where simpler phrasing works. Sound like a knowledgeable colleague.
5. Never start with "Based on the text", "The definition says", or "According to".
   Just explain directly.
6. SCOPE: if the question asks for code, translation, general knowledge, or
   anything unrelated to telecom analytics, reply only:
   "I'm a telecom analytics assistant — I can only help with KPIs, data
   queries, or definitions related to the database."

Write the rewritten response now."""


# ── Clarifier: explains an error or missing info in natural language ──────────
_CLARIFIER_SYSTEM = """You are a helpful telecom analytics assistant.
Something went wrong or is missing. Explain the issue naturally — without
technical jargon — and ask for what you need to continue.

RULES:
1. Write in the SAME LANGUAGE the user used (French → French, Arabic → Arabic, English → English).
2. One sentence: say what couldn't be done, plainly (no SQL jargon, no error codes).
3. One question: ask exactly what extra information you need, or suggest a rephrasing.
4. NEVER say "no such column", "SQL error", "query failed", "rows returned".
   Say "I couldn't find the data" or "I need more details about what you want".
5. Max 3 short sentences total. Warm, professional tone.

Example — no recipient found:
GOOD: "I've prepared the email but I don't know who to send it to — could you name a recipient?"
BAD:  "Email drafted but no recipient was named — pick one from the contacts list."

Write your clarification now."""


class Polisher:
    """Small natural-language refiner. Default Qwen2.5-1.5B-Instruct: small
    enough to stay cheap on T4, fluent enough for natural prose. Loaded
    lazily on first use."""

    def __init__(self, hub_id: str | None = None):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.hub_id = hub_id or V6Config.POLISHER_HUB_ID
        self.device = V6Config.device()
        dtype = (torch.float16 if self.device.type in ("cuda", "mps")
                 else torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hub_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hub_id, torch_dtype=dtype,
            device_map={"": str(self.device)})
        self.model.eval()

    def stream(self, raw_answer: str, question: str = "",
               role: str = "analyze"):
        """Yield polished tokens one-by-one via TextIteratorStreamer.

        role: 'analyze' (SQL data → analytical prose) |
              'polish'  (RAG/definition → natural rewrite) |
              'clarify' (error / missing info → helpful clarification)
        The model infers the response language from the user's question text.
        """
        from transformers import TextIteratorStreamer

        _systems = {
            "analyze": _ANALYST_SYSTEM,
            "polish":  _POLISHER_SYSTEM,
            "clarify": _CLARIFIER_SYSTEM,
        }
        system = _systems.get(role, _ANALYST_SYSTEM)

        if role == "clarify":
            user_msg = (f"User's original question: {question}\n\n"
                        f"Issue to clarify: {raw_answer}")
        else:
            user_msg = (f"User question: {question}\n\nRaw data block:\n{raw_answer}"
                        if question else f"Raw data block:\n{raw_answer}")

        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user_msg}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        exc_holder: list[BaseException] = []

        def _gen():
            try:
                self.model.generate(
                    **enc,
                    streamer=streamer,
                    max_new_tokens=V6Config.POLISHER_MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.5,
                    top_p=0.9,
                    repetition_penalty=1.05,
                    pad_token_id=self.tokenizer.eos_token_id)
            except Exception as e:  # noqa: BLE001
                exc_holder.append(e)
                streamer.end()  # unblock the iterator

        t = threading.Thread(target=_gen, daemon=True)
        t.start()
        for token in streamer:
            yield token
        t.join()
        if exc_holder:
            raise exc_holder[0]

    @torch.no_grad()
    def complete(self, system: str, user: str, max_new: int = 120) -> str:
        """Blocking single-shot completion. Used for tasks like recipient
        resolution where streaming offers no benefit."""
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **enc, max_new_tokens=max_new, do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(
            out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()


_polisher: Polisher | None = None


def get_polisher() -> Polisher:
    global _polisher
    if _polisher is None:
        _polisher = Polisher()
    return _polisher
