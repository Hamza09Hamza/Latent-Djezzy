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
        draft_id = V6Config.draft_slm_id(model_id) if V6Config.USE_SPECULATIVE else None
        if draft_id and draft_id != model_id:
            try:
                draft_kwargs = {"torch_dtype": torch.float16 if self.device.type == "cuda" else torch.float32}
                self._draft = AutoModelForCausalLM.from_pretrained(draft_id, **draft_kwargs)
                self._draft.to(self.device)
                self._draft.eval()
            except Exception:
                self._draft = None

        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self._im_end = (im_end if isinstance(im_end, int) and im_end >= 0
                        else self.tokenizer.eos_token_id)

        # thread_id → phase-1 result (sequences + KV cache) for the hand-off
        self._store: dict[str, dict] = {}
        self.model_id = model_id

    # ── single-turn chat (used by the direct-answer node) ────────────────
    @torch.no_grad()
    def chat(self, messages: list[dict], max_new_tokens: int = 256) -> str:
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        gen_kw = dict(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                      pad_token_id=self.tokenizer.eos_token_id)
        if self._draft is not None:
            gen_kw["assistant_model"] = self._draft
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
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        gen_kw: dict = dict(
            **enc, max_new_tokens=max_new, do_sample=False,
            return_dict_in_generate=True, use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id)
        if self._draft is not None:
            gen_kw["assistant_model"] = self._draft  # speculative decoding
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
        """Generate SQL, reusing the router's KV cache when available."""
        max_new = max_new or V6Config.SQLGEN_MAX_NEW_TOKENS
        rr = self._store.get(thread_id)
        if rr is None:
            return {"sql_output": "", "kv_reused": False, "sqlgen_ms": 0.0,
                    "error": "no router state for thread"}

        t0 = time.time()
        sql_out, kv_used = None, False
        if kv_reuse and rr.get("_cache") is not None:
            try:
                sql_out = self._sqlgen_kv(
                    rr["_seq1"], rr["_cache"], instruction, max_new)
                kv_used = True
            except Exception:  # noqa: BLE001 — fall back to re-encode
                sql_out = None
        if sql_out is None:
            sql_out = self._sqlgen_plain(
                rr["_messages"], rr["router_output"], instruction, max_new)
        return {"sql_output": sql_out, "kv_reused": kv_used,
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
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
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
    def _sqlgen_kv(self, seq1, cache, instruction: str, max_new: int) -> str:
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
        out2 = self.model.generate(
            input_ids=full_ids, attention_mask=attn, past_key_values=cache,
            max_new_tokens=max_new, do_sample=False,
            return_dict_in_generate=True, use_cache=True,
            pad_token_id=self.tokenizer.eos_token_id)
        return self.tokenizer.decode(
            out2.sequences[0, full_ids.shape[1]:],
            skip_special_tokens=True).strip()

    def _sqlgen_plain(self, router_messages, router_out, instruction,
                      max_new) -> str:
        messages = list(router_messages) + [
            {"role": "assistant", "content": router_out},
            {"role": "user", "content": instruction},
        ]
        return self.chat(messages, max_new_tokens=max_new)


_slm: DualRoleSLM | None = None


def get_slm() -> DualRoleSLM:
    global _slm
    if _slm is None:
        _slm = DualRoleSLM()
    return _slm


class Polisher:
    """0.5B model for streaming response polish. Loaded lazily on first use."""

    HUB_ID = "Qwen/Qwen2.5-0.5B-Instruct"

    def __init__(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.device = V6Config.device()
        dtype = (torch.float16 if self.device.type in ("cuda", "mps")
                 else torch.float32)
        self.tokenizer = AutoTokenizer.from_pretrained(self.HUB_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.HUB_ID, torch_dtype=dtype,
            device_map={"": str(self.device)})
        self.model.eval()

    def stream(self, raw_answer: str, question: str = ""):
        """Yield polished tokens one-by-one via TextIteratorStreamer."""
        from transformers import TextIteratorStreamer

        system = (
            "You are a professional telecom analytics assistant. "
            "Rewrite the answer in clear, natural language. "
            "Keep ALL numbers and KPI values exactly as given. "
            "Be concise — maximum 3 sentences.")
        user_msg = (f"Q: {question}\nA: {raw_answer}" if question
                    else raw_answer)

        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": user_msg}]
        inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt",
            add_generation_prompt=True).to(self.device)

        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        t = threading.Thread(target=self.model.generate, kwargs={
            "input_ids": inputs,
            "streamer": streamer,
            "max_new_tokens": 120,
            "do_sample": True,
            "temperature": 0.4,
            "pad_token_id": self.tokenizer.eos_token_id,
        })
        t.start()
        for token in streamer:
            yield token
        t.join()


_polisher: Polisher | None = None


def get_polisher() -> Polisher:
    global _polisher
    if _polisher is None:
        _polisher = Polisher()
    return _polisher
