"""v6/config.py — Configuration for the agentic pipeline.

Every tunable is read from the environment with a `V6_` prefix and falls
back to the legacy `V5_` prefix, so an existing Colab runtime keeps working.
Paths resolve relative to the repo so the system runs with no network.
"""

from __future__ import annotations
import os

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)


def _env(name: str, default: str = "") -> str:
    """Read V6_<name>, then V5_<name>, then the default."""
    return os.environ.get(f"V6_{name}", os.environ.get(f"V5_{name}", default))


class V6Config:
    # ── Paths ────────────────────────────────────────────────────────────
    BASE_DIR     = _HERE
    REPO_ROOT    = _REPO_ROOT
    DATA_DIR     = os.path.join(_HERE, "data")
    TEMPLATE_DIR = os.path.join(_HERE, "templates")
    MODELS_DIR   = os.path.join(_REPO_ROOT, "models")

    GLOSSARY_PATH        = os.path.join(DATA_DIR, "glossary.json")
    KPI_CATALOG_PATH     = os.path.join(DATA_DIR, "kpi_catalog.json")
    WILAYA_ALIASES_PATH  = os.path.join(DATA_DIR, "wilaya_aliases.json")

    # Charts and rendered reports land here (created on demand).
    # Kept as class attr for CLI banner; use output_dir() for actual paths.
    OUTPUT_DIR = _env("OUTPUT_DIR") or (
        "/content/v6_output" if os.path.isdir("/content")
        else os.path.join(_REPO_ROOT, "v6_output"))

    # ── Database backend ─────────────────────────────────────────────────
    # SQLite on Colab (MySQL is unreachable there); MySQL locally.
    USE_SQLITE  = _env("USE_SQLITE", "0") == "1"
    _SQLITE_ENV = _env("SQLITE_PATH")

    MYSQL_HOST     = os.environ.get("LATENTMIND_MYSQL_HOST", "localhost")
    MYSQL_PORT     = int(os.environ.get("LATENTMIND_MYSQL_PORT", "3306"))
    MYSQL_USER     = os.environ.get("LATENTMIND_MYSQL_USER", "root")
    MYSQL_PASSWORD = os.environ.get("LATENTMIND_MYSQL_PASSWORD", "2003Hamza2003!")
    MYSQL_DB       = os.environ.get("LATENTMIND_MYSQL_DB", "interndb")

    # ── SLM — one model, two roles ───────────────────────────────────────
    # Router and SQL generator share weights; that is what makes the
    # KV-cache hand-off (latent communication) valid.
    #
    # Size toggle (V6_SLM_SIZE):
    #   "3b"  → Qwen2.5-Coder-3B-Instruct       (~6.4 GB fp16) — safe baseline
    #   "4b"  → Qwen3-4B-Instruct-2507          (~8.0 GB fp16) — current default
    #            text-only, native Instruct, no thinking mode, 262K ctx, Apache-2.0
    #            drafter = Qwen3-0.6B (same tokenizer family)
    #   "7b"  → Qwen2.5-Coder-7B-Instruct       (~14 GB fp16)  — needs V6_4BIT=1
    #
    # NOT used: Qwen3.5-4B is the multimodal (vision+text) branch with a new
    # `qwen3_5` model_type that requires transformers from git main, and ships
    # without an Instruct head — wrong fit for a text-only SQL router.
    SLM_SIZE = _env("SLM_SIZE", "4b").lower()
    SLM_CANDIDATES_3B = [
        "qwen2.5-coder-3b-instruct",
        "qwen2.5-coder-1.5b-instruct",
        "qwen2.5-coder-0.5b-instruct",
    ]
    SLM_CANDIDATES_4B = [
        "qwen3-4b-instruct-2507",        # primary: Qwen3-4B-Instruct-2507 (text-only, Instruct)
        "qwen3-4b",                      # fallback: base Qwen3-4B if Instruct-2507 not cached
        "qwen2.5-coder-3b-instruct",
    ]
    SLM_CANDIDATES_7B = [
        "qwen2.5-coder-7b-instruct",
        "qwen2.5-coder-3b-instruct",
    ]
    SLM_OVERRIDE   = _env("SLM_OVERRIDE")          # force a Hub model id
    SLM_HUB_ID_3B  = "Qwen/Qwen2.5-Coder-3B-Instruct"
    SLM_HUB_ID_4B  = "Qwen/Qwen3-4B-Instruct-2507"
    SLM_HUB_ID_7B  = "Qwen/Qwen2.5-Coder-7B-Instruct"
    USE_4BIT         = _env("4BIT", "0") == "1"      # 4-bit NF4 quantization
    USE_SPECULATIVE  = _env("SPECULATIVE", "1") == "1"  # drafter → 2-4x speed

    ROUTER_MAX_NEW_TOKENS = 128   # routing JSON rarely exceeds 80 tokens
    SQLGEN_MAX_NEW_TOKENS = 256   # unconstrained fallback ceiling
    SQLGEN_CONSTRAINED_MAX_NEW_TOKENS = 150  # constrained: pure SQL fits in ~110-130 tokens

    # ── Polisher — small natural-language refiner ────────────────────────
    # Qwen2.5-1.5B-Instruct: small (~1.5 GB fp16) but produces noticeably
    # more natural prose than the 0.5B and stays cheap on T4. The polisher
    # never invents numbers — it rewords the raw data summary.
    POLISHER_HUB_ID = _env("POLISHER_HUB_ID", "Qwen/Qwen2.5-1.5B-Instruct")
    POLISHER_MAX_NEW_TOKENS = int(_env("POLISHER_MAX_NEW", "180"))

    # ── Constrained SQL decoding (lm-format-enforcer) ────────────────────
    # When ON: a grammar processor masks non-SQL tokens at every generation
    # step — no preamble, no comments, no explanation, output is always
    # valid SQL structure, ~40% fewer tokens generated.
    # Toggle: V6_CONSTRAINED_SQL=1 (on) / 0 (off, default)
    # Requires: pip install lm-format-enforcer>=0.10.3
    USE_CONSTRAINED_SQL = _env("CONSTRAINED_SQL", "0") == "1"

    # ── RAG encoder — frozen BGE-M3 (1024-d) ─────────────────────────────
    BGE_M3_LOCAL_DIR = os.path.join(MODELS_DIR, "bge-m3")
    BGE_M3_HUB_ID    = "BAAI/bge-m3"
    EMBED_DIM        = 1024
    RAG_TOP_K        = 5
    RAG_LOW_CONF     = 0.45        # top cosine below this → weak grounding

    # ── Brain — the policy loop that decides the next action ─────────────
    # A trained 3-head MLP, called once per loop step. It reads the query,
    # the conversation memory and the outcome of the last action, then
    # predicts the intent, the next action, and a "continue" score — the
    # seuil. Below the seuil the loop ends at the communicator. Build it:
    #   python3 -m v6.brain_data    # synthesize agentic traces
    #   python3 -m v6.train_brain   # train models/brain_head.pt
    BRAIN_HEAD_PATH    = os.path.join(MODELS_DIR, "brain_head.pt")
    BRAIN_TRAIN_PATH   = os.path.join(DATA_DIR, "brain_train.jsonl")
    BRAIN_PROTOTYPES   = os.path.join(DATA_DIR, "planner_prototypes.json")
    BRAIN_SEUIL        = float(_env("BRAIN_SEUIL", "0.5"))      # continue ≥ this → keep going
    BRAIN_CONF_MIN     = float(_env("BRAIN_CONF_MIN", "0.35"))  # action conf below → communicator
    BRAIN_MAX_STEPS    = int(_env("BRAIN_MAX_STEPS", "8"))      # loop safety cap

    # ── Speech I/O — STT (front) + TTS (back) ────────────────────────────
    # The voice layer wraps the text pipeline: an audio question is
    # transcribed by faster-whisper, the normal graph runs, and the polished
    # answer is spoken sentence-by-sentence by XTTS-v2 as it streams.
    #
    # STT — faster-whisper (CTranslate2). large-v3 is the strongest FR/EN/AR
    # model; compute_type float16 on GPU, int8_float16 to save ~1.5 GB VRAM.
    STT_ENABLED    = _env("STT_ENABLED", "1") == "1"
    STT_MODEL      = _env("STT_MODEL", "large-v3")
    STT_COMPUTE    = _env("STT_COMPUTE", "float16")   # int8_float16 to save VRAM
    STT_BEAM_SIZE  = int(_env("STT_BEAM_SIZE", "5"))
    STT_LANGUAGE   = _env("STT_LANGUAGE", "")          # "" = auto; or "fr"/"en"

    # TTS — Coqui XTTS-v2 (multilingual, streaming, clean female studio
    # voices). A built-in speaker name is used by default; set a reference
    # WAV to clone a specific voice instead. Both default voices are female.
    TTS_ENABLED    = _env("TTS_ENABLED", "1") == "1"
    TTS_MODEL      = _env("TTS_MODEL",
                          "tts_models/multilingual/multi-dataset/xtts_v2")
    TTS_SPEAKER_FR = _env("TTS_SPEAKER_FR", "Daisy Studious")   # clean female
    TTS_SPEAKER_EN = _env("TTS_SPEAKER_EN", "Daisy Studious")   # clean female
    TTS_SPEAKER_WAV_FR = _env("TTS_SPEAKER_WAV_FR", "")  # overrides name if set
    TTS_SPEAKER_WAV_EN = _env("TTS_SPEAKER_WAV_EN", "")
    TTS_SAMPLE_RATE    = 24000                          # XTTS-v2 native rate
    TTS_SPEED          = float(_env("TTS_SPEED", "1.0"))

    # Benchmark fixtures
    BENCH_QUERIES_PATH = os.path.join(DATA_DIR, "bench_queries.json")

    # ── SQL runner safety ────────────────────────────────────────────────
    SQL_MAX_ROWS    = 1000         # LIMIT injected when the model omits one
    SQL_TIMEOUT_S   = 10
    SQL_MAX_RETRIES = 1            # one regeneration attempt on a bad query

    # ── Conversation memory ──────────────────────────────────────────────
    MAX_TURNS = 6                  # rolling window kept in graph state

    # ── Devices ──────────────────────────────────────────────────────────
    @staticmethod
    def device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @classmethod
    def encoder_device(cls) -> torch.device:
        # BGE-M3 (XLM-RoBERTa) returns wrong embeddings for batch-of-1
        # encodes on Apple MPS — verified. CUDA and CPU are both correct.
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    # ── Resolvers ────────────────────────────────────────────────────────
    @classmethod
    def sqlite_path(cls) -> str:
        """Resolve the SQLite DB: env → Colab path → repo root."""
        if cls._SQLITE_ENV:
            return cls._SQLITE_ENV
        for cand in ("/content/interndb.sqlite",
                     os.path.join(cls.REPO_ROOT, "interndb.sqlite")):
            if os.path.isfile(cand):
                return cand
        return os.path.join(cls.REPO_ROOT, "interndb.sqlite")

    @classmethod
    def slm_id(cls) -> str:
        """Override → first local model dir that exists → Hub id.
        Honors V6_SLM_SIZE=3b|4b|7b (default 4b)."""
        if cls.SLM_OVERRIDE:
            return cls.SLM_OVERRIDE
        if cls.SLM_SIZE == "7b":
            candidates, hub = cls.SLM_CANDIDATES_7B, cls.SLM_HUB_ID_7B
        elif cls.SLM_SIZE == "4b":
            candidates, hub = cls.SLM_CANDIDATES_4B, cls.SLM_HUB_ID_4B
        else:
            candidates, hub = cls.SLM_CANDIDATES_3B, cls.SLM_HUB_ID_3B
        for name in candidates:
            path = os.path.join(cls.MODELS_DIR, name)
            if os.path.isdir(path):
                return path
        return hub

    @classmethod
    def draft_slm_id(cls, main_id: str) -> str | None:
        """Return the speculative-decoding drafter for the given main model.

        Qwen3 family (Qwen3-*, including Qwen3-4B-Instruct-2507) → Qwen3-0.6B:
        same tokenizer + same architecture family, best speculation quality.
        Qwen2.5-Coder → Qwen2.5-Coder-0.5B.
        Returns None when the main model IS the smallest available drafter.
        """
        m = main_id.lower()
        if "0.5b" in m or "0.6b" in m:
            return None   # already the drafter — no smaller model to speculate with

        if "qwen3" in m:
            local = os.path.join(cls.MODELS_DIR, "qwen3-0.6b")
            return local if os.path.isdir(local) else "Qwen/Qwen3-0.6B"

        for name in cls.SLM_CANDIDATES_3B:
            if "0.5b" in name:
                path = os.path.join(cls.MODELS_DIR, name)
                if os.path.isdir(path):
                    return path
        return "Qwen/Qwen2.5-Coder-0.5B-Instruct"

    @classmethod
    def bge_m3_id(cls) -> str:
        return (cls.BGE_M3_LOCAL_DIR if os.path.isdir(cls.BGE_M3_LOCAL_DIR)
                else cls.BGE_M3_HUB_ID)

    @classmethod
    def output_dir(cls) -> str:
        """Re-read the env var at call time so Cell 4 env changes take effect."""
        env_val = os.environ.get("V6_OUTPUT_DIR", os.environ.get("V5_OUTPUT_DIR", ""))
        return (env_val or
                ("/content/v6_output" if os.path.isdir("/content")
                 else os.path.join(cls.REPO_ROOT, "v6_output")))

    @classmethod
    def chart_dir(cls) -> str:
        d = os.path.join(cls.output_dir(), "charts")
        os.makedirs(d, exist_ok=True)
        return d

    @classmethod
    def report_dir(cls) -> str:
        d = os.path.join(cls.output_dir(), "reports")
        os.makedirs(d, exist_ok=True)
        return d

    @classmethod
    def audio_dir(cls) -> str:
        """Where generated speech (TTS output, test fixtures) is written."""
        d = os.path.join(cls.output_dir(), "audio")
        os.makedirs(d, exist_ok=True)
        return d
