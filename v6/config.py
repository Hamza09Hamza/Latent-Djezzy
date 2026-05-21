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

    GLOSSARY_PATH    = os.path.join(DATA_DIR, "glossary.json")
    KPI_CATALOG_PATH = os.path.join(DATA_DIR, "kpi_catalog.json")

    # Charts and rendered reports land here (created on demand).
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
    SLM_CANDIDATES = [
        "qwen2.5-coder-3b-instruct",
        "qwen2.5-coder-1.5b-instruct",
        "qwen2.5-coder-0.5b-instruct",
    ]
    SLM_OVERRIDE   = _env("SLM_OVERRIDE")          # force a Hub model id
    SLM_HUB_ID     = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    USE_4BIT         = _env("4BIT", "0") == "1"      # 4-bit NF4 quantization
    USE_SPECULATIVE  = _env("SPECULATIVE", "1") == "1"  # 0.5B drafter → 2-4x speed

    ROUTER_MAX_NEW_TOKENS = 128   # routing JSON rarely exceeds 80 tokens
    SQLGEN_MAX_NEW_TOKENS = 160   # SQL rarely exceeds 120 tokens

    # ── RAG encoder — frozen BGE-M3 (1024-d) ─────────────────────────────
    BGE_M3_LOCAL_DIR = os.path.join(MODELS_DIR, "bge-m3")
    BGE_M3_HUB_ID    = "BAAI/bge-m3"
    EMBED_DIM        = 1024
    RAG_TOP_K        = 5
    RAG_LOW_CONF     = 0.45        # top cosine below this → weak grounding

    # ── Latent planner — decides intent + capabilities ───────────────────
    # The planner classifies the query in BGE-M3 embedding space, not with
    # regex. Mode "prototype" needs no training (nearest-prototype). Mode
    # "mlp" loads a trained head — see planner_data.py and train_planner.py.
    PLANNER_MODE       = _env("PLANNER", "prototype")     # prototype | mlp
    PLANNER_HEAD_PATH  = os.path.join(MODELS_DIR, "planner_head.pt")
    PLANNER_PROTOTYPES = os.path.join(DATA_DIR, "planner_prototypes.json")
    CAP_THRESHOLD      = 0.46      # capability cosine above this → active
    PLANNER_LOW_CONF   = 0.40      # intent score below this → weak plan
    MAX_REPLAN         = 1         # times the graph may loop back to re-plan

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
        """Override → first local model dir that exists → Hub id."""
        if cls.SLM_OVERRIDE:
            return cls.SLM_OVERRIDE
        for name in cls.SLM_CANDIDATES:
            path = os.path.join(cls.MODELS_DIR, name)
            if os.path.isdir(path):
                return path
        return cls.SLM_HUB_ID

    @classmethod
    def draft_slm_id(cls, main_id: str) -> str | None:
        """Return the speculative-decoding drafter id for the given main model.

        If the main model is the 0.5B, no drafter is needed (it IS the drafter).
        Otherwise return the 0.5B Hub id so the verifier can use it.
        """
        if "0.5b" in main_id.lower():
            return None
        # prefer a local copy if present
        for name in cls.SLM_CANDIDATES:
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
    def chart_dir(cls) -> str:
        d = os.path.join(cls.OUTPUT_DIR, "charts")
        os.makedirs(d, exist_ok=True)
        return d

    @classmethod
    def report_dir(cls) -> str:
        d = os.path.join(cls.OUTPUT_DIR, "reports")
        os.makedirs(d, exist_ok=True)
        return d
