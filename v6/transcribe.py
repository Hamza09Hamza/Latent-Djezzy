"""v6/transcribe.py — STT term biasing + post-correction for Algerian
telecom speech.

Generic Whisper mishears two things constantly in this domain:
  - Algerian wilaya names: "Ouargla" → "Wargla", "M'Sila" → "Mcila",
    "Bordj Bou Arréridj" → "Borge Bou Areridj".
  - telecom / finance jargon: "ARPU" → "are you" / "a r p u",
    "EBITDA" → "e bit da", "OPEX" → "op ex", "CAPEX" → "cap ex".

Two data-driven defences (both sourced from the live DB + KPI catalog, so
they never drift from the schema):

  1. build_bias_prompt() — an `initial_prompt` fed to Whisper that primes the
     decoder with the real wilaya and KPI vocabulary before decoding starts.
  2. correct_transcript() — RapidFuzz post-correction that snaps near-miss
     wilaya names back to their canonical DB spelling and normalises a
     curated set of KPI acronyms.

`speech.STT.transcribe` calls both. Nothing here loads an audio model.
RapidFuzz is optional: if it is missing, correction degrades to a no-op
(the bias prompt still works).
"""

from __future__ import annotations
import json
import re
import unicodedata

from .config import V6Config

try:
    from rapidfuzz import fuzz, process
    _HAS_RF = True
except Exception:  # noqa: BLE001 — correction degrades to a no-op
    _HAS_RF = False


# Hardcoded fallback used only when the DB is unreachable; the live
# dim_location table (via the resolver) is the real source of truth.
WILAYAS = [
    "Adrar", "Chlef", "Laghouat", "Oum El Bouaghi", "Batna", "Bejaia",
    "Biskra", "Bechar", "Blida", "Bouira", "Tamanrasset", "Tebessa",
    "Tlemcen", "Tiaret", "Tizi Ouzou", "Alger", "Djelfa", "Jijel", "Setif",
    "Saida", "Skikda", "Sidi Bel Abbes", "Annaba", "Guelma", "Constantine",
    "Medea", "Mostaganem", "M'Sila", "Mascara", "Ouargla", "Oran",
    "El Bayadh", "Illizi", "Bordj Bou Arreridj", "Boumerdes", "El Tarf",
    "Tindouf", "Tissemsilt", "El Oued", "Khenchela", "Souk Ahras", "Tipaza",
    "Mila", "Ain Defla", "Naama", "Ain Temouchent", "Ghardaia", "Relizane",
    "Timimoun", "Bordj Badji Mokhtar", "Ouled Djellal", "Beni Abbes",
    "In Salah", "In Guezzam", "Touggourt", "Djanet", "El M'Ghair",
    "El Meniaa",
]

# KPI display terms worth priming the decoder with (canonical, human form).
_KPI_BIAS_TERMS = [
    "ARPU", "EBITDA", "EBIT", "OPEX", "CAPEX", "OCF", "FCF", "DSO",
    "churn rate", "gross margin", "net income", "total revenue",
    "active base", "subscribers", "recharge rate", "migration rate",
    "market share", "gross adds", "net adds", "data usage", "voice usage",
    "EBITDA margin", "free cash flow", "operating cash flow",
]

# Acronyms generic Whisper reliably garbles; correction restores canonical
# casing. Kept length >= 3 and matched at a high threshold so common short
# words can't false-trigger. B2B/B2C added explicitly (digits, not [A-Z]+).
_KPI_ACRONYMS = ["ARPU", "EBITDA", "EBIT", "OPEX", "CAPEX", "OCF", "FCF",
                 "DSO", "ATV", "B2B", "B2C"]

# Words that must never be rewritten into an acronym even on a fuzzy hit.
_STOPWORDS = {"of", "off", "so", "the", "to", "too", "are", "our", "or",
              "for", "i", "a", "an", "is", "it", "be", "by", "do"}


# ── normalization ──────────────────────────────────────────────────────────
def normalize_text(text: str) -> str:
    """Lowercase, strip accents/apostrophes/punctuation for robust matching."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("'", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ── data-driven term sources ─────────────────────────────────────────────
def _live_wilayas() -> list[str]:
    """Canonical wilaya names from dim_location; fall back to the constant."""
    try:
        from .entities import get_resolver
        names = get_resolver().wilayas
        return names or WILAYAS
    except Exception:  # noqa: BLE001 — DB down → hardcoded list
        return WILAYAS


_bias_prompt: str | None = None


def build_bias_prompt() -> str:
    """The `initial_prompt` priming Whisper with wilaya + KPI vocabulary."""
    global _bias_prompt
    if _bias_prompt is not None:
        return _bias_prompt
    wilayas = ", ".join(_live_wilayas())
    kpis = ", ".join(_KPI_BIAS_TERMS)
    _bias_prompt = (
        "Conversation with the Djezzy telecom analytics assistant about "
        "Algerian wilayas and KPIs. "
        f"Wilaya names: {wilayas}. "
        f"Metrics and terms: {kpis}.")
    return _bias_prompt


# ── generic fuzzy n-gram replacer ───────────────────────────────────────────
def _fuzzy_replace(text: str, mapping: dict[str, str], threshold: int,
                   scorer, max_ngram: int = 3, collapse: bool = False,
                   min_key_chars: int = 3) -> str:
    """Replace fuzzy-matched n-grams with their canonical form.

    `mapping` is {normalized_key: canonical_output}. For each 1..max_ngram
    window the normalized (optionally space-collapsed) phrase is matched
    against the keys; the best non-overlapping matches above `threshold`
    win, longest+highest first.
    """
    if not _HAS_RF or not mapping:
        return text
    tokens = text.split()
    keys = list(mapping)

    candidates = []
    for i in range(len(tokens)):
        for n in range(1, max_ngram + 1):
            if i + n <= len(tokens):
                candidates.append((i, i + n, " ".join(tokens[i:i + n])))

    hits = []
    for start, end, phrase in candidates:
        norm = normalize_text(phrase)
        key = norm.replace(" ", "") if collapse else norm
        if len(key) < min_key_chars or key in _STOPWORDS:
            continue
        m = process.extractOne(key, keys, scorer=scorer)
        if m and m[1] >= threshold:
            hits.append((start, end, mapping[m[0]], m[1], end - start))

    hits.sort(key=lambda x: (x[3], x[4]), reverse=True)
    used: set[int] = set()
    final: dict[int, tuple[int, str]] = {}
    for start, end, canon, _score, _span in hits:
        if any(i in used for i in range(start, end)):
            continue
        used.update(range(start, end))
        final[start] = (end, canon)

    out, i = [], 0
    while i < len(tokens):
        if i in final:
            end, canon = final[i]
            out.append(canon)
            i = end
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out)


def correct_wilayas(text: str, threshold: int = 90) -> str:
    """Snap near-miss wilaya names to their canonical DB spelling.

    Deliberately conservative (threshold 90): the accent-only differences we
    most want to fix score ~100 (normalize_text strips accents on both sides,
    so "Setif"→"Sétif", "Ain Temouchent"→"Aïn Témouchent" are exact), while
    riskier consonant-level guesses are left for the bias prompt and the
    downstream entity resolver. A wrong "correction" (e.g. "Mcila"→"Mila",
    a different province) is worse than none — better to pass the raw token
    to the resolver, which can fail gracefully, than to invent a real but
    wrong wilaya.
    """
    mapping = {normalize_text(w): w for w in _live_wilayas()}
    return _fuzzy_replace(text, mapping, threshold,
                          scorer=fuzz.token_sort_ratio if _HAS_RF else None,
                          collapse=False, min_key_chars=3)


def correct_kpis(text: str, threshold: int = 90) -> str:
    """Restore garbled KPI acronyms (ARPU, EBITDA, OPEX, …) to canonical form.

    Matches on the SPACE-COLLAPSED phrase so split mishears like "a r p u"
    or "e bit da" snap back. Conservative: length >= 3, high threshold,
    stopword guard — so ordinary words are left untouched.
    """
    mapping = {a.lower(): a for a in _KPI_ACRONYMS}
    return _fuzzy_replace(text, mapping, threshold,
                          scorer=fuzz.ratio if _HAS_RF else None,
                          collapse=True, min_key_chars=3)


def correct_transcript(text: str) -> str:
    """Full post-correction: wilaya names first, then KPI acronyms."""
    if not text:
        return text
    return correct_kpis(correct_wilayas(text))
