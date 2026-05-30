"""v6/numfmt.py — Deterministic, locale-aware number humanization.

WHY THIS EXISTS
---------------
This is an analytics assistant: a wrong number is worse than no number. The
polisher is a 1.5B model, and a model that small mangles long figures — we have
proof ("52,590,189,81", a double-comma corruption it produced while "rounding"
1,087,355,290.78). The fix is to never ask the model to format a number at all.

Numbers are rounded and given a scale word HERE, in plain Python, before they
ever reach the polisher. The polisher's job becomes pure prose: it copies the
already-clean figure verbatim. A frozen "253.4 million DZD" is trivial to copy;
a raw "253,387,711.02" is what invites corruption.

The same clean figure is also what the TTS layer speaks, so the voice says
"two hundred fifty three point four million dinars" instead of reading a
twelve-digit number digit by digit.

UNITS come from data/kpi_catalog.json (column → unit), with a keyword fallback
for aliased/aggregated columns (avg_, total_, …) the catalog doesn't list
verbatim.
"""

from __future__ import annotations
import json
import os
import re

from .config import V6Config

# ── column → unit, sourced from the KPI catalog (loaded once) ───────────────
_UNIT_CACHE: dict[str, str] | None = None
# Aggregation/qualifier prefixes a SQL alias may wrap a base column in, e.g.
# avg_gross_margin → gross_margin, total_arpu → arpu.
_AGG_PREFIX = re.compile(
    r"^(avg|average|mean|median|total|sum|min|max|count|num|n|cnt)_")


def _unit_map() -> dict[str, str]:
    global _UNIT_CACHE
    if _UNIT_CACHE is not None:
        return _UNIT_CACHE
    out: dict[str, str] = {}
    try:
        with open(V6Config.KPI_CATALOG_PATH, encoding="utf-8") as fh:
            for row in json.load(fh):
                col, unit = row.get("column"), row.get("unit")
                if col and unit:
                    out[col.lower()] = unit
    except Exception:  # noqa: BLE001 — catalog missing → pure heuristics below
        pass
    _UNIT_CACHE = out
    return out


def unit_for_column(col: str) -> str | None:
    """Best-effort unit for a (possibly aggregated/aliased) SQL column.

    Resolution order: exact catalog hit → catalog hit after stripping an
    aggregation prefix → keyword heuristic. Returns one of
    'DZD' | '%' | 'count' | 'GB' | 'min' | 'months' | 'days', or None.
    """
    if not col:
        return None
    c = col.lower().strip()
    catalog = _unit_map()
    if c in catalog:
        return catalog[c]
    stripped = _AGG_PREFIX.sub("", c)
    if stripped in catalog:
        return catalog[stripped]

    # Keyword fallback — ordered so the more specific units win first.
    if stripped.endswith("_gb") or "usage_gb" in stripped:
        return "GB"
    if "dso" in stripped or stripped.endswith("_days"):
        return "days"
    if stripped.endswith("_min") or "voice_usage" in stripped or "minutes" in stripped:
        return "min"
    if "tenure" in stripped or stripped.endswith("_months"):
        return "months"
    if any(k in stripped for k in
           ("rate", "ratio", "margin", "share", "pct", "percent")):
        return "%"
    if any(k in stripped for k in
           ("revenue", "income", "opex", "capex", "ebitda", "ebit",
            "arpu", "fcf", "ocf", "recharge", "cost")):
        return "DZD"
    if any(k in stripped for k in
           ("subscriber", "active_base", "adds", "count", "base")):
        return "count"
    return None


# ── scale words ─────────────────────────────────────────────────────────────
_SCALE = {
    "en": {"b": "billion", "m": "million", "pct": "%"},
    "fr": {"b": "milliards", "bs": "milliard", "m": "millions", "ms": "million",
           "pct": " %"},
    "ar": {"b": "مليار", "m": "مليون", "pct": "٪"},
}


def _trim(s: str) -> str:
    """Drop trailing zeros and a dangling decimal separator: 3.50→3.5, 42.00→42."""
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    elif "," in s and re.fullmatch(r"\d+,\d+", s):   # french decimal
        s = s.rstrip("0").rstrip(",")
    return s


def _dec(value: float, places: int, lang: str) -> str:
    """Format a float to `places` decimals, then trim, using the locale's
    decimal mark (',' for French)."""
    s = _trim(f"{value:.{places}f}")
    return s.replace(".", ",") if lang == "fr" else s


def _grouped(value: float, lang: str) -> str:
    """Whole number with thousands separators in the locale's style."""
    n = f"{round(value):,}"                 # 253388 → '253,388'
    if lang == "fr":                        # french groups with thin space
        n = n.replace(",", " ")
    return n


def humanize(value, unit: str | None, lang: str = "en") -> str:
    """Render one numeric value as a clean, speech-ready string with a scale
    word (for large figures) and its unit. Non-numeric values pass through
    unchanged; None becomes an em dash.

    Examples (en): 1_087_355_290.78,'DZD' → '1.09 billion DZD'
                   253_387_711.02,'DZD'   → '253.4 million DZD'
                   42.4247,'%'            → '42.42%'
                   3.071,'DZD'            → '3.07 DZD'
    Examples (fr): 470_403_313.75,'DZD'   → '470,4 millions DZD'
                   42.4247,'%'            → '42,42 %'
    """
    if value is None:
        return "—"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)

    sw = _SCALE.get(lang, _SCALE["en"])
    neg = "-" if value < 0 else ""
    v = abs(float(value))

    # Percentages and other rate-like units: 2 decimals, keep the symbol.
    if unit == "%":
        return f"{neg}{_dec(v, 2, lang)}{sw['pct']}"

    money_like = unit in ("DZD", None, "count")
    if money_like:
        if v >= 1_000_000_000:
            num = _dec(v / 1_000_000_000, 2, lang)
            scale = sw["b"]
            if lang == "fr" and v < 2_000_000_000:
                scale = sw["bs"]
            body = f"{neg}{num} {scale}"
        elif v >= 1_000_000:
            num = _dec(v / 1_000_000, 1, lang)
            scale = sw["m"]
            if lang == "fr" and v < 2_000_000:
                scale = sw["ms"]
            body = f"{neg}{num} {scale}"
        elif v >= 1000:
            body = f"{neg}{_grouped(v, lang)}"
        else:
            body = f"{neg}{_dec(v, 2, lang)}"
        if unit == "DZD":
            body += " DZD"
        return body

    # Physical units (GB, minutes, months, days): 1–2 decimals + unit word.
    return f"{neg}{_dec(v, 2, lang)} {unit}"


def humanize_cell(col: str, value, lang: str = "en") -> str:
    """Humanize a value using the unit inferred from its column name."""
    return humanize(value, unit_for_column(col), lang)


if __name__ == "__main__":   # pragma: no cover — quick visual check
    cases = [
        ("total_revenue", 1_087_355_290.78),
        ("total_revenue", 253_387_711.02),
        ("net_income", 470_403_313.75),
        ("avg_gross_margin", 42.4247),
        ("avg_churn_rate", 1.3675),
        ("avg_arpu", 3.071),
        ("avg_migration_rate", 0.9218),
        ("active_base", 1_240_000),
        ("avg_data_usage_gb", 4.05),
        ("month_start", "2025-07-01"),
        ("wilaya", "Tizi Ouzou"),
        ("net_income", None),
    ]
    for lang in ("en", "fr"):
        print(f"── {lang} ──")
        for col, val in cases:
            print(f"  {col:22} {str(val):18} → {humanize_cell(col, val, lang)}")
