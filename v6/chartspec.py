"""v6/chartspec.py — Build a typed chart spec (JSON) from a SQL result set.

The web client renders these with Recharts. The spec decides WHAT to chart —
type, axes, series — from the data shape + the question + the executed SQL, and
it carries every figure already FROZEN by numfmt: the raw numeric (for the axis)
AND the humanized string under `<col>_fmt` (for labels/tooltips). So the browser
chart can never re-round a figure — the exact same trust boundary as the text
and voice paths (see numfmt.py).

Deterministic. No LLM, no matplotlib. Returns None when there's nothing sensible
to chart (the caller just shows the text answer).

Spec shape (all keys stable — the frontend ChartView reads them):

    {
      "type": "line" | "area" | "bar" | "hbar" | "pie" | "stat",
      "title": "Top 5 wilayas by net revenue",
      "x": {"key": "wilaya", "label": "Wilaya"},        # omitted for "stat"
      "series": [{"key": "net_revenue", "label": "Net Revenue", "unit": "DZD"}],
      "data": [
        {"wilaya": "Alger", "net_revenue": 2.53e8,
         "net_revenue_fmt": "253.4 million DZD"}, ...
      ],
      "meta": {"rows": 5, "kind": "ranking", "lang": "en"}
    }

For a multi-series time chart the series keys are the category VALUES (one line
per wilaya) and each datum carries `<value>` + `<value>_fmt` for that line.
"""

from __future__ import annotations
import datetime
import re

from .numfmt import humanize_cell, unit_for_column

# An ISO date literal at the start of a value — what SQLite/MySQL emit for
# DATE/DATETIME columns. Detect the time axis by what values look like, not by
# the column's name (works for week_start, as_of_dt, period, …).
_ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}")
# Name-based fallback for a time axis when values aren't full ISO dates
# (e.g. a "2025-07" month label, or a "quarter" column).
_TIME_NAME = re.compile(
    r"(date|time|month|week|period|year|day|quarter|trimest|mois|semaine|ann[ée])",
    re.IGNORECASE)

# Acronyms to keep upper-cased when prettifying a column name into a label.
_ACRONYMS = {"arpu", "opex", "capex", "ebitda", "ebit", "dzd", "kpi", "b2b",
             "b2c", "sms", "gb", "fcf", "ocf", "dso", "id", "arr", "mrr"}

# Multilingual ranking / composition cues (en · fr · ar). Query is lowercased
# before matching; accents are kept (French) so include accented forms.
_RANK_DESC = ("top", "highest", "best", "most", "largest", "leading", "maximum",
              "meilleur", "plus élevé", "plus eleve", "plus haut", "plus grand",
              "أعلى", "أكبر", "أفضل")
_RANK_ASC = ("lowest", "worst", "least", "smallest", "minimum", "bottom",
             "pire", "plus bas", "plus faible", "plus petit", "moins",
             "أدنى", "أصغر", "أسوأ", "أقل")
_RANK_ANY = ("rank", "ranking", "classement", "premiers", "derniers",
             "ترتيب", "تصنيف")
_COMPOSITION = ("share", "proportion", "distribution", "composition", "mix",
                "répartition", "repartition", "ventilation", "part de",
                "parts de", "نسبة", "حصة", "توزيع")

_MAX_SERIES = 4        # cap simultaneous measures on one chart
_MAX_PIE_SLICES = 6    # a donut past this is unreadable → use a bar instead


# ── small predicates ─────────────────────────────────────────────────────────
def _looks_like_date(v) -> bool:
    if isinstance(v, (datetime.date, datetime.datetime)):
        return True
    if v is None:
        return False
    return bool(_ISO_DATE_RE.match(str(v)))


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _numeric_columns(rows: list[dict], columns: list[str]) -> list[str]:
    out = []
    for c in columns:
        vals = [r.get(c) for r in rows if r.get(c) is not None]
        if vals and all(_is_number(v) for v in vals):
            out.append(c)
    return out


def _pretty_label(col: str) -> str:
    if not col:
        return ""
    parts = re.split(r"[_\s]+", str(col).strip())
    out = [p.upper() if p.lower() in _ACRONYMS else p.capitalize()
           for p in parts if p]
    return " ".join(out) or str(col)


def _title(query: str) -> str:
    q = (query or "").strip().rstrip("?.!؟").strip()
    if not q:
        return "Query result"
    return (q[0].upper() + q[1:])[:90]


# ── column-role + sort detection ───────────────────────────────────────────
def _date_column(rows: list[dict], columns: list[str],
                 numeric: list[str]) -> str | None:
    # 1) values look like ISO dates
    for c in columns:
        if c in numeric:
            continue
        vals = [r.get(c) for r in rows if r.get(c) is not None]
        if vals and all(_looks_like_date(v) for v in vals):
            return c
    # 2) name says time + the column actually varies
    for c in columns:
        if c in numeric:
            continue
        if _TIME_NAME.search(c) and len({str(r.get(c)) for r in rows}) > 1:
            return c
    return None


def _sort_signal(sql: str, query: str) -> tuple[str | None, bool]:
    """(direction, is_ranking). direction ∈ {'asc','desc',None}.

    The executed SQL is the strongest signal — a ranking query carries an
    explicit ORDER BY … DESC/ASC (+ LIMIT). Fall back to question keywords.
    """
    s = (sql or "").lower()
    q = (query or "").lower()
    direction: str | None = None
    m = re.search(r"order\s+by\s+[^;]+?\b(asc|desc)\b", s)
    if m:
        direction = m.group(1)
    if direction is None:
        if any(k in q for k in _RANK_ASC):
            direction = "asc"
        elif any(k in q for k in _RANK_DESC):
            direction = "desc"
    has_limit = bool(re.search(r"\blimit\b", s))
    is_ranking = (has_limit or direction is not None
                  or any(k in q for k in _RANK_ANY))
    return direction, is_ranking


def _is_composition(query: str) -> bool:
    q = (query or "").lower()
    return any(k in q for k in _COMPOSITION)


# ── data packing (freezes every numeric via numfmt) ─────────────────────────
def _pack_rows(rows: list[dict], columns: list[str], numeric: list[str],
               lang: str) -> list[dict]:
    out: list[dict] = []
    for r in rows:
        d: dict = {}
        for c in columns:
            v = r.get(c)
            d[c] = v
            if c in numeric and _is_number(v):
                d[f"{c}_fmt"] = humanize_cell(c, v, lang)
        out.append(d)
    return out


def _series_of(numeric: list[str]) -> list[dict]:
    return [{"key": c, "label": _pretty_label(c), "unit": unit_for_column(c) or ""}
            for c in numeric[:_MAX_SERIES]]


def _pivot_timeseries(rows: list[dict], date_col: str, cat_col: str,
                      metric: str, lang: str) -> dict:
    """One line per category value: data keyed by date, a column per series."""
    from collections import OrderedDict
    cats: list[str] = []
    by_date: "OrderedDict[str, dict]" = OrderedDict()
    for r in sorted(rows, key=lambda x: str(x.get(date_col))):
        d = str(r.get(date_col))
        cv = str(r.get(cat_col, ""))
        if cv not in cats:
            cats.append(cv)
        row = by_date.setdefault(d, {date_col: d})
        row[cv] = r.get(metric)
        if _is_number(r.get(metric)):
            row[f"{cv}_fmt"] = humanize_cell(metric, r.get(metric), lang)
    unit = unit_for_column(metric) or ""
    series = [{"key": c, "label": c, "unit": unit} for c in cats[:_MAX_SERIES]]
    return {
        "type": "line",
        "x": {"key": date_col, "label": _pretty_label(date_col)},
        "series": series,
        "data": list(by_date.values()),
    }


# ── the builder ──────────────────────────────────────────────────────────────
def build_chart_spec(rows: list[dict], columns: list[str], query: str = "",
                     lang: str = "en", sql: str = "",
                     routing: dict | None = None) -> dict | None:
    """Return a chart spec for a result set, or None if it isn't chartable."""
    if not rows or not columns:
        return None
    numeric = _numeric_columns(rows, columns)
    date_col = _date_column(rows, columns, numeric)
    numeric = [c for c in numeric if c != date_col]
    if not numeric:
        return None
    cat_col = next((c for c in columns if c not in numeric and c != date_col),
                   None)
    title = _title(query)
    base_meta = {"rows": len(rows), "lang": lang}

    # 1) time series — trends over a date axis
    if date_col and len(rows) > 1:
        if cat_col and len({str(r.get(cat_col)) for r in rows}) > 1:
            spec = _pivot_timeseries(rows, date_col, cat_col, numeric[0], lang)
            spec["title"] = title
            spec["meta"] = {**base_meta, "kind": "trend",
                            "category": _pretty_label(cat_col)}
            return spec
        series = _series_of(numeric)
        ordered = sorted(rows, key=lambda r: str(r.get(date_col)))
        return {
            "type": "area" if len(series) == 1 else "line",
            "title": title,
            "x": {"key": date_col, "label": _pretty_label(date_col)},
            "series": series,
            "data": _pack_rows(ordered, columns, numeric, lang),
            "meta": {**base_meta, "kind": "trend"},
        }

    # 2) category vs measure — comparisons & rankings
    if cat_col:
        metric = numeric[0]
        direction, is_ranking = _sort_signal(sql, query)
        rs = list(rows)
        if direction == "asc":
            rs.sort(key=lambda r: (r.get(metric) is None,
                                   r.get(metric) if _is_number(r.get(metric)) else 0))
        elif direction == "desc" or is_ranking:
            rs.sort(key=lambda r: (r.get(metric) is None,
                                   -(r.get(metric) if _is_number(r.get(metric)) else 0)))
        data = _pack_rows(rs, columns, numeric, lang)

        # composition (donut): explicit share/répartition wording, few positive
        # single-measure slices. Conservative — a wrong pie is worse than a bar.
        if (_is_composition(query) and len(numeric) == 1 and len(rs) <= _MAX_PIE_SLICES
                and all((r.get(metric) or 0) >= 0 for r in rs)):
            return {
                "type": "pie", "title": title,
                "x": {"key": cat_col, "label": _pretty_label(cat_col)},
                "series": _series_of([metric]),
                "data": data,
                "meta": {**base_meta, "kind": "composition"},
            }

        long_labels = any(len(str(r.get(cat_col, ""))) > 10 for r in rs)
        horizontal = is_ranking or direction is not None or len(rs) > 6 or long_labels
        return {
            "type": "hbar" if horizontal else "bar",
            "title": title,
            "x": {"key": cat_col, "label": _pretty_label(cat_col)},
            "series": _series_of(numeric),
            "data": data,
            "meta": {**base_meta,
                     "kind": "ranking" if (is_ranking or direction) else "comparison",
                     "direction": direction or ("desc" if is_ranking else "")},
        }

    # 3) a single row — one KPI, or a bar across its measures
    if len(rows) == 1:
        if len(numeric) == 1:
            c = numeric[0]
            v = rows[0].get(c)
            return {
                "type": "stat", "title": title,
                "series": [{"key": c, "label": _pretty_label(c),
                            "unit": unit_for_column(c) or ""}],
                "data": [{c: v, f"{c}_fmt": humanize_cell(c, v, lang)}],
                "meta": {**base_meta, "kind": "single"},
            }
        # Keep only the dominant unit group so DZD and % don't share one axis.
        from collections import Counter
        unit_groups: dict[str, list[str]] = {}
        for c in numeric:
            u = unit_for_column(c) or "DZD"
            unit_groups.setdefault(u, []).append(c)
        dominant_unit = max(unit_groups, key=lambda u: len(unit_groups[u]))
        chartable = unit_groups[dominant_unit]
        data = [{"metric": _pretty_label(c), "value": rows[0].get(c),
                 "value_fmt": humanize_cell(c, rows[0].get(c), lang)}
                for c in chartable]
        return {
            "type": "bar", "title": title,
            "x": {"key": "metric", "label": "Metric"},
            "series": [{"key": "value", "label": "Value",
                        "unit": dominant_unit if dominant_unit != "DZD" else ""}],
            "data": data,
            "meta": {**base_meta, "kind": "metrics"},
        }

    return None


if __name__ == "__main__":   # pragma: no cover — quick visual check
    import json
    samples = [
        ("Top 5 wilayas by net revenue",
         [{"wilaya": "Alger", "net_revenue": 253387711.02},
          {"wilaya": "Oran", "net_revenue": 198120044.5},
          {"wilaya": "Constantine", "net_revenue": 142000000.0}],
         "SELECT wilaya, SUM(net_revenue) net_revenue ... ORDER BY net_revenue DESC LIMIT 5"),
        ("Net income trend for Alger over 2025",
         [{"month_start": "2025-01-01", "net_income": 12_000_000},
          {"month_start": "2025-02-01", "net_income": 13_500_000}],
         "SELECT month_start, net_income ... ORDER BY month_start"),
        ("Total revenue for Oran last month",
         [{"total_revenue": 1_087_355_290.78}], "SELECT SUM(...) ..."),
    ]
    for q, rows, sql in samples:
        spec = build_chart_spec(rows, list(rows[0].keys()), q, "en", sql=sql)
        print(f"\n— {q}\n  type={spec['type']}  kind={spec['meta'].get('kind')}")
        print("  " + json.dumps(spec, ensure_ascii=False)[:300])
