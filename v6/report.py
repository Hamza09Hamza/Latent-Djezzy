"""v6/report.py — Build a structured ReportSpec from query results.

The spec is a JSON document that mirrors what frontend ReportView.tsx renders:
metric cards, headings, body text, an embedded chart, and a data table.
It is emitted as `reportSpec` in the artifact event so the web client can
render it interactively (with real Recharts charts) and print to PDF.

Shape (TypeScript interface is in v6/frontend/lib/api.ts):
    {
        title: str,
        subtitle: str,
        generated_at: str  (ISO-8601 UTC),
        sections: [
            {type: "metrics", items: [{label, value, change?, trend?}]},
            {type: "heading", text: str},
            {type: "text",    text: str},
            {type: "chart",   spec: ChartSpec},
            {type: "table",   columns: [...], rows: [[...], ...]},
        ]
    }
"""

from __future__ import annotations
from datetime import datetime, timezone
from typing import Any

from .numfmt import humanize_cell
from .chartspec import build_chart_spec, _pretty_label

# ── locale strings ────────────────────────────────────────────────────────────

_L = {
    "heading_summary": {"en": "Executive Summary", "fr": "Résumé exécutif", "ar": "ملخص تنفيذي"},
    "heading_data":    {"en": "Detailed Data",      "fr": "Données détaillées", "ar": "البيانات التفصيلية"},
    "subtitle":        {"en": "Djezzy Analytics Report", "fr": "Rapport analytique Djezzy", "ar": "تقرير تحليلات جيزي"},
}


def _t(key: str, lang: str) -> str:
    return _L[key].get(lang, _L[key]["en"])


# ── helpers ───────────────────────────────────────────────────────────────────

def _fmt_cell(col: str, val: Any, lang: str) -> str:
    """Humanize a table cell the same way the answer text does, so the report
    shows '2.68 billion DZD', never the raw float '2684113522.9099994'. A
    pre-frozen '{col}_fmt' on the row wins if present; otherwise we humanize by
    the column's inferred unit. Strings/dates pass through; None → em dash."""
    if val is None:
        return "—"
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return str(val)
    return humanize_cell(col, val, lang)


def _extract_metrics(
    rows: list[dict],
    columns: list[str],
    chart_spec: dict | None,
    lang: str = "en",
) -> list[dict]:
    """Pull the first few numeric KPIs from the result as metric chips, each
    formatted with numfmt (so the chip reads '2.68 billion DZD')."""
    if not rows:
        return []
    first = rows[0]
    items: list[dict] = []
    for col in columns:
        val = first.get(col)
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            continue
        fmt_val = first.get(f"{col}_fmt") or humanize_cell(col, val, lang)
        label = col.replace("_", " ").title()
        items.append({"label": label, "value": fmt_val})
        if len(items) >= 4:
            break
    return items


def _make_title(query: str, lang: str) -> str:
    """Best-effort short title from the query text (first ~60 chars)."""
    q = query.strip()
    if len(q) > 60:
        q = q[:57].rsplit(" ", 1)[0] + "…"
    return q or _t("subtitle", lang)


# ── public API ────────────────────────────────────────────────────────────────

def build_report_spec(
    query: str,
    answer_text: str,
    chart_spec: dict | None,
    rows: list[dict],
    columns: list[str],
    lang: str = "en",
    generated_at: str | None = None,
    title: str | None = None,
) -> dict:
    """Return a ReportSpec dict ready to be JSON-serialised into the artifact event.

    Args:
        query        -- the original user query (used for the title)
        answer_text  -- the communicator's final answer (executive summary)
        chart_spec   -- the ChartSpec built by chartspec.py (may be None)
        rows         -- SQL result rows (list of dicts)
        columns      -- SQL column order
        lang         -- "en" | "fr" | "ar"
        generated_at -- ISO-8601 timestamp; defaults to now (UTC)
        title        -- override auto-generated title
    """
    if generated_at is None:
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    report_title = title or _make_title(query, lang)
    subtitle = _t("subtitle", lang)

    # Auto-generate a chart spec when the caller didn't provide one (i.e. the
    # brain routed sql → template without a separate chart step). The report
    # should always include a visual when the data is chartable.
    if chart_spec is None and rows and columns:
        chart_spec = build_chart_spec(rows, columns, query, lang=lang)

    sections: list[dict] = []

    # 1 — Metric chips (top KPIs from first result row)
    metrics = _extract_metrics(rows, columns, chart_spec, lang)
    if metrics:
        sections.append({"type": "metrics", "items": metrics})

    # 2 — Executive summary
    sections.append({"type": "heading", "text": _t("heading_summary", lang)})
    if answer_text:
        sections.append({"type": "text", "text": answer_text})

    # 3 — Chart (reuses or auto-built ChartSpec; renders with Recharts in the browser)
    if chart_spec:
        sections.append({"type": "chart", "spec": chart_spec})

    # 4 — Data table (capped at 25 rows; human-readable column headers)
    if rows and columns:
        sections.append({"type": "heading", "text": _t("heading_data", lang)})
        table_rows = [
            [_fmt_cell(c, row.get(c), lang) for c in columns]
            for row in rows[:25]
        ]
        sections.append({
            "type": "table",
            "columns": [_pretty_label(c) for c in columns],
            "rows": table_rows,
        })

    return {
        "title": report_title,
        "subtitle": subtitle,
        "generated_at": generated_at,
        "sections": sections,
    }
