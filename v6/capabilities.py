"""v6/capabilities.py — Output capabilities beyond a plain SQL answer.

Three things the graph can do with a result set, each a pure function the
capability nodes call:

  make_chart()           — render a matplotlib PNG (line for trends, bar for
                           comparisons) and return its path.
  compose_email_draft()  — resolve a recipient from the `contacts` table and
                           fill an email body. It DRAFTS only; send_email()
                           is a separate, explicit action the graph never
                           calls on its own.
  fill_report()          — render a Jinja2 report template with the data and
                           write it to disk.
"""

from __future__ import annotations
import datetime
import os

import matplotlib
matplotlib.use("Agg")                       # headless — no display needed
import matplotlib.pyplot as plt             # noqa: E402
from jinja2 import Environment, FileSystemLoader, select_autoescape  # noqa: E402

from .config import V6Config                # noqa: E402
from .schema import db_connect              # noqa: E402

_env = Environment(
    loader=FileSystemLoader(V6Config.TEMPLATE_DIR),
    autoescape=select_autoescape(["html", "xml"]),
    trim_blocks=True, lstrip_blocks=True)

_DATE_HINTS = ("week_start", "week", "date", "month", "day", "period")


# ── shared helpers ───────────────────────────────────────────────────────
def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:,.2f}"
    return str(v)


def _markdown_table(rows: list[dict], columns: list[str], limit: int = 30) -> str:
    if not rows or not columns:
        return "_(no rows)_"
    head = "| " + " | ".join(str(c) for c in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(_fmt(r.get(c)) for c in columns) + " |"
            for r in rows[:limit]]
    out = "\n".join([head, sep] + body)
    if len(rows) > limit:
        out += f"\n\n_({len(rows) - limit} more rows not shown)_"
    return out


def _numeric_columns(rows: list[dict], columns: list[str]) -> list[str]:
    out = []
    for c in columns:
        vals = [r.get(c) for r in rows if r.get(c) is not None]
        if vals and all(_is_number(v) for v in vals):
            out.append(c)
    return out


def summarize(rows: list[dict], columns: list[str]) -> list[str]:
    """One human-readable stat line per numeric column."""
    lines: list[str] = []
    for c in _numeric_columns(rows, columns):
        vals = [r.get(c) for r in rows if _is_number(r.get(c))]
        if not vals:
            continue
        lines.append(
            f"**{c}** — min {min(vals):,.2f}, max {max(vals):,.2f}, "
            f"avg {sum(vals) / len(vals):,.2f} over {len(vals)} value(s)")
    return lines


# ── 1. visualization ─────────────────────────────────────────────────────
def make_chart(rows: list[dict], columns: list[str], query: str = "",
               title: str | None = None) -> dict:
    """Render a chart from a result set. Returns {ok, path, chart_type, error}."""
    if not rows or not columns:
        return {"ok": False, "path": "", "chart_type": "",
                "error": "no data to chart"}

    date_col = next(
        (c for c in columns
         if any(h in c.lower() for h in _DATE_HINTS)), None)
    numeric = [c for c in _numeric_columns(rows, columns) if c != date_col]
    if not numeric:
        return {"ok": False, "path": "", "chart_type": "",
                "error": "no numeric column to plot"}
    cat_col = next((c for c in columns
                    if c not in numeric and c != date_col), None)

    fig, ax = plt.subplots(figsize=(9, 5))
    try:
        if date_col and len(rows) > 1:
            xs = [str(r.get(date_col)) for r in rows]
            for c in numeric[:3]:
                ax.plot(xs, [r.get(c) for r in rows], marker="o", label=c)
            ax.set_xlabel(date_col)
            chart_type = "line"
            if len(numeric) > 1:
                ax.legend()
            plt.xticks(rotation=45, ha="right")
        elif cat_col:
            labels = [str(r.get(cat_col)) for r in rows]
            y = numeric[0]
            ax.bar(labels, [r.get(y) or 0 for r in rows])
            ax.set_xlabel(cat_col)
            ax.set_ylabel(y)
            chart_type = "bar"
            plt.xticks(rotation=45, ha="right")
        else:                                   # one row → bar over metrics
            ax.bar(numeric, [rows[0].get(c) or 0 for c in numeric])
            chart_type = "bar"
        ax.set_title(title or (query[:80] if query else "Query result"))
        fig.tight_layout()
        os.makedirs(V6Config.chart_dir(), exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = os.path.join(V6Config.chart_dir(), f"chart_{stamp}.png")
        fig.savefig(path, dpi=120)
        return {"ok": True, "path": path, "chart_type": chart_type,
                "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "path": "", "chart_type": "", "error": str(exc)}
    finally:
        plt.close(fig)


# ── 2. email drafting ────────────────────────────────────────────────────
def load_contacts() -> list[dict]:
    """The recipient book — the database's `contacts` table."""
    try:
        conn = db_connect()
        cur = conn.cursor()
        cur.execute("SELECT id, email, name, role, department FROM contacts")
        out = [{"id": r[0], "email": r[1], "name": r[2],
                "role": r[3], "department": r[4]} for r in cur.fetchall()]
        conn.close()
        return out
    except Exception:  # noqa: BLE001
        return []


def resolve_recipient(query: str, contacts: list[dict]) -> tuple[dict | None, list[dict]]:
    """Best-matching contact for the query, plus all scored candidates."""
    q = (query or "").lower()
    scored: list[tuple[int, dict]] = []
    for c in contacts:
        score = 0
        for part in (c.get("name") or "").lower().split():
            if len(part) > 2 and part in q:
                score += 3
        role = (c.get("role") or "").lower()
        dept = (c.get("department") or "").lower()
        if role and role in q:
            score += 2
        if dept and dept in q:
            score += 2
        if score:
            scored.append((score, c))
    scored.sort(key=lambda x: -x[0])
    if not scored:
        return None, []
    return scored[0][1], [c for _, c in scored]


def compose_email_draft(query: str, answer: str, rows: list[dict],
                        columns: list[str]) -> dict:
    """Draft an email — never sends. Returns a draft dict with status."""
    contacts = load_contacts()
    recipient, candidates = resolve_recipient(query, contacts)

    table = _markdown_table(rows, columns, limit=15) if rows else ""
    subject = "Telecom analytics: " + (query.strip()[:60] or "your request")
    name = (recipient or {}).get("name", "there")
    intro = "Here are the analytics figures you asked for."

    body = _env.get_template("email_report.md.j2").render(
        recipient_name=name, intro=intro, answer=answer,
        table=table, query=query)

    if recipient is None:
        return {"to": None, "to_name": None, "subject": subject, "body": body,
                "status": "needs_recipient", "candidates": contacts,
                "note": "No recipient named — pick one from contacts."}
    return {"to": recipient["email"], "to_name": recipient["name"],
            "subject": subject, "body": body, "status": "draft",
            "candidates": candidates,
            "note": ("Drafted — not sent. Call capabilities.send_email(draft) "
                     "to send after review."
                     + (f" ({len(candidates)} contacts matched; "
                        f"using {recipient['name']}.)"
                        if len(candidates) > 1 else ""))}


def send_email(draft: dict, smtp_host: str | None = None,
               smtp_port: int = 587, user: str | None = None,
               password: str | None = None) -> dict:
    """Actually send a drafted email over SMTP.

    Deliberately NOT wired into the graph: the graph drafts, a human sends.
    Credentials come from V6_SMTP_USER / V6_SMTP_PASSWORD / V6_SMTP_HOST.
    """
    import smtplib
    from email.mime.text import MIMEText

    user = user or os.environ.get("V6_SMTP_USER")
    password = password or os.environ.get("V6_SMTP_PASSWORD")
    smtp_host = smtp_host or os.environ.get("V6_SMTP_HOST", "smtp.gmail.com")
    if not (draft.get("to") and user and password):
        return {"ok": False, "error": "missing recipient or SMTP credentials"}

    msg = MIMEText(draft["body"])
    msg["Subject"] = draft["subject"]
    msg["From"] = user
    msg["To"] = draft["to"]
    try:
        with smtplib.SMTP(smtp_host, smtp_port) as srv:
            srv.starttls()
            srv.login(user, password)
            srv.send_message(msg)
        return {"ok": True, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}


# ── 3. report template filling ───────────────────────────────────────────
def fill_report(query: str, rows: list[dict], columns: list[str],
                answer: str, entities: dict | None = None) -> dict:
    """Render the report template with the data and write it to disk."""
    try:
        table = _markdown_table(rows, columns, limit=50)
        content = _env.get_template("report.md.j2").render(
            generated_at=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            query=query, answer=answer, table=table,
            row_count=len(rows), columns=columns,
            stats_lines=summarize(rows, columns),
            entities=entities or {})
        os.makedirs(V6Config.report_dir(), exist_ok=True)
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(V6Config.report_dir(), f"report_{stamp}.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return {"ok": True, "path": path, "error": None}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "path": "", "error": str(exc)}
