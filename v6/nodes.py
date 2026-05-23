"""v6/nodes.py — The LangGraph node functions for the policy loop.

The graph is a star: every action returns to the brain, which re-decides.

  brain        — one tick of the policy MLP: intent + next action + the
                 continue score (the seuil). route_after_brain reads it.
  rag          — retrieve KPI knowledge, record the grounding score.
  sql          — the full router → validate → generate → execute chain
                 (run_sql_pipeline), reusing every v6 SQL function as-is.
  chart/email/template — the output capabilities, one per action.
  communicator — terminal: compose the final answer from whatever the
                 loop produced, and roll the conversation memory.

Every action node appends one outcome dict to `step_log`; that history is
exactly what the brain reads back through encode_outcome on the next tick.
Nodes also append short `thoughts` — the feed the notebook streams as the
"thinking" UX.
"""

from __future__ import annotations
import time

from .brain import get_brain, row_bucket
from .capabilities import compose_email_draft, fill_report, make_chart
from .config import V6Config
from .entities import get_resolver
from .knowledge import get_retriever
from .orchestrator import assemble
from .prompts import (build_router_messages, build_sqlgen_instruction,
                      parse_router_output)
from .schema import get_db_schema
from .slm import get_slm
from .sql_tools import (clean_sql, consistency_check, correction_hint,
                        enforce_limit, execute_sql, validate_sql)

_MAX_COMPACT_CHARS = 600
_MAX_RAW_TURNS = 2
_ACTIONS = ("rag", "sql", "chart", "email", "template")


# ── small helpers ────────────────────────────────────────────────────────
def _trace(state: dict, *msgs: str) -> list[str]:
    return list(state.get("trace", [])) + list(msgs)


def _timing(state: dict, key: str, ms: float) -> dict:
    t = dict(state.get("timings", {}))
    t[key] = round(ms, 1)
    return t


def _thoughts(state: dict, *items: dict) -> list[dict]:
    """Append to the streamed thinking feed."""
    return list(state.get("thoughts", [])) + list(items)


def _step(state: dict, entry: dict) -> list[dict]:
    """Append one executed-action outcome to the brain's step log."""
    return list(state.get("step_log", [])) + [entry]


def _attempt(state: dict, action: str) -> int:
    """1 for the first run of `action` this turn, 2 for the next, ..."""
    return 1 + sum(1 for s in state.get("step_log", [])
                   if s.get("action") == action)


def _compact_turns(turns: list[dict]) -> str:
    """Deterministic compression: extract key facts without LLM overhead."""
    parts = []
    for t in turns:
        q = (t.get("query") or "")[:70]
        intent = t.get("intent", "")
        answer = (t.get("answer") or "")
        if not q:
            continue
        if intent == "data" and answer:
            parts.append(f"Q: {q}\nA: {answer[:180]}")
        elif intent == "definition":
            parts.append(f"Q: {q} [definition]")
    return ("\n---\n".join(parts))[:_MAX_COMPACT_CHARS] if parts else ""


def _history_text(turns: list[dict], memory_summary: str = "",
                  limit: int = _MAX_RAW_TURNS) -> str:
    """Recent raw turns + compacted older memory, for prompts and the brain."""
    if not turns and not memory_summary:
        return ""
    lines: list[str] = []
    if memory_summary:
        lines.append(f"[Earlier context]\n{memory_summary}")
    for i, t in enumerate(turns[-limit:], 1):
        lines.append(f"{i}. Q: {t.get('query', '')}")
        if t.get("intent") == "data" and t.get("answer"):
            lines.append(f"   A: {t.get('answer', '')[:100]}")
    return "\n".join(lines)


def _fmt_val(v) -> str:
    if isinstance(v, float):
        if abs(v) >= 1_000:
            return f"{v:,.2f}"
        return f"{v:.4f}"
    if isinstance(v, int) and abs(v) >= 1_000:
        return f"{v:,}"
    return str(v)


def _summarize_rows(rows: list[dict], cols: list[str]) -> str:
    n = len(rows)
    if n == 1:
        return " | ".join(f"{c}: {_fmt_val(rows[0].get(c))}" for c in cols)
    head = rows[:8]
    body = "\n".join(
        "  " + " | ".join(f"{c}: {_fmt_val(r.get(c))}" for c in cols)
        for r in head)
    more = f"\n  ... ({n - 8} more rows)" if n > 8 else ""
    return f"{n} rows returned:\n{body}{more}"


# ── the brain ────────────────────────────────────────────────────────────
_ACTION_THOUGHT = {
    "rag": "Let me check the reference knowledge first.",
    "sql": "I'll query the database for the numbers.",
    "chart": "The data's in — let me turn it into a chart.",
    "email": "Let me draft an email with these results.",
    "template": "Let me put this into a report.",
}


def _brain_thought(action: str, will_stop: bool, step_log: list[dict]) -> str:
    if will_stop:
        return ("Let me answer that directly." if not step_log
                else "I have what I need — writing the answer now.")
    if action == "sql" and any(s.get("action") == "sql" for s in step_log):
        return "That query didn't land — let me try it a different way."
    return _ACTION_THOUGHT.get(action, "Working on the next step.")


def brain_node(state: dict) -> dict:
    """One tick of the policy MLP — pick the next action, judge the seuil."""
    t0 = time.time()
    step = state.get("brain_step", 0)
    memory = _history_text(state.get("turns", []),
                           state.get("memory_summary", ""))
    decision = get_brain().decide(
        state["query"], memory, state.get("step_log", []),
        grounding=state.get("grounding", 0.0),
        thread_id=state.get("thread_id", "default"))

    # intent is decided once, on the first tick, then held for the turn
    intent = state.get("intent") or decision.intent

    will_stop = (step + 1 >= V6Config.BRAIN_MAX_STEPS
                 or decision.continue_score < V6Config.BRAIN_SEUIL
                 or decision.action_conf < V6Config.BRAIN_CONF_MIN)
    thought = _brain_thought(decision.action, will_stop,
                             state.get("step_log", []))

    return {
        "brain_step": step + 1,
        "intent": intent,
        "next_action": decision.action,
        "continue_score": decision.continue_score,
        "brain_scores": {
            "intent": decision.intent_scores,
            "action": decision.action_scores,
            "action_conf": decision.action_conf,
            "continue": decision.continue_score,
        },
        "thoughts": _thoughts(state, {"kind": "thinking", "text": thought}),
        "trace": _trace(state, f"brain#{step}: intent={intent} "
                               f"next={decision.action} "
                               f"cont={decision.continue_score} "
                               f"conf={decision.action_conf} "
                               f"{'STOP' if will_stop else 'GO'}"),
        "timings": _timing(state, f"brain{step}_ms", (time.time() - t0) * 1000),
    }


def route_after_brain(state: dict) -> str:
    """The seuil: keep looping, or hand off to the communicator."""
    if state.get("brain_step", 0) >= V6Config.BRAIN_MAX_STEPS:
        return "communicator"
    if state.get("continue_score", 0.0) < V6Config.BRAIN_SEUIL:
        return "communicator"
    conf = (state.get("brain_scores", {}) or {}).get("action_conf", 1.0)
    if conf < V6Config.BRAIN_CONF_MIN:
        return "communicator"
    action = state.get("next_action", "")
    return action if action in _ACTIONS else "communicator"


# ── rag ──────────────────────────────────────────────────────────────────
def rag_node(state: dict) -> dict:
    t0 = time.time()
    knowledge, grounding = get_retriever().knowledge_block(state["query"])
    weak = grounding < V6Config.RAG_LOW_CONF
    entry = {"action": "rag", "ok": True,
             "error_type": "rag_weak" if weak else "none",
             "row_bucket": "none", "attempt": _attempt(state, "rag")}
    return {
        "knowledge": knowledge,
        "grounding": grounding,
        "step_log": _step(state, entry),
        "thoughts": _thoughts(state, {"kind": "thinking",
            "text": f"Pulled reference knowledge (grounding {grounding:.2f})."}),
        "trace": _trace(state, f"rag: grounding={grounding:.3f}"),
        "timings": _timing(state, "rag_ms", (time.time() - t0) * 1000),
    }


# ── sql (the full router → generate → execute chain) ─────────────────────
def run_sql_pipeline(state: dict) -> tuple[dict, list[dict]]:
    """The body of the `sql` action. Reuses every v6 SQL function unchanged;
    only the wiring lives here. The generate⇄validate retry is the micro-
    retry; the brain owns the macro-retry by re-picking the `sql` action.
    Returns (state-updates, thoughts)."""
    schema = get_db_schema()
    thoughts: list[dict] = []
    query = state["query"]
    thread = state.get("thread_id", "default")

    # phase 1 — router (SLM), then deterministic schema validation
    history = _history_text(state.get("turns", []),
                            state.get("memory_summary", ""))
    messages = build_router_messages(
        query, schema.prompt(), state.get("knowledge", ""),
        history, feedback=state.get("feedback", ""))
    rr = get_slm().run_router(messages, thread_id=thread)
    routing = parse_router_output(rr["router_output"])
    decision = assemble(query, routing, [], False,
                        state.get("grounding", 0.0),
                        state.get("turns", []), schema)
    routing = decision["routing"]
    thoughts.append({"kind": "thinking",
                     "text": f"Mapped it to tables: "
                             f"{routing.get('tables') or '—'}."})

    # low confidence — no resolvable metric table → ask the user to refine
    if decision["confidence"] == "low":
        return ({
            "router_raw": rr["router_output"], "routing": routing,
            "entities": {}, "sql": "", "sql_valid": False, "sql_issues": [],
            "exec_ok": False, "rows": [], "columns": [], "feedback": "",
            "final_answer": ("I'm not sure which KPI you mean. Could you name "
                             "one — revenue, ARPU, churn, subscribers, EBITDA, "
                             "OPEX/CAPEX — and optionally a wilaya and period?"),
        }, thoughts)

    # resolve entities (deterministic): names → canonical + location_ids
    max_date = schema.date_range[1] if schema.date_range else None
    entities = get_resolver().resolve_all(
        query, routing.get("filters", {}), max_date)
    if entities.get("wilayas"):
        id_map = ", ".join(f"{w}→{i}" for w, i in
                           zip(entities["wilayas"],
                               entities.get("wilaya_ids", [])))
        thoughts.append({"kind": "thinking",
                         "text": f"Resolved wilayas: {id_map}."})

    # phase 2 — generate ⇄ validate (bounded micro-retry)
    sql, sql_valid, sql_issues = "", False, []
    for attempt in range(1, V6Config.SQL_MAX_RETRIES + 2):
        instr = build_sqlgen_instruction(query, routing, entities, schema)
        if attempt > 1 and sql_issues:
            hint = correction_hint(sql_issues, entities)
            if hint:
                thoughts.append({"kind": "thinking",
                                 "text": (f"Retrying SQL — last attempt had "
                                          f"issues: {'; '.join(sql_issues[:2])}.")})
                instr += ("\n\nCORRECTION — your previous attempt was wrong. "
                          + hint)
        res = get_slm().run_sqlgen(thread, instr)
        v = validate_sql(clean_sql(res.get("sql_output", "")), schema)
        sql, sql_valid = v["sql"], v["valid"]
        sql_issues = list(v["errors"]) + consistency_check(sql, entities, query,
                                                           schema=schema)
        if sql_valid and not sql_issues:
            break
    thoughts.append({"kind": "thinking",
                     "text": (f"Built the query: {sql}" if sql
                              else "I couldn't form a valid query.")})

    # execute
    rows, columns, exec_ok, err = [], [], False, None
    if sql_valid and sql:
        final_sql = enforce_limit(sql)
        ex = execute_sql(final_sql)
        rows, columns, exec_ok, err = (ex["rows"], ex["columns"],
                                       ex["ok"], ex["error"])
        sql = final_sql
    thoughts.append({"kind": "thinking",
                     "text": (f"Ran it — {len(rows)} row(s) back." if exec_ok
                              else "The query did not run cleanly.")})

    # compose the data answer
    if not sql:
        answer = ("I couldn't build a valid SQL query for that. "
                  "Could you rephrase it with a clearer KPI?")
    elif not exec_ok:
        answer = f"I built a query but it failed to run ({err}). SQL: {sql}"
    elif not rows:
        answer = "The query ran but returned no matching rows."
    else:
        answer = _summarize_rows(rows, columns)
        caveats = [i for i in sql_issues if "wilaya" in i or "week_start" in i]
        if caveats:
            answer += ("\n(Note: the query may not perfectly match your "
                       "request — " + caveats[0] + ".)")

    errors = list(state.get("errors", []))
    if err:
        errors.append(err)

    return ({
        "router_raw": rr["router_output"], "routing": routing,
        "entities": entities, "sql": sql, "sql_valid": sql_valid,
        "sql_issues": sql_issues, "rows": rows, "columns": columns,
        "exec_ok": exec_ok, "errors": errors, "final_answer": answer,
        "feedback": (f"the generated SQL failed ({err})" if err else ""),
    }, thoughts)


def sql_node(state: dict) -> dict:
    t0 = time.time()
    attempt = _attempt(state, "sql")
    updates, thoughts = run_sql_pipeline(state)

    rows = updates.get("rows", [])
    if not updates.get("sql"):
        error_type, ok = "sql_no_query", False
    elif not updates.get("exec_ok"):
        error_type, ok = "sql_error", False
    elif not rows:
        error_type, ok = "sql_no_rows", True
    else:
        error_type, ok = "none", True
    entry = {"action": "sql", "ok": ok, "error_type": error_type,
             "row_bucket": (row_bucket(len(rows)) if updates.get("exec_ok")
                            else "none"),
             "attempt": attempt}

    out = dict(updates)
    out["step_log"] = _step(state, entry)
    out["thoughts"] = _thoughts(state, *thoughts)
    out["trace"] = _trace(state, f"sql (attempt {attempt}): ok={ok} "
                                 f"rows={len(rows)} type={error_type}")
    out["timings"] = _timing(state, f"sql{attempt}_ms",
                             (time.time() - t0) * 1000)
    return out


# ── capabilities ─────────────────────────────────────────────────────────
def chart_node(state: dict) -> dict:
    t0 = time.time()
    rows, cols = state.get("rows", []), state.get("columns", [])
    ok, path = False, ""
    if state.get("exec_ok") and rows:
        res = make_chart(rows, cols, state["query"])
        ok, path = res["ok"], res.get("path", "")
    entry = {"action": "chart", "ok": ok,
             "error_type": "none" if ok else "artifact_failed",
             "row_bucket": "none", "attempt": _attempt(state, "chart")}
    out = {
        "step_log": _step(state, entry),
        "thoughts": _thoughts(state, {"kind": "thinking",
            "text": "Chart saved." if ok else "I couldn't chart this result."}),
        "trace": _trace(state, f"chart: ok={ok}"),
        "timings": _timing(state, "chart_ms", (time.time() - t0) * 1000),
    }
    if ok:
        out["chart_path"] = path
    return out


def email_node(state: dict) -> dict:
    t0 = time.time()
    draft = compose_email_draft(
        state["query"], state.get("final_answer", ""),
        state.get("rows", []), state.get("columns", []))
    ok = draft.get("status") == "draft"
    entry = {"action": "email", "ok": ok,
             "error_type": "none" if ok else "email_no_recipient",
             "row_bucket": "none", "attempt": _attempt(state, "email")}
    return {
        "email_draft": draft,
        "step_log": _step(state, entry),
        "thoughts": _thoughts(state, {"kind": "thinking",
            "text": (f"Drafted an email to {draft.get('to_name')}." if ok
                     else "Drafted an email, but no recipient was named.")}),
        "trace": _trace(state, f"email: {draft.get('status')}"),
        "timings": _timing(state, "email_ms", (time.time() - t0) * 1000),
    }


def template_node(state: dict) -> dict:
    t0 = time.time()
    res = fill_report(state["query"], state.get("rows", []),
                      state.get("columns", []), state.get("final_answer", ""),
                      state.get("entities", {}))
    ok = res["ok"]
    entry = {"action": "template", "ok": ok,
             "error_type": "none" if ok else "artifact_failed",
             "row_bucket": "none", "attempt": _attempt(state, "template")}
    out = {
        "step_log": _step(state, entry),
        "thoughts": _thoughts(state, {"kind": "thinking",
            "text": "Report saved." if ok else "I couldn't fill the report."}),
        "trace": _trace(state, f"template: ok={ok}"),
        "timings": _timing(state, "template_ms", (time.time() - t0) * 1000),
    }
    if ok:
        out["document_path"] = res["path"]
    return out


# ── communicator (terminal) ──────────────────────────────────────────────
_GREETING_TEXT = (
    "Hello! I'm LatentMind V6 — ask me about telecom KPIs (revenue, ARPU, "
    "churn, subscribers, OPEX, CAPEX, profit) for any Algerian wilaya. I can "
    "chart a result, draft an email about it, or fill a report — just ask.")
_META_TEXT = (
    "I'm LatentMind V6, an analytics agent over the interndb telecom "
    "database. I turn a question into SQL, run it, and report the numbers — "
    "and on request I chart the result, draft an email, or fill a report. I "
    "remember the conversation, so follow-ups like 'and for Oran?' work.")
_UNANSWERABLE_TEXT = (
    "I can't answer that — it needs a KPI or table that isn't in the "
    "database. Try revenue, ARPU, churn, subscribers, EBITDA, or OPEX/CAPEX "
    "for an Algerian wilaya.")


def communicator_node(state: dict) -> dict:
    """Compose the final answer from whatever the loop produced; roll memory."""
    t0 = time.time()
    intent = state.get("intent", "data")
    answer = state.get("final_answer", "")

    if intent == "greeting":
        answer = _GREETING_TEXT
    elif intent == "meta":
        answer = _META_TEXT
    elif intent == "definition":
        answer = get_retriever().definition_for(state["query"]) or (
            "I don't have a stored definition for that term. Try ARPU, churn "
            "rate, active base, total revenue, EBITDA, OPEX or CAPEX.")
    elif intent == "unanswerable":
        answer = _UNANSWERABLE_TEXT
    elif not answer:
        answer = ("I wasn't able to pull the data for that. Could you "
                  "rephrase it with a clearer KPI, wilaya and period?")

    # capability notes
    notes: list[str] = []
    if state.get("chart_path"):
        notes.append(f"📊 Chart saved: {state['chart_path']}")
    if state.get("document_path"):
        notes.append(f"📄 Report saved: {state['document_path']}")
    draft = state.get("email_draft")
    if draft:
        if draft.get("status") == "draft":
            notes.append(f"📧 Email drafted to {draft['to_name']} "
                         f"<{draft['to']}> — subject \"{draft['subject']}\". "
                         f"Not sent; review then send.")
        else:
            notes.append("📧 Email drafted, but no recipient was named — "
                         "pick one from the contacts list.")
    if notes:
        answer = answer + "\n\n" + "\n".join(notes)

    # roll conversation memory: keep the last 2 raw turns, compact the rest
    routing = state.get("routing", {}) or {}
    turn = {
        "query": state["query"], "intent": intent,
        "answer": state.get("final_answer", "") or answer,
        "sql": state.get("sql", ""),
        "tables": routing.get("tables", []),
        "columns": routing.get("columns", []),
    }
    turns_all = list(state.get("turns", [])) + [turn]
    memory_summary = state.get("memory_summary", "")
    if len(turns_all) > _MAX_RAW_TURNS:
        new_compact = _compact_turns(turns_all[:len(turns_all) - _MAX_RAW_TURNS])
        if new_compact:
            memory_summary = (memory_summary + "\n---\n" + new_compact
                              if memory_summary else new_compact)
            memory_summary = memory_summary[-_MAX_COMPACT_CHARS:]
        turns_final = turns_all[-_MAX_RAW_TURNS:]
    else:
        turns_final = turns_all

    out = {
        "final_answer": answer,
        "turns": turns_final,
        "memory_summary": memory_summary,
        "thoughts": _thoughts(state, {"kind": "answer", "text": answer}),
        "trace": _trace(state, "communicator: answer ready, memory updated"),
        "timings": _timing(state, "communicator_ms", (time.time() - t0) * 1000),
    }
    if intent == "data" and state.get("exec_ok"):
        out["last_rows"] = state.get("rows", [])
        out["last_columns"] = state.get("columns", [])
    return out
