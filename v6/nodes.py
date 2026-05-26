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
    """Recent raw turns + compacted older memory, for prompts and the brain.

    For each prior data turn we also surface the schema mapping the router
    chose (tables + columns), so a short follow-up like "and for Tiaret?"
    has the columns visible for RULE 5 inheritance in the router prompt.
    Without this the router sees only the Q+A snippet and cannot tell
    which KPI to inherit.
    """
    if not turns and not memory_summary:
        return ""
    lines: list[str] = []
    if memory_summary:
        lines.append(f"[Earlier context]\n{memory_summary}")
    for i, t in enumerate(turns[-limit:], 1):
        lines.append(f"{i}. Q: {t.get('query', '')}")
        if t.get("intent") == "data":
            tables = t.get("tables") or []
            cols = t.get("columns") or []
            if tables:
                lines.append(f"   tables: {tables}")
            if cols:
                lines.append(f"   columns: {cols}")
            if t.get("answer"):
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
    # Non-data intents never need an action — the communicator handles them
    # directly. Guard here so a misfiring action head can't send a greeting
    # or unanswerable query into the SQL pipeline.
    intent = state.get("intent", "")
    if intent in ("greeting", "meta", "definition", "unanswerable"):
        return "communicator"
    if state.get("brain_step", 0) >= V6Config.BRAIN_MAX_STEPS:
        return "communicator"

    action = state.get("next_action", "")
    query_l = state.get("query", "").lower()

    # Strong template bypass: when the user asks for a report AND data is
    # available — either from this turn's SQL (exec_ok) OR the previous
    # turn's persisted last_rows (cross-turn follow-up like "Put it in a
    # report") — skip whatever action the brain chose and go straight to
    # template. Fires before the seuil check so a low continue score can't
    # suppress it.
    _template_kw = {"report", "put it in", "document", "fill",
                    "rapport", "mettre", "generate a report", "rapporter"}
    _template_done = any(s.get("action") == "template" and s.get("ok", False)
                         for s in state.get("step_log", []))
    if (not _template_done
            and any(kw in query_l for kw in _template_kw)
            and (state.get("exec_ok") or state.get("last_rows"))):
        return "template"

    if state.get("continue_score", 0.0) < V6Config.BRAIN_SEUIL:
        return "communicator"
    conf = (state.get("brain_scores", {}) or {}).get("action_conf", 1.0)
    if conf < V6Config.BRAIN_CONF_MIN:
        return "communicator"
    if action not in _ACTIONS:
        return "communicator"

    # Safety: never repeat a terminal action that already succeeded.
    _TERMINAL = {"chart", "email", "template"}
    if action in _TERMINAL:
        step_log = state.get("step_log", [])
        already_done = any(
            s.get("action") == action and s.get("ok", False)
            for s in step_log)
        if already_done:
            return "communicator"

    # Keyword guards: terminal actions need an explicit signal in the query.
    # Prevents the brain from triggering chart/email/template when the user
    # didn't ask for one (a brain misfiring until training improves).
    if action == "chart":
        _chart_kw = {
            "chart", "plot", "draw", "graphique", "graphe", "graph",
            "visualize", "visualise", "trend", "tendance", "evolution",
            "évolution", "courbe", "figure", "show me a", "montre",
        }
        if not any(kw in query_l for kw in _chart_kw):
            return "communicator"
    if action == "email":
        _email_kw = {
            "email", "mail", "send", "envoyer", "envoie", "message",
            "director", "directeur", "manager", "responsable",
        }
        if not any(kw in query_l for kw in _email_kw):
            return "communicator"

    return action


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

    # If the router itself classifies the query as non-data, respect it —
    # the communicator will give the right answer for that intent.
    if routing.get("intent") in ("unanswerable", "greeting", "meta", "definition"):
        return ({
            "router_raw": rr["router_output"], "routing": routing,
            "entities": {}, "sql": "", "sql_valid": False, "sql_issues": [],
            "exec_ok": False, "rows": [], "columns": [], "feedback": "",
            "final_answer": "",  # communicator fills this from intent
        }, thoughts)

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

    # resolve entities (deterministic): names → canonical French spellings
    max_date = schema.date_range[1] if schema.date_range else None
    entities = get_resolver().resolve_all(
        query, routing.get("filters", {}), max_date)
    if entities.get("wilayas"):
        ids_map = entities.get("wilaya_ids_map", {}) or {}
        summary = ", ".join(
            f"{w}={len(ids_map.get(w, []))} communes"
            for w in entities["wilayas"])
        thoughts.append({"kind": "thinking",
                         "text": f"Resolved wilayas: {summary}."})

    # phase 2 — generate ⇄ validate ⇄ execute (bounded micro-retry).
    # Static issues (consistency_check) AND runtime errors from the database
    # both feed the same correction loop — the SLM gets to see what really
    # went wrong, not a generic phrase.
    sql, sql_valid, sql_issues = "", False, []
    exec_error: str | None = None
    rows, columns, exec_ok, err = [], [], False, None

    # Carry-over errors from a previous macro-retry (brain re-picked sql).
    # Inject a strong hint so the model doesn't repeat the same bad column.
    _prior_errors = state.get("errors", [])
    _prior_error_hint = ""
    if _prior_errors:
        _last_err = _prior_errors[-1][:300]
        import re as _re
        _bad_col_m = _re.search(r'no such column[:  ]+(\w+)', _last_err, _re.IGNORECASE)
        if _bad_col_m:
            _bad = _bad_col_m.group(1)
            _prior_error_hint = (
                f"\n\nPRIOR FAILURE: Column `{_bad}` was rejected by the database "
                f"(error: '{_last_err[:120]}'). That column does not exist. "
                f"The MANDATORY COLUMNS above are the ONLY valid names — "
                f"do NOT write `{_bad}` anywhere in your SQL.")
        else:
            _prior_error_hint = (
                f"\n\nPRIOR FAILURE: {_last_err[:150]}. "
                f"Fix this and check MANDATORY COLUMNS carefully.")

    for attempt in range(1, V6Config.SQL_MAX_RETRIES + 2):
        instr = build_sqlgen_instruction(query, routing, entities, schema)
        if attempt == 1 and _prior_error_hint:
            instr += _prior_error_hint
        if attempt > 1 and (sql_issues or exec_error):
            hint = correction_hint(sql_issues, entities,
                                   exec_error=exec_error)
            if hint:
                preview = (exec_error or "; ".join(sql_issues[:2]))[:120]
                thoughts.append({"kind": "thinking",
                                 "text": f"Retrying SQL — last attempt: "
                                         f"{preview}."})
                instr += ("\n\nCORRECTION — your previous attempt was wrong. "
                          + hint)
        res = get_slm().run_sqlgen(thread, instr)
        v = validate_sql(clean_sql(res.get("sql_output", "")), schema)
        sql, sql_valid = v["sql"], v["valid"]
        sql_issues = list(v["errors"]) + consistency_check(
            sql, entities, query, schema=schema)
        exec_error = None
        if not (sql_valid and not sql_issues):
            continue
        # passed static checks → try executing
        final_sql = enforce_limit(sql)
        ex = execute_sql(final_sql)
        sql = final_sql
        if ex["ok"]:
            rows, columns, exec_ok, err = (ex["rows"], ex["columns"],
                                           True, None)
            break
        exec_error = ex["error"]
        err = ex["error"]
    thoughts.append({"kind": "thinking",
                     "text": (f"Built the query: {sql}" if sql
                              else "I couldn't form a valid query.")})
    thoughts.append({"kind": "thinking",
                     "text": (f"Ran it — {len(rows)} row(s) back." if exec_ok
                              else f"The query did not run cleanly"
                              + (f" ({err})." if err else "."))})

    # compose the data answer
    _db_range = schema.date_range  # e.g. ("2025-07-16", "2026-04-29")
    if not sql:
        answer = ("I couldn't build a valid SQL query for that. "
                  "Could you rephrase it with a clearer KPI?")
    elif not exec_ok:
        answer = f"I built a query but it failed to run ({err}). SQL: {sql}"
    elif not rows:
        # 0-row result — most often a period outside the DB coverage window.
        if _db_range:
            answer = (f"No data found for that period — the database covers "
                      f"{_db_range[0]} to {_db_range[1]}. "
                      f"Try asking about a date within this range.")
        else:
            answer = "The query ran but returned no matching rows."
    else:
        # Detect all-NULL aggregate (e.g. SUM over an empty date range)
        _all_null = (len(rows) == 1
                     and all(v is None for v in rows[0].values()))
        if _all_null:
            if _db_range:
                answer = (f"No data found for that period — the database "
                          f"covers {_db_range[0]} to {_db_range[1]}. "
                          f"Try asking about a date within this range.")
            else:
                answer = "No data available for that query."
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
    # Fall back to the previous turn's persisted rows when the current turn
    # didn't run SQL (e.g. "Put it in a report" as a cross-turn follow-up).
    rows = state.get("rows") or state.get("last_rows", [])
    cols = state.get("columns") or state.get("last_columns", [])
    res = fill_report(state["query"], rows, cols, state.get("final_answer", ""),
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
    "Hello! I'm LatentMind V6 — your telecom analytics agent for Algeria. "
    "Ask me about revenue, ARPU, churn, subscribers, OPEX, CAPEX, or "
    "profitability for any wilaya. I can also chart the result, draft an "
    "email, or fill a report.")
_META_TEXT = (
    "I'm LatentMind V6, a telecom analytics agent for the Algerian market. "
    "I query the database, analyze the results, and can chart, email, or "
    "report what I find. Ask me about KPIs, trends, comparisons, or "
    "breakdowns — by wilaya, period, or segment.")
_UNANSWERABLE_TEXT = (
    "That metric isn't in the database. I can answer questions about "
    "revenue, ARPU, churn, subscribers, EBITDA, OPEX, CAPEX, or "
    "profitability — for any Algerian wilaya or time period.")


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
        if state.get("document_path"):
            # Template ran using last_rows from a prior turn — no SQL this turn.
            answer = "Report generated from the previous query result."
        else:
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
