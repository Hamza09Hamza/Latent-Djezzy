"""v6/nodes.py — The LangGraph node functions.

Every node takes the shared AgentState and returns a partial update.

Flow of decisions:
  plan_node          — the latent planner decides intent + capabilities.
  router_node        — the SLM maps a data query onto schema tables/columns.
  orchestrator_node  — deterministic: validates that mapping, assembles the
                       plan, or routes to `clarify`.
  the capability nodes (visualize / template / email) self-skip unless their
  tag is in `plan` — the plan, set once, is the only thing that decides what
  runs. On a failed execution `route_after_answer` loops back through
  `replan` so the system gets a second, feedback-informed attempt.
"""

from __future__ import annotations
import time

from .capabilities import compose_email_draft, fill_report, make_chart
from .config import V6Config
from .entities import get_resolver
from .knowledge import get_retriever
from .orchestrator import assemble
from .planner import get_planner
from .prompts import (build_router_messages, build_sqlgen_instruction,
                      parse_router_output)
from .schema import get_db_schema
from .slm import get_slm
from .sql_tools import (clean_sql, consistency_check, correction_hint,
                        enforce_limit, execute_sql, validate_sql)


# ── small helpers ────────────────────────────────────────────────────────
def _trace(state: dict, *msgs: str) -> list[str]:
    return list(state.get("trace", [])) + list(msgs)


def _timing(state: dict, key: str, ms: float) -> dict:
    t = dict(state.get("timings", {}))
    t[key] = round(ms, 1)
    return t


def _history_text(turns: list[dict]) -> str:
    if not turns:
        return ""
    lines: list[str] = []
    for i, t in enumerate(turns[-V6Config.MAX_TURNS:], 1):
        lines.append(f"{i}. Q: {t.get('query', '')}")
        if t.get("intent") == "data" and t.get("answer"):
            lines.append(f"   A: {t.get('final_answer', '')[:140]}")
    return "\n".join(lines)


def _summarize_rows(rows: list[dict], cols: list[str]) -> str:
    n = len(rows)
    if n == 1:
        return "Result: " + ", ".join(f"{c} = {rows[0].get(c)}" for c in cols)
    head = rows[:8]
    body = "\n".join(
        "  " + ", ".join(f"{c}={r.get(c)}" for c in cols) for r in head)
    more = f"\n  ... ({n - 8} more rows)" if n > 8 else ""
    return f"{n} rows returned:\n{body}{more}"


# ── 1. plan (the latent planner — decides intent + capabilities) ─────────
def plan_node(state: dict) -> dict:
    t0 = time.time()
    turns = state.get("turns", [])
    decision = get_planner().plan(state["query"], _history_text(turns))
    intent = decision.intent

    # a terse follow-up ("and for Oran?") continues the prior turn's thread —
    # a memory-continuation rule, not query classification
    note = ""
    if (decision.followup and turns
            and turns[-1].get("intent") == "data" and intent != "data"):
        note = f" [follow-up → data, was {intent}]"
        intent = "data"

    is_direct = intent in ("greeting", "meta", "definition", "unanswerable")
    # capabilities only apply to data queries — zero them out for direct answers
    caps = decision.capabilities if not is_direct else []
    return {
        "intent": intent,
        "capabilities": caps,
        "exec_plan": ["direct_answer"] if is_direct else ["data"],
        "plan_scores": {
            "mode": decision.mode, "score": decision.intent_score,
            "margin": decision.intent_margin, "all": decision.intent_scores,
            "caps": decision.cap_scores, "followup": decision.followup,
        },
        "trace": _trace(state, f"plan: intent={intent} "
                               f"({decision.intent_score}) "
                               f"caps={decision.capabilities} "
                               f"mode={decision.mode}"
                               + (" [followup]" if decision.followup else "")
                               + note),
        "timings": _timing(state, "plan_ms", (time.time() - t0) * 1000),
    }


# ── 2. retrieve (RAG — data path only) ───────────────────────────────────
def retrieve_node(state: dict) -> dict:
    t0 = time.time()
    knowledge, grounding = get_retriever().knowledge_block(state["query"])
    return {
        "knowledge": knowledge,
        "grounding": grounding,
        "trace": _trace(state, f"retrieve: grounding={grounding:.3f}"),
        "timings": _timing(state, "retrieve_ms", (time.time() - t0) * 1000),
    }


# ── 3. router (SLM phase 1 — schema mapping) ─────────────────────────────
def router_node(state: dict) -> dict:
    schema = get_db_schema()
    history = _history_text(state.get("turns", []))
    messages = build_router_messages(
        state["query"], schema.prompt(), state.get("knowledge", ""),
        history, feedback=state.get("feedback", ""))
    res = get_slm().run_router(messages, thread_id=state["thread_id"])
    routing = parse_router_output(res["router_output"])
    return {
        "router_raw": res["router_output"],
        "routing": routing,
        "trace": _trace(state, f"router: tables={routing.get('tables')} "
                               f"parse_ok={routing.get('_parse_ok')}"),
        "timings": _timing(state, "router_ms", res["router_ms"]),
    }


# ── 4. orchestrator (deterministic validation + plan assembly) ───────────
def orchestrator_node(state: dict) -> dict:
    schema = get_db_schema()
    scores = state.get("plan_scores", {})
    decision = assemble(
        state["query"], state.get("routing", {}),
        state.get("capabilities", []), scores.get("followup", False),
        state.get("grounding", 0.0), state.get("turns", []), schema)
    return {
        "routing": decision["routing"],
        "capabilities": decision["capabilities"],
        "exec_plan": decision["plan"],
        "confidence": decision["confidence"],
        "trace": _trace(state, *(f"orchestrator: {m}"
                                 for m in decision["trace"])),
    }


# ── 5. resolve entities (deterministic) ──────────────────────────────────
def resolve_entities_node(state: dict) -> dict:
    schema = get_db_schema()
    max_date = schema.date_range[1] if schema.date_range else None
    entities = get_resolver().resolve_all(
        state["query"], state.get("routing", {}).get("filters", {}), max_date)
    msg = f"resolve: wilayas={entities['wilayas']}"
    if entities.get("segment"):
        msg += f" segment={entities['segment']}"
    if entities.get("time_range"):
        msg += " time=" + entities["time_range"]["start"]
    if entities.get("unresolved_wilayas"):
        msg += f" UNRESOLVED={entities['unresolved_wilayas']}"
    return {"entities": entities, "trace": _trace(state, msg)}


# ── 6. SQL generate (SLM phase 2, reuses the router KV cache) ────────────
def sql_generate_node(state: dict) -> dict:
    schema = get_db_schema()
    attempts = state.get("sql_attempts", 0)
    instr = build_sqlgen_instruction(
        state["query"], state.get("routing", {}),
        state.get("entities", {}), schema)
    if attempts >= 1 and state.get("sql_issues"):
        hint = correction_hint(state["sql_issues"], state.get("entities", {}))
        if hint:
            instr += "\n\nCORRECTION — your previous attempt was wrong. " + hint
    res = get_slm().run_sqlgen(state["thread_id"], instr)
    sql = clean_sql(res.get("sql_output", ""))
    return {
        "sql": sql,
        "sql_attempts": attempts + 1,
        "trace": _trace(state, f"sql_generate (attempt {attempts + 1}, "
                               f"kv_reused={res.get('kv_reused')}): "
                               f"{sql[:90]}"),
        "timings": _timing(state, "sqlgen_ms", res.get("sqlgen_ms", 0.0)),
    }


# ── 7. SQL validate (static safety + intent consistency) ─────────────────
def sql_validate_node(state: dict) -> dict:
    schema = get_db_schema()
    v = validate_sql(state.get("sql", ""), schema)
    issues = consistency_check(
        v["sql"], state.get("entities", {}), state["query"])
    all_issues = list(v["errors"]) + list(issues)
    return {
        "sql": v["sql"],
        "sql_valid": v["valid"],
        "sql_issues": all_issues,
        "trace": _trace(state, f"sql_validate: valid={v['valid']} "
                               f"issues={all_issues or 'none'}"),
    }


# ── 8. SQL execute ───────────────────────────────────────────────────────
def sql_execute_node(state: dict) -> dict:
    t0 = time.time()
    sql = enforce_limit(state.get("sql", ""))
    res = execute_sql(sql)
    errors = list(state.get("errors", []))
    if res["error"]:
        errors.append(res["error"])
    return {
        "sql": sql,
        "rows": res["rows"],
        "columns": res["columns"],
        "exec_ok": res["ok"],
        "errors": errors,
        "trace": _trace(state, f"sql_execute: ok={res['ok']} "
                               f"rows={len(res['rows'])}"),
        "timings": _timing(state, "exec_ms", (time.time() - t0) * 1000),
    }


# ── 9. answer (compose the base data answer) ─────────────────────────────
def answer_node(state: dict) -> dict:
    rows, cols = state.get("rows", []), state.get("columns", [])
    if not state.get("sql"):
        ans = ("I couldn't build a valid SQL query for that question. "
               "Could you rephrase it with a clearer KPI?")
    elif not state.get("exec_ok"):
        err = (state.get("errors") or ["unknown error"])[-1]
        ans = (f"I built a query but it failed to run ({err}). "
               f"SQL: {state.get('sql', '')}")
    elif not rows:
        ans = "The query ran successfully but returned no matching rows."
    else:
        ans = _summarize_rows(rows, cols)
        caveats = [i for i in state.get("sql_issues", [])
                   if "wilaya" in i or "week_start" in i]
        if caveats:
            ans += ("\n(Note: the query may not perfectly match your "
                    "request — " + caveats[0] + ".)")
    return {"final_answer": ans, "trace": _trace(state, "answer: composed")}


# ── 10. replan (loop back after a failed execution) ──────────────────────
def replan_node(state: dict) -> dict:
    count = state.get("replan_count", 0) + 1
    err = (state.get("errors") or ["the query did not run"])[-1]
    return {
        "replan_count": count,
        "feedback": f"the generated SQL failed ({err})",
        "sql_attempts": 0,
        "sql_issues": [],
        "trace": _trace(state, f"replan #{count}: re-routing after failure"),
    }


# ── 11. visualize (self-skips unless 'viz' in plan) ──────────────────────
def visualize_node(state: dict) -> dict:
    if "viz" not in state.get("exec_plan", []):
        return {}
    if not state.get("exec_ok") or not state.get("rows"):
        return {"trace": _trace(state, "visualize: skipped (no data)")}
    res = make_chart(state["rows"], state["columns"], state["query"])
    if res["ok"]:
        return {"chart_path": res["path"],
                "trace": _trace(state, f"visualize: {res['chart_type']} "
                                       f"chart → {res['path']}")}
    return {"trace": _trace(state, f"visualize: failed ({res['error']})")}


# ── 12. template (self-skips unless 'template' in plan) ──────────────────
def template_node(state: dict) -> dict:
    if "template" not in state.get("exec_plan", []):
        return {}
    if not state.get("exec_ok"):
        return {"trace": _trace(state, "template: skipped (no data)")}
    res = fill_report(state["query"], state.get("rows", []),
                      state.get("columns", []), state.get("final_answer", ""),
                      state.get("entities", {}))
    if res["ok"]:
        return {"document_path": res["path"],
                "trace": _trace(state, f"template: report → {res['path']}")}
    return {"trace": _trace(state, f"template: failed ({res['error']})")}


# ── 13. email (self-skips unless 'email' in plan) — DRAFT ONLY ───────────
def email_node(state: dict) -> dict:
    if "email" not in state.get("exec_plan", []):
        return {}
    draft = compose_email_draft(
        state["query"], state.get("final_answer", ""),
        state.get("rows", []), state.get("columns", []))
    return {"email_draft": draft,
            "trace": _trace(state, f"email: {draft['status']} "
                                   f"(to={draft.get('to_name')})")}


# ── 14. direct answer (greeting / meta / definition / unanswerable) ──────
def direct_answer_node(state: dict) -> dict:
    intent = state.get("intent", "greeting")
    if intent == "greeting":
        ans = ("Hello! I'm LatentMind V6 — ask me about telecom KPIs "
               "(revenue, ARPU, churn, subscribers, OPEX, CAPEX, profit) "
               "for any Algerian wilaya. I can also chart a result, draft an "
               "email about it, or fill a report template.")
    elif intent == "meta":
        ans = ("I'm LatentMind V6, an analytics agent over the interndb "
               "telecom database. I turn a question into SQL, run it, and "
               "report the numbers — and on request I chart the result, "
               "draft an email, or fill a report. I remember the "
               "conversation, so follow-ups like 'and for Oran?' work.")
    elif intent == "definition":
        ans = get_retriever().definition_for(state["query"]) or (
            "I don't have a stored definition for that term. Try ARPU, "
            "churn rate, active base, total revenue, EBITDA, OPEX or CAPEX.")
    else:  # unanswerable
        ans = ("I can't answer that — it needs a KPI or table that isn't in "
               "the database. Try revenue, ARPU, churn, subscribers, EBITDA, "
               "or OPEX/CAPEX for an Algerian wilaya.")
    return {"final_answer": ans, "trace": _trace(state, f"direct_answer ({intent})")}


# ── 15. clarify (low-confidence data query) ──────────────────────────────
def clarify_node(state: dict) -> dict:
    unresolved = (state.get("entities", {}) or {}).get(
        "unresolved_wilayas", [])
    extra = (f" I also didn't recognise: {', '.join(map(str, unresolved))}."
             if unresolved else "")
    ans = ("I'm not sure which metric you mean." + extra
           + " Could you name a KPI — revenue, ARPU, churn, subscribers, "
             "EBITDA, OPEX/CAPEX — and optionally a wilaya and time period?")
    return {"final_answer": ans, "trace": _trace(state, "clarify: asked to rephrase")}


# ── 16. finalize (append capability notes, update memory) ────────────────
def finalize_node(state: dict) -> dict:
    answer = state.get("final_answer", "")
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

    routing = state.get("routing", {}) or {}
    turn = {
        "query": state["query"],
        "intent": state.get("intent", ""),
        "answer": state.get("final_answer", ""),
        "sql": state.get("sql", ""),
        "tables": routing.get("tables", []),
        "columns": routing.get("columns", []),
    }
    turns = (list(state.get("turns", [])) + [turn])[-V6Config.MAX_TURNS:]

    out = {
        "final_answer": answer,
        "turns": turns,
        "trace": _trace(state, "finalize: answer ready, memory updated"),
    }
    if state.get("intent") == "data" and state.get("exec_ok"):
        out["last_rows"] = state.get("rows", [])
        out["last_columns"] = state.get("columns", [])
    return out


# ── conditional-edge routers ─────────────────────────────────────────────
def route_after_plan(state: dict) -> str:
    return ("direct_answer"
            if (state.get("exec_plan") or [""])[0] == "direct_answer"
            else "retrieve")


def route_after_orchestrator(state: dict) -> str:
    return ("clarify" if (state.get("exec_plan") or [""])[0] == "clarify"
            else "resolve_entities")


def route_after_validate(state: dict) -> str:
    valid = state.get("sql_valid", False)
    needs_fix = (not valid) or bool(state.get("sql_issues"))
    if needs_fix and state.get("sql_attempts", 0) <= V6Config.SQL_MAX_RETRIES:
        return "sql_generate"
    return "sql_execute" if valid else "answer"


def route_after_answer(state: dict) -> str:
    failed = (state.get("intent") == "data" and bool(state.get("sql"))
              and not state.get("exec_ok"))
    if failed and state.get("replan_count", 0) < V6Config.MAX_REPLAN:
        return "replan"
    return "visualize"
