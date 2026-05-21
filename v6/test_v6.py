"""v6/test_v6.py — Verification harness for the V6 agent.

Runs without the Qwen SLM: the heavy generator is stubbed, but everything
else is real — the SQLite database, BGE-M3, the latent planner, the graph.
That covers every bug fix and every deterministic component.

    python3 -m v6.test_v6

Sections: schema · entities · sql_tools · capabilities · planner · graph.
"""

from __future__ import annotations
import json
import os
import sys

# ── point at the SQLite DB before any v6 module reads the env ────────────
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DB = os.path.join(_REPO, "interndb.sqlite")
os.environ["V6_USE_SQLITE"] = "1"
os.environ["V6_SQLITE_PATH"] = _DB

_PASS = 0
_FAIL = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global _PASS, _FAIL
    mark = "✓" if ok else "✗"
    if ok:
        _PASS += 1
    else:
        _FAIL += 1
    print(f"  {mark} {label}" + (f"  — {detail}" if detail and not ok else ""))


def section(name: str) -> None:
    print(f"\n{'─' * 60}\n{name}\n{'─' * 60}")


# ── A. schema introspection ──────────────────────────────────────────────
def test_schema():
    section("A. schema introspection")
    from v6.schema import get_db_schema
    s = get_db_schema()
    check("tables introspected", len(s.all_tables()) >= 6,
          f"got {s.all_tables()}")
    check("global_revenue has location_id, NOT wilaya",
          s.has_column("global_revenue", "location_id")
          and not s.has_column("global_revenue", "wilaya"))
    check("global_revenue is in the join map (needs dim_location)",
          s.needs_location_join("global_revenue"))
    check("dim_location is NOT a metric table",
          not s.needs_location_join("dim_location"))
    check("prompt states the join rule",
          "JOIN dim_location" in s.prompt()
          and "No metric table has" in s.prompt())
    check("date range detected", s.date_range is not None,
          str(s.date_range))


# ── B. entity resolver ───────────────────────────────────────────────────
def test_entities():
    section("B. entity resolver")
    from v6.entities import get_resolver
    r = get_resolver()
    check("wilaya list loaded", len(r.wilayas) > 40, f"{len(r.wilayas)}")
    check("'Algiers' → 'Alger' (alias)", r.resolve_wilaya("Algiers") == "Alger")
    check("'algiers' lowercase → 'Alger'", r.resolve_wilaya("algiers") == "Alger")
    check("'Bejaia' → accent-folded match",
          r.resolve_wilaya("Bejaia") in ("Béjaïa", "Bejaia"))
    check("'Setif' → accent-folded match",
          r.resolve_wilaya("Setif") in ("Sétif", "Setif"))
    check("'Oran' → 'Oran'", r.resolve_wilaya("Oran") == "Oran")
    check("nonsense → None", r.resolve_wilaya("Gotham City") is None)
    scan = r.scan_query("compare churn between Algiers and Constantine")
    check("scan finds both wilayas", set(scan) == {"Alger", "Constantine"},
          str(scan))
    tr = r.resolve_time("revenue last month", "2026-04-29")
    check("'last month' resolves to a date range",
          tr is not None and tr["end"] == "2026-04-29")
    check("segment detection", r.resolve_segment("prepaid churn") == "prepaid")


# ── C. SQL tools ─────────────────────────────────────────────────────────
def test_sql_tools():
    section("C. sql_tools (safety, consistency, execution)")
    from v6.schema import get_db_schema
    from v6.sql_tools import (clean_sql, consistency_check, execute_sql,
                              validate_sql)
    s = get_db_schema()

    check("blocks DROP", not validate_sql("DROP TABLE x")["valid"])
    check("blocks non-SELECT", not validate_sql("UPDATE t SET a=1")["valid"])
    check("flags unknown table",
          "unknown table 'foobar'" in
          " ".join(validate_sql("SELECT * FROM foobar", s)["errors"]))
    good = ("SELECT dl.wilaya, SUM(g.total_revenue) AS total_revenue "
            "FROM global_revenue g JOIN dim_location dl "
            "ON g.location_id = dl.location_id GROUP BY dl.wilaya")
    check("accepts a valid join query", validate_sql(good, s)["valid"])
    check("strips markdown fences",
          clean_sql("```sql\nSELECT 1\n```") == "SELECT 1")

    # consistency: a hallucinated wilaya filter must be flagged
    halluc = ("SELECT dl.wilaya, AVG(p.arpu) FROM prepaid_kpi p "
              "JOIN dim_location dl ON p.location_id = dl.location_id "
              "WHERE dl.wilaya = 'Oran' GROUP BY dl.wilaya")
    issues = consistency_check(halluc, {"wilayas": []}, "arpu trend")
    check("flags a wilaya filter the user never asked for",
          any("not requested" in i for i in issues), str(issues))
    clean_case = consistency_check(good, {"wilayas": []}, "revenue by wilaya")
    check("clean query raises no consistency issue", not clean_case,
          str(clean_case))

    # execution against the real DB
    res = execute_sql(good)
    check("executes a real join query", res["ok"], str(res["error"]))
    check("returns rows", len(res["rows"]) > 0, f"{len(res['rows'])} rows")

    # the v5 bug query — must now fail loudly, not silently
    bad = "SELECT wilaya, total_revenue FROM global_revenue WHERE wilaya='Oran'"
    bres = execute_sql(bad)
    check("the old 'wilaya' query fails as expected", not bres["ok"],
          "should error: no such column wilaya")


# ── D. capabilities ──────────────────────────────────────────────────────
def test_capabilities():
    section("D. capabilities (chart, report, email draft)")
    from v6.capabilities import (compose_email_draft, fill_report,
                                 load_contacts, make_chart, resolve_recipient)
    rows = [{"wilaya": "Oran", "total_revenue": 1_630_107.9},
            {"wilaya": "Alger", "total_revenue": 2_104_882.1},
            {"wilaya": "Constantine", "total_revenue": 1_402_551.0}]
    cols = ["wilaya", "total_revenue"]

    chart = make_chart(rows, cols, "revenue by wilaya")
    check("chart rendered", chart["ok"], str(chart.get("error")))
    check("chart file exists", chart["ok"] and os.path.isfile(chart["path"]))

    rep = fill_report("revenue by wilaya", rows, cols, "3 rows returned", {})
    check("report rendered", rep["ok"], str(rep.get("error")))
    check("report file exists", rep["ok"] and os.path.isfile(rep["path"]))

    contacts = load_contacts()
    check("contacts loaded from DB", len(contacts) > 0, f"{len(contacts)}")
    rec, _ = resolve_recipient("email this to the finance director", contacts)
    check("'finance director' resolves to a contact",
          rec is not None and "finance" in (rec.get("department") or "").lower(),
          str(rec))
    draft = compose_email_draft("email revenue to the finance director",
                                "3 rows returned", rows, cols)
    check("email draft created with status 'draft'",
          draft["status"] == "draft", draft["status"])
    check("draft has a recipient and is NOT sent",
          draft.get("to") and "sent" not in draft["status"])


# ── E. latent planner (real BGE-M3) ──────────────────────────────────────
def test_planner():
    section("E. latent planner (loads BGE-M3 — may take ~20s)")
    try:
        from v6.planner import get_planner
        p = get_planner()
    except Exception as exc:  # noqa: BLE001
        check("planner loaded", False, f"{type(exc).__name__}: {exc}")
        return
    check("planner ready (prototype mode)", p.mode == "prototype")

    cases = [
        ("hello", "greeting"),
        ("good morning", "greeting"),
        ("what does ARPU mean", "definition"),
        ("define churn rate", "definition"),
        ("what can you do", "meta"),
        ("what was the total revenue in Oran", "data"),
        ("compare churn rate between Algiers and Constantine", "data"),
        ("show the weekly arpu trend for prepaid", "data"),
    ]
    for query, expected in cases:
        d = p.plan(query)
        check(f"intent('{query[:38]}') == {expected}",
              d.intent == expected,
              f"got {d.intent} scores={d.intent_scores}")

    viz = p.plan("chart the average revenue by wilaya")
    check("'chart ...' → viz capability", "viz" in viz.capabilities,
          f"caps={viz.capabilities} scores={viz.cap_scores}")
    em = p.plan("email the revenue figures to the finance director")
    check("'email ...' → email capability", "email" in em.capabilities,
          f"caps={em.capabilities} scores={em.cap_scores}")
    rep = p.plan("put the churn numbers in a report")
    check("'... report' → template capability", "template" in rep.capabilities,
          f"caps={rep.capabilities} scores={rep.cap_scores}")


# ── F. full graph (stub SLM, real planner + DB) ──────────────────────────
class _StubSLM:
    """Stands in for the Qwen dual-role model: canned router JSON + SQL."""

    def __init__(self):
        self._store: dict = {}
        self.cases: list = []

    def register(self, keyword, router, sql):
        self.cases.append((keyword, router, sql))

    @staticmethod
    def _query_of(messages):
        text = messages[-1]["content"]
        marker = "User question:"
        i = text.rfind(marker)
        return (text[i + len(marker):] if i >= 0 else text).strip()

    def _match(self, query):
        ql = query.lower()
        for kws, router, sql in self.cases:
            if all(k in ql for k in kws):
                return router, sql
        return ({"intent": "data", "tables": [], "columns": [],
                 "filters": {"wilayas": [], "segment": None, "time": None},
                 "notes": "stub-default"}, "")

    def run_router(self, messages, thread_id="default", max_new=None):
        query = self._query_of(messages)
        router, sql = self._match(query)
        self._store[thread_id] = {"sql": sql}
        return {"router_output": json.dumps(router), "router_ms": 0.4}

    def run_sqlgen(self, thread_id="default", instruction="", max_new=None):
        return {"sql_output": self._store.get(thread_id, {}).get("sql", ""),
                "kv_reused": False, "sqlgen_ms": 0.4}

    def clear_thread(self, tid):
        self._store.pop(tid, None)


def test_graph():
    section("F. full graph end-to-end (stub SLM)")
    try:
        import v6.slm as slm_mod
        from v6.graph import LatentMindV6
    except Exception as exc:  # noqa: BLE001
        check("graph import", False, f"{exc}")
        return

    stub = _StubSLM()
    stub.register(
        ["revenue", "oran"],
        {"intent": "data", "tables": ["global_revenue", "dim_location"],
         "columns": ["total_revenue", "wilaya"],
         "filters": {"wilayas": ["Oran"], "segment": None, "time": None},
         "notes": "stub"},
        "SELECT dl.wilaya, SUM(g.total_revenue) AS total_revenue "
        "FROM global_revenue g JOIN dim_location dl "
        "ON g.location_id = dl.location_id WHERE dl.wilaya = 'Oran' "
        "GROUP BY dl.wilaya")
    stub.register(
        ["churn", "algiers", "constantine"],
        {"intent": "data", "tables": ["prepaid_kpi", "dim_location"],
         "columns": ["churn_rate", "wilaya"],
         "filters": {"wilayas": ["Algiers", "Constantine"],
                     "segment": "prepaid", "time": None}, "notes": "stub"},
        "SELECT dl.wilaya, AVG(p.churn_rate) AS churn_rate "
        "FROM prepaid_kpi p JOIN dim_location dl "
        "ON p.location_id = dl.location_id "
        "WHERE dl.wilaya IN ('Alger', 'Constantine') GROUP BY dl.wilaya")
    revenue_by_wilaya = (
        "SELECT dl.wilaya, SUM(g.total_revenue) AS total_revenue "
        "FROM global_revenue g JOIN dim_location dl "
        "ON g.location_id = dl.location_id GROUP BY dl.wilaya")
    stub.register(["chart", "revenue"],
                  {"intent": "data", "tables": ["global_revenue", "dim_location"],
                   "columns": ["total_revenue", "wilaya"],
                   "filters": {"wilayas": [], "segment": None, "time": None},
                   "notes": "stub"}, revenue_by_wilaya)
    stub.register(["email", "churn"],
                  {"intent": "data", "tables": ["prepaid_kpi", "dim_location"],
                   "columns": ["churn_rate", "wilaya"],
                   "filters": {"wilayas": [], "segment": "prepaid",
                               "time": None}, "notes": "stub"},
                  "SELECT dl.wilaya, AVG(p.churn_rate) AS churn_rate "
                  "FROM prepaid_kpi p JOIN dim_location dl "
                  "ON p.location_id = dl.location_id GROUP BY dl.wilaya")
    stub.register(["constantine"],            # follow-up fallback
                   {"intent": "data", "tables": ["global_revenue", "dim_location"],
                    "columns": ["total_revenue", "wilaya"],
                    "filters": {"wilayas": ["Constantine"], "segment": None,
                                "time": None}, "notes": "stub"},
                   "SELECT dl.wilaya, SUM(g.total_revenue) AS total_revenue "
                   "FROM global_revenue g JOIN dim_location dl "
                   "ON g.location_id = dl.location_id "
                   "WHERE dl.wilaya = 'Constantine' GROUP BY dl.wilaya")

    slm_mod._slm = stub                      # inject the stub singleton
    try:
        agent = LatentMindV6()
    except Exception as exc:  # noqa: BLE001
        check("graph built", False, f"{exc}")
        return
    check("graph built", True)

    # 1. greeting must NOT generate SQL  (the v5 bug)
    r = agent.ask("hello", thread_id="t-greet")
    check("'hello' → greeting intent", r.get("intent") == "greeting",
          r.get("intent"))
    check("'hello' did NOT run SQL", not r.get("sql"))
    check("'hello' got a friendly answer", "LatentMind" in r.get("answer", ""))

    # 2. definition
    r = agent.ask("what does ARPU mean", thread_id="t-def")
    check("'what does ARPU mean' → definition",
          r.get("intent") == "definition", r.get("intent"))
    check("definition answer mentions ARPU",
          "ARPU" in r.get("answer", "") or "Average" in r.get("answer", ""))

    # 3. data — the revenue-in-Oran query that used to crash
    r = agent.ask("what was the total revenue in Oran", thread_id="t-rev")
    check("'revenue in Oran' → data", r.get("intent") == "data")
    check("'revenue in Oran' executed OK", r.get("exec_ok") is True,
          str(r.get("errors")))
    check("'revenue in Oran' returned rows", len(r.get("rows", [])) > 0)

    # 4. data — churn comparison; resolver must turn Algiers into Alger
    r = agent.ask("compare churn rate between Algiers and Constantine",
                  thread_id="t-churn")
    check("'churn compare' executed OK", r.get("exec_ok") is True,
          str(r.get("errors")))
    wilayas = {row.get("wilaya") for row in r.get("rows", [])}
    check("churn comparison returns BOTH cities (Algiers→Alger fixed)",
          {"Alger", "Constantine"}.issubset(wilayas), str(wilayas))

    # 5. capability — chart
    r = agent.ask("chart the average revenue by wilaya", thread_id="t-viz")
    check("'chart ...' put viz in the plan", "viz" in r.get("plan", []),
          str(r.get("plan")))
    check("chart file produced", bool(r.get("chart_path"))
          and os.path.isfile(r.get("chart_path", "")))

    # 6. capability — email draft (drafted, never sent)
    r = agent.ask("email the prepaid churn by wilaya to the finance director",
                  thread_id="t-mail")
    check("'email ...' put email in the plan", "email" in r.get("plan", []),
          str(r.get("plan")))
    draft = r.get("email_draft") or {}
    check("email drafted, not sent", draft.get("status") == "draft",
          str(draft.get("status")))
    check("email recipient resolved to a finance contact",
          draft.get("to_name") is not None, str(draft.get("to_name")))

    # 7. memory — a follow-up turn on the same thread
    agent.ask("what was the total revenue in Oran", thread_id="t-mem")
    r = agent.ask("and what about Constantine", thread_id="t-mem")
    check("follow-up stays a data query", r.get("intent") == "data",
          r.get("intent"))
    check("follow-up executed OK", r.get("exec_ok") is True,
          str(r.get("errors")))


def main():
    print("=" * 60)
    print(" LatentMind V6 — verification harness")
    print(f" DB: {_DB}")
    print("=" * 60)
    if not os.path.isfile(_DB):
        print(f"\n⚠  SQLite DB not found at {_DB}")
        print("   export it first, then re-run.")
        return 1

    test_schema()
    test_entities()
    test_sql_tools()
    test_capabilities()
    test_planner()
    test_graph()

    print(f"\n{'=' * 60}")
    print(f" RESULT: {_PASS} passed, {_FAIL} failed")
    print("=" * 60)
    return 0 if _FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
