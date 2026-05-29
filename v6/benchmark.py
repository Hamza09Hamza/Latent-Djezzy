"""v6/benchmark.py — Clean, reproducible benchmark for the voice pipeline.

What it measures, and why it is split into "automatic" vs "manual":

  AUTOMATIC (objective — graded here):
    - intent accuracy        : predicted intent vs the labelled intent
    - SQL execution rate      : share of data queries that ran and returned rows
    - latency per stage       : STT · brain · SQL · total, per category
    - STT word-error-rate     : transcription vs the known script text

  MANUAL (the polisher rephrases freely, so correctness can't be auto-graded):
    - the spoken/written answer for each query is captured and laid out in a
      review table; you read it and mark it right or wrong.

The 20 queries (data/bench_queries.json) alternate French and English and
cover every message type. The same text is synthesised to audio (TTS) and
fed back through STT, so the speech round-trip is exercised end to end.

    # text pipeline only (latency + intent + SQL):
    from v6.benchmark import run_text_benchmark
    run_text_benchmark(agent)

    # full voice round-trip (needs GPU + faster-whisper + XTTS):
    from v6.benchmark import generate_audio_fixtures, run_full
    generate_audio_fixtures()          # TTS → wav per query
    run_full(agent, transcribe=True)   # pipeline + STT WER + review table
"""

from __future__ import annotations
import json
import os
import re
import time

from .config import V6Config


# ── query set ──────────────────────────────────────────────────────────────
def load_queries() -> list[dict]:
    with open(V6Config.BENCH_QUERIES_PATH, encoding="utf-8") as f:
        return json.load(f)["queries"]


# ── word-error-rate (for the STT round-trip) ───────────────────────────────
def _normalize(s: str) -> list[str]:
    """Lowercase, drop punctuation, collapse spaces → word list for WER."""
    s = (s or "").lower()
    s = re.sub(r"[^\w\s]", " ", s, flags=re.UNICODE)
    return s.split()


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate = edit distance / reference length, in [0, ∞)."""
    ref, hyp = _normalize(reference), _normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    # Levenshtein over word sequences
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cost = 0 if r == h else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return round(prev[-1] / len(ref), 3)


# ── polishing: turn the raw graph answer into the spoken/written text ───────
# The pipeline's final_answer for data is a raw row dump (_summarize_rows);
# what the USER hears is that dump rewritten by slm.Polisher (the analyst that
# applies the spoken-number rounding rule). graph.invoke() never runs it, so
# without this the review table shows raw figures and the rounding fix is
# invisible. We mirror the notebook's _pick_role routing exactly.
_CLARIFY_MARKERS = (
    "couldn't build", "failed to run", "no matching rows", "wasn't able to pull",
    "no data found", "couldn't match", "not sure which kpi", "isn't in the database",
)


def _spoken_role(row: dict) -> tuple[str | None, str]:
    """(role, text) for the polisher — None means speak the text verbatim.

    Identical logic to the notebook's _pick_role so the benchmark grades the
    same answer the live voice/text path produces.
    """
    intent = row.get("pred_intent") or "data"
    answer = row.get("answer", "") or ""
    exec_ok = bool(row.get("exec_ok"))
    if intent in ("greeting", "meta"):
        # Warm 'chat' persona responds to the utterance, not a canned blurb.
        return "chat", answer
    if intent in ("definition", "unanswerable"):
        return "polish", answer
    if row.get("document_path") and not exec_ok:
        return None, "Report generated from the previous result."
    low = answer.lower()
    if not exec_ok or any(p in low for p in _CLARIFY_MARKERS):
        return "clarify", answer
    # data with rows — strip artifact notes, analyse the figures
    data_part = answer.split("📧")[0].split("📊")[0].split("📄")[0].strip()
    return "analyze", data_part


def _polish_rows(rows: list[dict], enabled: bool = True) -> list[dict]:
    """Attach `spoken` (the polished answer) to every row.

    Degrades gracefully: if the polisher can't load (no GPU / model), `spoken`
    falls back to the raw answer so the rest of the benchmark still runs.
    """
    if not enabled:
        for r in rows:
            r["spoken"] = r.get("answer", "")
        return rows
    try:
        from v6.slm import get_polisher
        pol = get_polisher()
    except Exception as exc:  # noqa: BLE001
        print(f"  (polisher unavailable: {exc} — showing raw answers)")
        for r in rows:
            r["spoken"] = r.get("answer", "")
        return rows
    for r in rows:
        role, text = _spoken_role(r)
        if role is None:
            r["spoken"] = text
            continue
        try:
            r["spoken"] = "".join(pol.stream(text, r["text"], role)).strip()
        except Exception:  # noqa: BLE001 — a single polish failure isn't fatal
            r["spoken"] = text
    return rows


# ── per-stage latency extraction ────────────────────────────────────────────
def _stage_ms(timings: dict) -> dict:
    brain = sum(v for k, v in timings.items() if k.startswith("brain"))
    sql = sum(v for k, v in timings.items() if k.startswith("sql"))
    return {
        "brain_ms": round(brain, 1),
        "rag_ms": round(timings.get("rag_ms", 0.0), 1),
        "sql_ms": round(sql, 1),
        "total_ms": round(timings.get("total_ms", 0.0), 1),
    }


# ── run one query through the text pipeline ─────────────────────────────────
def _run_one(agent, q: dict) -> dict:
    from v6.state import initial_state

    cfg = {"configurable": {"thread_id": q["thread"]}, "recursion_limit": 60}
    t0 = time.time()
    r = agent.graph.invoke(initial_state(q["text"], q["thread"]), cfg)
    elapsed_ms = round((time.time() - t0) * 1000, 1)

    timings = dict(r.get("timings", {}))
    timings.setdefault("total_ms", elapsed_ms)
    actions = [s.get("action") for s in r.get("step_log", [])]
    routing = r.get("routing", {}) or {}

    pred_intent = r.get("intent", "")
    intent_ok = (pred_intent == q["intent"])
    # SQL success only matters for data queries
    exec_ok = bool(r.get("exec_ok"))

    # free the per-thread KV cache so memory stays flat across the run
    try:
        from v6.slm import get_slm
        get_slm().clear_thread(q["thread"])
    except Exception:  # noqa: BLE001
        pass

    return {
        "id": q["id"], "lang": q["lang"], "category": q["category"],
        "text": q["text"], "expects": q.get("expects", ""),
        "label_intent": q["intent"], "pred_intent": pred_intent,
        "intent_ok": intent_ok,
        "exec_ok": exec_ok,
        "actions": actions,
        "tables": routing.get("tables", []),
        "sql": r.get("sql", ""),
        "n_rows": len(r.get("rows", [])),
        "chart_path": r.get("chart_path", ""),
        "document_path": r.get("document_path", ""),
        "email_status": (r.get("email_draft") or {}).get("status", ""),
        "answer": r.get("final_answer", ""),
        "stage_ms": _stage_ms(timings),
    }


# ── audio fixtures (TTS) ─────────────────────────────────────────────────────
def generate_audio_fixtures(out_dir: str | None = None) -> list[dict]:
    """Synthesise each query's text to {out_dir}/bench_<id>.wav via XTTS.

    Uses the language tag in the query to pick the matching voice. Returns
    one dict per file with its path and synthesis timing.
    """
    from v6.speech import get_tts

    out_dir = out_dir or V6Config.audio_dir()
    os.makedirs(out_dir, exist_ok=True)
    tts = get_tts()
    out: list[dict] = []
    for q in load_queries():
        path = os.path.join(out_dir, f"bench_{q['id']}.wav")
        res = tts.synthesize(q["text"], q["lang"], out_path=path)
        out.append({"id": q["id"], "lang": q["lang"], "path": path,
                    "seconds": res["seconds"], "synth_ms": res["ms"],
                    "first_chunk_ms": res["first_chunk_ms"]})
        print(f"  ♪ {q['id']} [{q['lang']}] {res['seconds']:.1f}s "
              f"(synth {res['ms']:.0f}ms, first chunk {res['first_chunk_ms']}ms)")
    return out


def transcribe_fixtures(audio_dir: str | None = None) -> list[dict]:
    """Transcribe each bench_<id>.wav and score WER against the script text."""
    from v6.speech import get_stt

    audio_dir = audio_dir or V6Config.audio_dir()
    stt = get_stt()
    out: list[dict] = []
    for q in load_queries():
        path = os.path.join(audio_dir, f"bench_{q['id']}.wav")
        if not os.path.isfile(path):
            out.append({"id": q["id"], "missing": True})
            continue
        res = stt.transcribe(path, language=q["lang"])
        raw = res.get("raw_text", res["text"])
        out.append({
            "id": q["id"], "lang": q["lang"], "ref": q["text"],
            "hyp": res["text"], "wer": wer(q["text"], res["text"]),
            "raw_hyp": raw, "raw_wer": wer(q["text"], raw),
            "det_lang": res["language"], "stt_ms": res["ms"],
        })
    return out


# ── reporting ────────────────────────────────────────────────────────────────
def _agg(rows: list[dict]) -> dict:
    data_rows = [r for r in rows if r["label_intent"] == "data"]
    intent_ok = sum(r["intent_ok"] for r in rows)
    exec_ok = sum(r["exec_ok"] for r in data_rows)
    avg_total = (sum(r["stage_ms"]["total_ms"] for r in rows) / len(rows)
                 if rows else 0.0)
    return {
        "n": len(rows),
        "intent_acc": round(intent_ok / len(rows), 3) if rows else 0.0,
        "sql_exec_rate": (round(exec_ok / len(data_rows), 3)
                          if data_rows else None),
        "avg_total_ms": round(avg_total, 1),
    }


def print_report(rows: list[dict], stt_rows: list[dict] | None = None) -> None:
    print("\n" + "=" * 78)
    print(" PIPELINE BENCHMARK — objective metrics")
    print("=" * 78)
    hdr = (f"{'id':<5}{'lang':<5}{'category':<16}{'intent':<13}"
           f"{'ok':<4}{'exec':<5}{'brain':>7}{'sql':>8}{'total':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        s = r["stage_ms"]
        intent_mark = "✓" if r["intent_ok"] else f"✗({r['pred_intent']})"
        exec_mark = ("—" if r["label_intent"] != "data"
                     else ("✓" if r["exec_ok"] else "✗"))
        print(f"{r['id']:<5}{r['lang']:<5}{r['category']:<16}"
              f"{r['label_intent']:<13}{intent_mark:<4}{exec_mark:<5}"
              f"{s['brain_ms']:>6.0f}m{s['sql_ms']/1000:>7.2f}s"
              f"{s['total_ms']/1000:>7.2f}s")

    a = _agg(rows)
    print("-" * len(hdr))
    print(f"  intent accuracy : {a['intent_acc']*100:.1f}%  "
          f"({a['n']} queries)")
    if a["sql_exec_rate"] is not None:
        print(f"  SQL exec rate   : {a['sql_exec_rate']*100:.1f}%  "
              f"(data queries only)")
    print(f"  avg latency     : {a['avg_total_ms']/1000:.2f}s per query")

    # per-category latency
    cats: dict[str, list[float]] = {}
    for r in rows:
        cats.setdefault(r["category"], []).append(r["stage_ms"]["total_ms"])
    print("\n  latency by category:")
    for cat, vals in cats.items():
        print(f"    {cat:<20} {sum(vals)/len(vals)/1000:>6.2f}s avg "
              f"({len(vals)})")

    # STT round-trip
    if stt_rows:
        scored = [s for s in stt_rows if not s.get("missing")]
        if scored:
            avg_wer = sum(s["wer"] for s in scored) / len(scored)
            avg_raw = sum(s.get("raw_wer", s["wer"]) for s in scored) / len(scored)
            avg_stt = sum(s["stt_ms"] for s in scored) / len(scored)
            print("\n" + "=" * 78)
            print(" STT ROUND-TRIP — word error rate, raw → corrected (lower is better)")
            print("=" * 78)
            for s in scored:
                fixed = "✎" if s.get("raw_wer", s["wer"]) > s["wer"] else " "
                print(f"  {s['id']:<5}{s['lang']:<4}{fixed} "
                      f"WER {s.get('raw_wer', s['wer']):<5}→{s['wer']:<6} "
                      f"({s['det_lang']}, {s['stt_ms']:.0f}ms)  hyp: {s['hyp'][:44]}")
            print("-" * 78)
            print(f"  avg WER: {avg_raw*100:.1f}% raw → {avg_wer*100:.1f}% "
                  f"corrected   avg STT latency: {avg_stt:.0f}ms")

    # manual-review block
    has_spoken = any("spoken" in r for r in rows)
    print("\n" + "=" * 78)
    if has_spoken:
        print(" ANSWER REVIEW — SPOKEN text (what the user hears, post-polish)")
    else:
        print(" ANSWER REVIEW — raw answers (polisher not run)")
    print("=" * 78)
    for r in rows:
        artifacts = []
        if r["chart_path"]:
            artifacts.append("chart")
        if r["document_path"]:
            artifacts.append("report")
        if r["email_status"]:
            artifacts.append(f"email:{r['email_status']}")
        tag = f"  [{', '.join(artifacts)}]" if artifacts else ""
        print(f"\n[{r['id']}] ({r['lang']}) {r['text']}")
        print(f"   expect : {r['expects']}")
        if has_spoken:
            print(f"   spoken : {r.get('spoken', '')[:300]}{tag}")
            print(f"   (raw)  : {r['answer'][:120]}")
        else:
            print(f"   answer : {r['answer'][:300]}{tag}")


# ── voice-vs-text regression diff ────────────────────────────────────────────
def _artifact_set(row: dict) -> set[str]:
    s: set[str] = set()
    if row.get("chart_path"):
        s.add("chart")
    if row.get("document_path"):
        s.add("report")
    if row.get("email_status") == "draft":
        s.add("email")
    return s


def print_regressions(text_rows: list[dict], voice_rows: list[dict]) -> None:
    """Flag the queries whose voice result diverged from the clean-text result.

    Compares intent, the artifact set (chart/report/email), and SQL exec per
    query id, so the ~4-5 queries STT noise actually broke jump out instead of
    being buried in two 20-row tables. Stable queries are not printed.
    """
    tmap = {r["id"]: r for r in text_rows}
    regs: list[tuple[dict, list[str]]] = []
    for v in voice_rows:
        t = tmap.get(v["id"])
        if not t:
            continue
        changes: list[str] = []
        if t["pred_intent"] != v["pred_intent"]:
            changes.append(f"intent {t['pred_intent']} → {v['pred_intent']}")
        ta, va = _artifact_set(t), _artifact_set(v)
        if ta != va:
            changes.append(f"artifacts {'+'.join(sorted(ta)) or '—'} → "
                           f"{'+'.join(sorted(va)) or '—'}")
        if bool(t["exec_ok"]) != bool(v["exec_ok"]):
            changes.append(f"exec {'ok' if t['exec_ok'] else 'fail'} → "
                           f"{'ok' if v['exec_ok'] else 'fail'}")
        if changes:
            regs.append((v, changes))

    print("\n" + "=" * 78)
    print(" VOICE vs TEXT — REGRESSIONS (only diverging queries shown)")
    print("=" * 78)
    if not regs:
        print(f"  none — all {len(voice_rows)} queries stable under voice ✓")
        return
    for v, changes in regs:
        wer_v = v.get("transcript_wer", 0.0)
        print(f"\n[{v['id']}] ({v['lang']}) WER {wer_v:.2f}")
        if v.get("transcript") and v.get("original_text"):
            print(f"   said  : {v['original_text']}")
            print(f"   heard : {v['transcript']}")
        for c in changes:
            print(f"   ⚠ {c}")
    print("-" * 78)
    print(f"  {len(regs)} of {len(voice_rows)} queries regressed under voice  ·  "
          f"{len(voice_rows) - len(regs)} stable")


def _save(rows: list[dict], stt_rows: list[dict] | None) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(V6Config.output_dir(), f"bench_results_{stamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"pipeline": rows, "stt": stt_rows or [],
                   "summary": _agg(rows)}, f, ensure_ascii=False, indent=2)
    return path


# ── entry points ─────────────────────────────────────────────────────────────
def run_text_benchmark(agent, save: bool = True,
                       polish: bool = True) -> list[dict]:
    """Run the 20 queries through the text pipeline; print + save metrics.

    With `polish=True` the raw graph answer is rewritten by the analyst/
    polisher exactly as the live system does, so the ANSWER REVIEW shows the
    spoken text (and you can verify the number-rounding rule). Set False to
    skip the polisher (faster, raw figures shown).
    """
    rows = []
    for q in load_queries():
        print(f"  · running {q['id']} [{q['lang']}] {q['text'][:50]}")
        rows.append(_run_one(agent, q))
    if polish:
        print("  · polishing answers (analyst rewrite for the review)…")
    _polish_rows(rows, enabled=polish)
    print_report(rows)
    if save:
        print(f"\n  saved → {_save(rows, None)}")
    return rows


def run_full(agent, transcribe: bool = True, save: bool = True,
             polish: bool = True) -> dict:
    """Full benchmark: text pipeline + (optional) STT round-trip on fixtures."""
    rows = [_run_one(agent, q) for q in load_queries()]
    _polish_rows(rows, enabled=polish)
    stt_rows = transcribe_fixtures() if transcribe else None
    print_report(rows, stt_rows)
    if save:
        print(f"\n  saved → {_save(rows, stt_rows)}")
    return {"pipeline": rows, "stt": stt_rows, "summary": _agg(rows)}


def run_voice_benchmark(agent, audio_dir: str | None = None,
                        save: bool = True, polish: bool = True,
                        text_rows: list[dict] | None = None) -> list[dict]:
    """End-to-end voice benchmark: .wav → STT → pipeline → answer.

    This is the realistic test: each audio file is transcribed (with biasing
    + post-correction), the transcript drives the pipeline, and results are
    graded exactly like run_text_benchmark. Comparing the two reveals how
    much STT noise degrades pipeline accuracy.

    Pass `text_rows` (the return of run_text_benchmark) to print a
    voice-vs-text REGRESSIONS table — the queries STT noise actually broke.

    Audio files must already exist — run generate_audio_fixtures() first.
    """
    from v6.speech import get_stt
    from v6.transcribe import correct_transcript

    audio_dir = audio_dir or V6Config.audio_dir()
    stt = get_stt()
    queries = load_queries()
    rows: list[dict] = []

    for q in queries:
        wav = os.path.join(audio_dir, f"bench_{q['id']}.wav")
        if not os.path.isfile(wav):
            print(f"  ✗ {q['id']} — audio missing ({wav}); skipping")
            continue

        # STT → corrected transcript
        res = stt.transcribe(wav, language=q["lang"])
        transcript = res["text"]          # already corrected by transcribe.py
        raw = res.get("raw_text", transcript)
        transcript_wer = wer(q["text"], transcript)

        print(f"  🎙 {q['id']} [{q['lang']}] "
              f"WER {transcript_wer:.2f}  hyp: {transcript[:50]}")

        # Run the pipeline with the STT transcript instead of the clean text
        voice_q = dict(q, text=transcript)  # swap in the transcript
        row = _run_one(agent, voice_q)
        row["transcript"] = transcript
        row["raw_transcript"] = raw
        row["transcript_wer"] = transcript_wer
        row["original_text"] = q["text"]
        rows.append(row)

    # Print same report format; label it as voice-driven
    if polish:
        print("  · polishing answers (analyst rewrite for the review)…")
    _polish_rows(rows, enabled=polish)
    print("\n" + "=" * 78)
    print(" VOICE BENCHMARK — .wav → STT → pipeline (end-to-end)")
    print("=" * 78)
    print_report(rows)

    if text_rows:
        print_regressions(text_rows, rows)

    if save:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(V6Config.output_dir(),
                            f"bench_voice_{stamp}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"voice_pipeline": rows, "summary": _agg(rows)},
                      f, ensure_ascii=False, indent=2)
        print(f"\n  saved → {path}")
    return rows
