"""v6/server.py — Temporary API server for the Djezzy Voice Assistant.

Exposes the agent over HTTP + WebSocket so a web or mobile client can call it.
Built to run on Colab behind an ngrok tunnel (see the notebook's "Serve" cell).

EVERYTHING runs server-side — STT, brain, SLM, polisher, TTS — because the
browser can't run your cloned XTTS voice or the Qwen/BGE models. Over one
WebSocket the client receives, in order:

    {type:"thinking", text}              the 💭 trace, as the brain works
    {type:"meta", intent, role}          how the answer will be produced
    {type:"token", text}                 the answer text, streamed
    {type:"audio", seq, sr, data}        cloned-voice PCM16 chunks (base64),
                                         ~1 sentence behind the text
    {type:"artifact", kind, ...}         chart / report / email pointers
    {type:"answer", text}                the full final text
    {type:"done"}

Design notes:
  - **Single GPU** → requests are serialized with an asyncio lock; the blocking
    pipeline runs in a thread so the event loop stays responsive.
  - **ngrok is public** → every endpoint needs the token in V6_API_TOKEN
    (header `Authorization: Bearer <token>` or `?token=<token>`). If unset, one
    is generated at startup and printed — copy it into your client.
  - This is a **demo/dev** server: a Colab session is ephemeral. For production,
    run the same app on a persistent GPU host (Modal / RunPod / HF Endpoints).

Run:  uvicorn v6.server:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations
import asyncio
import base64
import os
import secrets
import tempfile

from .config import V6Config
from .state import initial_state

# ── auth ─────────────────────────────────────────────────────────────────────
API_TOKEN = os.environ.get("V6_API_TOKEN", "")
if not API_TOKEN:
    API_TOKEN = secrets.token_urlsafe(16)
    print(f"[server] V6_API_TOKEN not set — generated one: {API_TOKEN}")

# One GPU: never run two pipelines at once.
_GPU_LOCK = asyncio.Lock()


# ── role selection (mirrors notebook _pick_role / benchmark _spoken_role) ────
def _pick_role(result: dict) -> tuple[str | None, str]:
    """(role, text) for the polisher. role=None → speak the text verbatim."""
    intent = result.get("intent", "data")
    answer = result.get("final_answer", "") or ""
    exec_ok = bool(result.get("exec_ok"))
    if intent in ("off_topic", "unanswerable"):
        return None, answer
    if intent in ("greeting", "meta"):
        return "chat", answer
    if intent == "definition":
        return "polish", answer
    if result.get("document_path") and not exec_ok:
        return None, "Report generated from the previous result."
    low = answer.lower()
    if any(p in low for p in ("couldn't build", "failed to run", "no matching rows",
                              "wasn't able to pull", "no data found")):
        return "clarify", answer
    if not exec_ok:
        return "clarify", answer
    data_part = answer.split("📧")[0].split("📊")[0].split("📄")[0].strip()
    return "analyze", data_part


def _recent_dialog(turns: list[dict], summary: str = "", limit: int = 2) -> str:
    lines = [summary.strip()] if summary else []
    for t in (turns or [])[-limit:]:
        q = (t.get("query") or "").strip()
        a = (t.get("answer") or "").strip()
        if q:
            lines.append(f"User: {q[:80]}")
        if a:
            lines.append(f"You: {a[:120]}")
    return "\n".join(lines)


def _pcm16_b64(chunk) -> str:
    """float32 audio (numpy 1-D, [-1,1]) → base64 little-endian PCM16."""
    import numpy as np
    x = np.clip(np.asarray(chunk, dtype="float32"), -1.0, 1.0)
    return base64.b64encode((x * 32767.0).astype("<i2").tobytes()).decode("ascii")


# ── the pipeline (sync) — emits events through a callback ────────────────────
def _run_pipeline(question: str, thread: str, emit, *, want_audio: bool) -> None:
    """Run one turn: stream thinking → answer tokens → (optional) audio chunks
    → artifacts. `emit(event_dict)` is called for every event; it is safe to
    call from this worker thread (the WS bridge marshals to the event loop)."""
    from .graph import get_agent
    from .slm import get_polisher, get_slm
    from .speech import get_tts, language_for

    agent = get_agent()
    state = initial_state(question, thread)
    cfg = {"configurable": {"thread_id": thread}, "recursion_limit": 60}

    shown = 0
    intent = ""
    final_answer = ""
    exec_ok = False
    chart_path = document_path = ""
    email_draft = None
    turns: list = []
    memory_summary = ""

    for event in agent.graph.stream(state, cfg, stream_mode="updates"):
        for node, data in event.items():
            if not data:
                continue
            th = data.get("thoughts")
            if th and len(th) > shown:
                for t in th[shown:]:
                    if t.get("kind") == "thinking":
                        emit({"type": "thinking", "text": t["text"]})
                shown = len(th)
            if node == "brain":
                intent = data.get("intent", intent)
            elif node == "sql":
                exec_ok = data.get("exec_ok", exec_ok)
            elif node == "chart":
                chart_path = data.get("chart_path", "") or chart_path
            elif node == "template":
                document_path = data.get("document_path", "") or document_path
            elif node == "email":
                email_draft = data.get("email_draft") or email_draft
            elif node == "communicator":
                final_answer = data.get("final_answer", "")
                turns = data.get("turns", turns)
                memory_summary = data.get("memory_summary", memory_summary)

    role, text = _pick_role({
        "intent": intent, "final_answer": final_answer, "exec_ok": exec_ok,
        "document_path": document_path})
    emit({"type": "meta", "intent": intent, "role": role or "verbatim"})

    lang = language_for(question)
    mem = (_recent_dialog(turns[:-1] if turns else [], memory_summary)
           if role == "chat" else "")
    collected: list[str] = []

    def token_source():
        if role is None:
            for w in text.split(" "):
                piece = w + " "
                collected.append(piece)
                emit({"type": "token", "text": piece})
                yield piece
            return
        for tok in get_polisher().stream(text, question, role=role, memory=mem):
            collected.append(tok)
            emit({"type": "token", "text": tok})
            yield tok

    if want_audio:
        sr = get_tts().sample_rate
        seq = 0
        for chunk in get_tts().stream(token_source(), lang):
            emit({"type": "audio", "seq": seq, "sr": sr, "data": _pcm16_b64(chunk)})
            seq += 1
    else:
        for _ in token_source():
            pass

    if chart_path:
        emit({"type": "artifact", "kind": "chart",
              "url": f"/chart/{os.path.basename(chart_path)}"})
    if document_path:
        emit({"type": "artifact", "kind": "report",
              "name": os.path.basename(document_path)})
    if email_draft and email_draft.get("to"):
        emit({"type": "artifact", "kind": "email",
              "to": email_draft.get("to"), "to_name": email_draft.get("to_name"),
              "subject": email_draft.get("subject", "")})

    emit({"type": "answer", "text": "".join(collected).strip(), "lang": lang})
    get_slm().clear_thread(thread)


# ── FastAPI app ──────────────────────────────────────────────────────────────
def _build_app():
    from fastapi import (FastAPI, Header, HTTPException, UploadFile, File,
                         WebSocket, WebSocketDisconnect)
    from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

    app = FastAPI(title="Djezzy Voice Assistant API")

    def _require_http(authorization: str | None, token: str | None) -> None:
        supplied = token or (authorization.split()[-1] if authorization else None)
        if supplied != API_TOKEN:
            raise HTTPException(status_code=401, detail="bad or missing token")

    async def _collect(question: str, thread: str, want_audio: bool) -> dict:
        """Run the pipeline to completion (no streaming) and return a summary."""
        loop = asyncio.get_event_loop()
        events: list[dict] = []
        audio_chunks: list = []

        def emit(ev):
            if ev.get("type") == "audio":
                audio_chunks.append(ev["data"])
            else:
                events.append(ev)

        def worker():
            _run_pipeline(question, thread, emit, want_audio=want_audio)

        async with _GPU_LOCK:
            await loop.run_in_executor(None, worker)

        answer = next((e["text"] for e in reversed(events)
                       if e.get("type") == "answer"), "")
        return {
            "question": question,
            "intent": next((e["intent"] for e in events if e.get("type") == "meta"), ""),
            "answer": answer,
            "thinking": [e["text"] for e in events if e.get("type") == "thinking"],
            "artifacts": [e for e in events if e.get("type") == "artifact"],
            "_audio_chunks": audio_chunks,
        }

    @app.get("/health")
    async def health():
        return {"ok": True, "service": "djezzy-voice-assistant"}

    @app.post("/ask")
    async def ask(payload: dict, authorization: str | None = Header(default=None),
                  token: str | None = None):
        _require_http(authorization, token)
        question = (payload or {}).get("question", "").strip()
        if not question:
            raise HTTPException(status_code=400, detail="missing 'question'")
        thread = (payload or {}).get("thread", "api")
        res = await _collect(question, thread, want_audio=False)
        res.pop("_audio_chunks", None)
        return JSONResponse(res)

    @app.post("/ask_voice")
    async def ask_voice(file: UploadFile = File(...),
                        authorization: str | None = Header(default=None),
                        token: str | None = None):
        _require_http(authorization, token)
        from .speech import get_stt
        suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
            fh.write(await file.read())
            path = fh.name
        try:
            question = get_stt().transcribe(path)["text"]
        finally:
            os.unlink(path)
        if not question.strip():
            raise HTTPException(status_code=422, detail="empty transcription")
        res = await _collect(question, "api_voice", want_audio=True)
        chunks = res.pop("_audio_chunks", [])
        res["audio_pcm16_b64"] = chunks         # list of base64 PCM16 chunks
        res["sample_rate"] = V6Config.TTS_SAMPLE_RATE
        return JSONResponse(res)

    @app.get("/chart/{name}")
    async def chart(name: str, authorization: str | None = Header(default=None),
                    token: str | None = None):
        _require_http(authorization, token)
        path = os.path.join(V6Config.chart_dir(), os.path.basename(name))
        if not os.path.isfile(path):
            raise HTTPException(status_code=404, detail="chart not found")
        return FileResponse(path, media_type="image/png")

    @app.websocket("/ws")
    async def ws(socket: WebSocket):
        if socket.query_params.get("token") != API_TOKEN:
            await socket.close(code=4401)
            return
        await socket.accept()
        loop = asyncio.get_event_loop()
        try:
            while True:
                msg = await socket.receive_json()
                question = (msg or {}).get("question", "").strip()
                if not question:
                    await socket.send_json(
                        {"type": "error", "text": "send {question: '...'}"})
                    continue
                thread = (msg or {}).get("thread", "web")
                queue: asyncio.Queue = asyncio.Queue()

                def emit(ev):
                    loop.call_soon_threadsafe(queue.put_nowait, ev)

                def worker():
                    try:
                        _run_pipeline(question, thread, emit, want_audio=True)
                    except Exception as exc:  # noqa: BLE001
                        emit({"type": "error", "text": str(exc)})
                    finally:
                        emit({"type": "done"})

                async with _GPU_LOCK:
                    fut = loop.run_in_executor(None, worker)
                    while True:
                        ev = await queue.get()
                        await socket.send_json(ev)
                        if ev.get("type") == "done":
                            break
                    await fut
        except WebSocketDisconnect:
            return

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return _CLIENT_HTML

    @app.on_event("startup")
    async def _warm():
        # Preload models so the first request isn't a multi-minute cold start.
        try:
            from .graph import get_agent
            from .slm import get_slm, get_polisher
            from .brain import get_brain
            get_agent(); get_slm(); get_brain(); get_polisher()
            if V6Config.TTS_ENABLED:
                from .speech import get_tts
                get_tts()
            print("[server] models warm — ready")
        except Exception as exc:  # noqa: BLE001
            print(f"[server] warm-up skipped: {exc}")

    return app


# ── minimal browser client (served at /) ─────────────────────────────────────
# Connects to /ws, shows the thinking trace + streamed answer text, and plays
# the cloned-voice PCM16 chunks gaplessly via the Web Audio API.
_CLIENT_HTML = """<!doctype html><html><head><meta charset="utf-8">
<title>Djezzy Voice Assistant</title><style>
 body{font:15px/1.5 system-ui;margin:0;background:#0f1115;color:#e6e6e6}
 .wrap{max-width:760px;margin:0 auto;padding:24px}
 h1{font-size:18px;color:#7fd1ae} .row{display:flex;gap:8px;margin:12px 0}
 input,button{font:15px system-ui;padding:10px;border-radius:8px;border:1px solid #333}
 input{flex:1;background:#1a1d24;color:#fff} button{background:#7fd1ae;border:0;cursor:pointer}
 #tok{padding:6px 0} .think{color:#7f8694;font-size:13px} .meta{color:#c9a227;font-size:12px}
 .art{color:#7fb2d1;font-size:13px} img{max-width:100%;border-radius:8px;margin-top:8px}
</style></head><body><div class="wrap">
 <h1>Djezzy Voice Assistant</h1>
 <div class="row">
   <input id="tok" placeholder="API token (from the serve cell)">
 </div>
 <div class="row">
   <input id="q" placeholder="Ask in French or English… e.g. Compare le revenu net entre Alger et Oran"
          onkeydown="if(event.key==='Enter')send()">
   <button onclick="send()">Ask</button>
 </div>
 <div id="think"></div><div id="meta"></div>
 <div id="ans" style="margin-top:10px;font-size:16px"></div>
 <div id="art"></div>
<script>
let ws, ac, playAt=0;
function audioCtx(){ if(!ac){ac=new (window.AudioContext||window.webkitAudioContext)();playAt=ac.currentTime;} return ac; }
function playChunk(b64, sr){
  const ctx=audioCtx(); const raw=atob(b64); const n=raw.length/2;
  const buf=ctx.createBuffer(1,n,sr); const ch=buf.getChannelData(0);
  for(let i=0;i<n;i++){ let s=(raw.charCodeAt(2*i)|(raw.charCodeAt(2*i+1)<<8)); if(s>32767)s-=65536; ch[i]=s/32768; }
  const src=ctx.createBufferSource(); src.buffer=buf; src.connect(ctx.destination);
  const t=Math.max(ctx.currentTime, playAt); src.start(t); playAt=t+buf.duration;
}
function send(){
  const token=document.getElementById('tok').value.trim();
  const q=document.getElementById('q').value.trim(); if(!q)return;
  document.getElementById('think').innerHTML='';
  document.getElementById('ans').textContent='';
  document.getElementById('meta').textContent='';
  document.getElementById('art').innerHTML='';
  audioCtx(); playAt=ac.currentTime;
  const proto=location.protocol==='https:'?'wss':'ws';
  ws=new WebSocket(`${proto}://${location.host}/ws?token=${encodeURIComponent(token)}`);
  ws.onopen=()=>ws.send(JSON.stringify({question:q,thread:'web'}));
  ws.onmessage=(e)=>{ const m=JSON.parse(e.data);
    if(m.type==='thinking'){ const d=document.createElement('div'); d.className='think'; d.textContent='💭 '+m.text; document.getElementById('think').appendChild(d); }
    else if(m.type==='meta'){ document.getElementById('meta').textContent=`intent: ${m.intent} · ${m.role}`; }
    else if(m.type==='token'){ document.getElementById('ans').textContent+=m.text; }
    else if(m.type==='audio'){ playChunk(m.data, m.sr); }
    else if(m.type==='artifact' && m.kind==='chart'){ const i=document.createElement('img'); i.src=m.url+'?token='+encodeURIComponent(token); document.getElementById('art').appendChild(i); }
    else if(m.type==='artifact'){ const d=document.createElement('div'); d.className='art'; d.textContent=`📎 ${m.kind}: ${m.subject||m.name||m.to||''}`; document.getElementById('art').appendChild(d); }
    else if(m.type==='error'){ document.getElementById('ans').textContent='⚠ '+m.text; }
    else if(m.type==='done'){ ws.close(); }
  };
}
</script></div></body></html>"""


# Module-level app for `uvicorn v6.server:app`. Built lazily-ish at import.
app = _build_app()
