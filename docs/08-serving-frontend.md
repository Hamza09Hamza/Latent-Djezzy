# 08 — Serving & Frontend

> Code: `v6/server.py` (FastAPI + WebSocket), `v6/frontend/` (Next.js client),
> `v6/cli.py` (terminal client), `v6/v6_serve.ipynb` (Colab serving cells).

What turns the agent into something a person can use — a streaming API and a web
client that plays the cloned voice back in real time.

---

## 1. Why everything runs server-side

A browser **cannot** run the models: not the XTTS voice clone, not the Qwen SLM,
not BGE-M3. And the browser's Web Speech API is robotic and can't clone a voice. So
**everything runs server-side** — STT, the brain, the SLM, the polisher,
cloned-voice TTS — and the server **streams** text and audio to a thin client whose
only jobs are: capture mic audio, render the streamed text/chart/report, and play
the streamed audio gaplessly.

---

## 2. The server (`server.py`) — FastAPI + WebSocket

### Streaming WebSocket (`/ws`) — one socket, text + audio

| Event | Payload | Meaning |
|---|---|---|
| `thinking` | text | the 💭 brain trace as it works (rendered dim) |
| `meta` | `{intent, role}` | how the answer is being produced |
| `token` | text | the answer text, streamed (ChatGPT-style) |
| `audio` | `{seq, sr, data}` | cloned-voice **PCM16 base64** chunks, ~1 sentence behind the text |
| `artifact` | chart/report/email pointers | links to generated artifacts |
| `answer` / `done` | final text / end-of-turn | terminal markers |

The blocking pipeline (GPU work) runs in a **thread**; events are marshalled to the
asyncio event loop through a **queue**, so the socket stays responsive while the GPU
is busy. Audio trails the text by ~one sentence because XTTS streams per sentence.

### REST fallbacks

- `POST /ask` — JSON, non-streaming (text + artifacts).
- `POST /ask_voice` — multipart audio → STT → pipeline → text + PCM16 chunks.
- `GET /chart/{name}` — fetch a generated chart PNG.
- `GET /health` — liveness.
- `GET /` — a minimal built-in browser client that plays the streamed audio
  gaplessly via the Web Audio API (a quick demo without the Next.js app).

### Constraints (it's a demo server)

- **One GPU → requests serialize.** An `asyncio.Lock` ensures one request at a time.
- **ngrok is public → auth required.** Every endpoint requires `V6_API_TOKEN`
  (bearer header or `?token=`); if unset, one is generated and printed at startup.
- **Colab is ephemeral.** This is a dev/demo server. For production, run the same
  FastAPI app on a persistent GPU host (Modal / RunPod / HF Endpoints) behind a
  queue.

On Colab, `v6_serve.ipynb` starts `uvicorn` in a thread and opens an ngrok tunnel;
locally it's `uvicorn v6.server:app --host 0.0.0.0 --port 8000`.

---

## 3. The frontend (`v6/frontend/`) — Next.js web client

**Stack:**

| Piece | Choice |
|---|---|
| Framework | **Next.js 15** (App Router) + **React 19** |
| Styling | **Tailwind CSS** + `tailwindcss-animate`, shadcn/ui-style primitives (`components/ui/*`: button, card, input, label, badge) |
| Charts | **Recharts** (`ChartView.tsx`) — renders the typed JSON chart spec from `chartspec.py` |
| Icons / theme / misc | `lucide-react`, `next-themes`, `geist` font, `sonner` toasts |

**Structure:**

```
app/
  layout.tsx        — root layout, theme provider
  page.tsx          — the main chat experience
  globals.css       — Tailwind base
  demo/page.tsx     — a scripted demo view
components/
  messages.tsx      — the chat transcript (thinking trace + streamed answer)
  ChartView.tsx     — interactive Recharts chart from the chart spec
  ReportView.tsx    — renders a filled report
  ui/*              — shadcn-style primitives
hooks/
  useAudioRecorder.ts — mic capture
lib/
  api.ts            — WebSocket + REST client to server.py
  audio.ts          — gapless PCM16 playback via Web Audio API
  config.ts         — server URL + token
  demo.ts           — demo script data
```

**How a turn renders:**

1. The user types or records (mic via `useAudioRecorder`).
2. `lib/api.ts` opens the WebSocket (`/ws`) with the bearer token.
3. `thinking` events stream into `messages.tsx` as a dim, live **brain trace** —
   the user literally watches the agent decide (rag → sql → …).
4. `token` events stream the answer ChatGPT-style.
5. `audio` chunks feed `lib/audio.ts`, which schedules gapless PCM16 playback in the
   **cloned voice**, ~1 sentence behind the text.
6. `artifact` events render an interactive **chart** (`ChartView`), a **report**
   (`ReportView`), or an email-draft pointer.

The chart is rendered from the **typed JSON spec** (not a static PNG), so it's
interactive — and because the figures were frozen by `numfmt`, the chart can't
re-round a number. The PNG is kept only as a notebook/report fallback.

---

## 4. The CLI (`cli.py`)

A terminal client (`python3 -m v6.cli`) for local, no-browser use. It runs the same
`LatentMindV6.ask()` loop and prints the thinking trace + answer — the fastest way
to exercise the pipeline during development.

---

## 5. The full serving picture

```
            ┌─────────────────── server (GPU host) ───────────────────┐
browser ◄──►│  FastAPI /ws  ◄─ queue ─  pipeline thread:               │
 (Next.js)  │   thinking/token/audio/   STT → brain loop → SLM →        │
            │   artifact events         polisher → XTTS (cloned voice)  │
mobile  ◄──►│  POST /ask, /ask_voice (REST fallbacks)                   │
            │  GET /, /chart/{name}, /health                            │
            └───────────────────────────────────────────────────────────┘
            auth: V6_API_TOKEN on every endpoint · one asyncio.Lock (serialize)
```

The client stays thin and portable (web today, mobile tomorrow); all the heavy,
private, model-dependent work stays on the GPU host where the voice and models live.

→ Next: [Training](09-training.md).
