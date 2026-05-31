# Djezzy AI — LatentMind V6 frontend

A Next.js web client for the V6 assistant. It talks to [`v6/server.py`](../server.py)
(FastAPI + WebSocket) and, over one socket, shows the agent's **reasoning trace**,
streams the **answer text**, plays the answer back in the **cloned voice**, and
renders any **chart / report / email** artifacts.

It's modeled on the `rova-server-tester` voice client (same dark teal/amber
shadcn-style stack) but rebuilt for V6's streaming contract.

## Point it at your backend

Every Colab session prints a **new ngrok URL + API token** (the notebook's
*Serve* cell). Set them in any one of these — later ones win:

1. **`lib/config.ts`** → `DEFAULT_CONFIG` (committed default).
2. **`.env.local`** → copy `.env.local.example`, fill `NEXT_PUBLIC_NGROK_URL`
   and `NEXT_PUBLIC_API_TOKEN`, restart `npm run dev`.
3. **In-app Settings** (gear icon) → paste URL + token, *Save & connect*. Stored
   in `localStorage`, overrides the others, no rebuild. **Best while Colab is
   warming up** — just paste the new URL each session.

The header has a status dot: teal = `/health` OK, red = offline, amber =
checking. Click it (or *Test connection*) to re-probe.

## Run

```bash
cd v6/frontend
npm install      # or: pnpm install
npm run dev      # http://localhost:3000
```

## How it maps to the server

| UI action | Endpoint | What you get |
|---|---|---|
| Type + Enter / ▶ | `WS /ws` | live `thinking` → `meta` (intent/role) → answer `tokens` → cloned-voice `audio` → `artifact`s → `done` |
| 🎤 mic | `POST /ask_voice` | STT transcription + answer + cloned-voice PCM16 chunks (played gaplessly) |
| status dot | `GET /health` | backend reachability |
| chart artifact | `GET /chart/{name}` | inline PNG (token in query) |

Audio is base64 little-endian PCM16; [`lib/audio.ts`](lib/audio.ts) schedules
chunks back-to-back via the Web Audio API so playback is gapless. The token is
sent as a bearer header on REST and as a `?token=` query param on the WebSocket
(browsers can't set WebSocket headers), plus `ngrok-skip-browser-warning` on
fetches to bypass ngrok's free-tier interstitial.

## Charts

Charts are **rendered in the browser**, not sent as images. The backend
([`v6/chartspec.py`](../chartspec.py)) decides the chart type from the data
shape + the executed SQL + the question and sends a typed JSON spec on the
`chart` artifact; [`components/ChartView.tsx`](components/ChartView.tsx) draws it
with Recharts in the teal/amber theme:

| Question shape | Spec type | Chart |
|---|---|---|
| top/bottom N, "highest/lowest" | `hbar` | sorted horizontal bars |
| trend over time | `line` / `area` | one line per series (per wilaya) |
| compare a few categories | `bar` | vertical bars |
| share / répartition (≤6 parts) | `pie` | donut |
| a single KPI | `stat` | big number card |

Every numeric arrives **frozen by `numfmt`**: the raw value drives the axis, the
humanized `<col>_fmt` string ("253.4 million DZD") drives the tooltip — so the
chart can never re-round a figure. If the spec is missing, it falls back to the
backend PNG (`/chart/{name}`).

## Notes

- One GPU on the backend → requests serialize; the UI disables input while a
  turn is in flight (the ▶ becomes a ■ stop).
- A Colab tunnel is ephemeral. When it dies, the dot goes red — open Settings
  and paste the new URL/token.
- For a persistent deployment, run `v6/server.py` on a real GPU host and put its
  URL in `.env.local`.
