// ============================================================================
//  V6 frontend — client for v6/server.py
//
//  Streaming paths, mapping 1:1 to the server's contract:
//    • streamAsk()   → WS /ws   text in, fully streamed out
//                               (thinking → meta → tokens → audio → done)
//    • streamVoice() → WS /ws   audio in; STT runs server-side and emits a
//                               {transcript} first, then streams identically.
//    • askVoice()    → POST /ask_voice  legacy non-streaming voice (buffered)
//    • checkHealth() → GET  /health     "is Colab up?" probe
// ============================================================================

import {
  ServerConfig,
  wsUrl,
  httpUrl,
  authHeaders,
} from "./config";

export interface ChartSeries {
  key: string;
  label: string;
  unit?: string;
}

export interface ChartSpec {
  type: "line" | "area" | "bar" | "hbar" | "pie" | "stat";
  title?: string;
  x?: { key: string; label?: string };
  series: ChartSeries[];
  data: Array<Record<string, unknown>>;
  meta?: Record<string, unknown>;
}

export interface ReportMetricItem {
  label: string;
  value: string;
  change?: string;
  trend?: "up" | "down" | "flat";
}

export interface ReportSection {
  type: "heading" | "text" | "chart" | "table" | "metrics";
  text?: string;
  spec?: ChartSpec;
  columns?: string[];
  rows?: (string | number)[][];
  items?: ReportMetricItem[];
}

export interface ReportSpec {
  title: string;
  subtitle?: string;
  generated_at?: string;
  sections: ReportSection[];
}

export interface Artifact {
  kind: "chart" | "report" | "email" | string;
  url?: string;
  spec?: ChartSpec;         // chart: interactive Recharts spec
  reportSpec?: ReportSpec;  // report: rich structured document
  name?: string;
  to?: string;
  to_name?: string;
  subject?: string;
  body?: string;            // email: draft body text
}

export interface StreamHandlers {
  onOpen?: () => void;
  onTranscript?: (text: string) => void; // voice only — what STT heard (first)
  onThinking?: (text: string) => void;
  onMeta?: (intent: string, role: string) => void;
  onToken?: (text: string) => void;
  onAudio?: (b64: string, sr: number) => void;
  onArtifact?: (art: Artifact) => void;
  onAnswer?: (text: string, lang?: string) => void;
  onError?: (text: string) => void;
  onDone?: () => void;
}

/**
 * Open a /ws WebSocket, dispatch streamed events to the handlers, and send the
 * opening frame once connected (text question or voice audio — `buildOpen`
 * returns the JSON to send). Returns a `close()` to abort; the socket closes
 * itself on `done`. Auth lives in the URL query (browsers can't set WS headers).
 */
function openStream(
  cfg: ServerConfig,
  h: StreamHandlers,
  buildOpen: () => Promise<string> | string
): () => void {
  let ws: WebSocket;
  try {
    ws = new WebSocket(wsUrl(cfg));
  } catch (e) {
    h.onError?.(
      "Could not open a WebSocket — check the server URL in Settings."
    );
    h.onDone?.();
    return () => {};
  }

  let closedByUs = false;

  ws.onopen = async () => {
    h.onOpen?.();
    try {
      ws.send(await buildOpen());
    } catch {
      h.onError?.("Could not prepare the request to send.");
      closedByUs = true;
      ws.close();
      h.onDone?.();
    }
  };

  ws.onmessage = (e) => {
    let m: any;
    try {
      m = JSON.parse(e.data);
    } catch {
      return;
    }
    switch (m.type) {
      case "transcript":
        h.onTranscript?.(m.text);
        break;
      case "thinking":
        h.onThinking?.(m.text);
        break;
      case "meta":
        h.onMeta?.(m.intent, m.role);
        break;
      case "token":
        h.onToken?.(m.text);
        break;
      case "audio":
        h.onAudio?.(m.data, m.sr);
        break;
      case "artifact":
        h.onArtifact?.(m as Artifact);
        break;
      case "answer":
        h.onAnswer?.(m.text, m.lang);
        break;
      case "error":
        h.onError?.(m.text);
        break;
      case "done":
        h.onDone?.();
        closedByUs = true;
        ws.close();
        break;
    }
  };

  ws.onerror = () => {
    h.onError?.(
      "WebSocket error — is the ngrok URL right and is Colab still running?"
    );
  };

  ws.onclose = (ev) => {
    if (ev.code === 4401) {
      h.onError?.("Unauthorized — the API token doesn't match the server.");
      h.onDone?.();
    } else if (!closedByUs) {
      // closed without a clean "done"
      h.onDone?.();
    }
  };

  return () => {
    closedByUs = true;
    try {
      ws.close();
    } catch {
      /* noop */
    }
  };
}

/** Strip the `data:<mime>;base64,` prefix a FileReader data URL carries. */
function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const url = reader.result as string;
      resolve(url.includes(",") ? url.slice(url.indexOf(",") + 1) : url);
    };
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

/** Text question → WS /ws, fully streamed. */
export function streamAsk(
  cfg: ServerConfig,
  question: string,
  thread: string,
  h: StreamHandlers
): () => void {
  return openStream(cfg, h, () => JSON.stringify({ question, thread }));
}

/**
 * Voice question → WS /ws. The audio is sent as base64; the server transcribes
 * it, emits {transcript} (→ onTranscript), then streams thinking/tokens/audio
 * exactly like a text turn. Same UX as text, no buffered HTTP round-trip.
 */
export function streamVoice(
  cfg: ServerConfig,
  blob: Blob,
  thread: string,
  h: StreamHandlers
): () => void {
  const fmt = (blob.type.split("/")[1] || "webm").split(";")[0];
  return openStream(cfg, h, async () =>
    JSON.stringify({ audio: await blobToBase64(blob), format: fmt, thread })
  );
}

export interface VoiceResult {
  question: string; // STT transcription
  intent: string;
  answer: string;
  thinking: string[];
  artifacts: Artifact[];
  audio_pcm16_b64: string[];
  sample_rate: number;
}

/** Upload a recording to /ask_voice → transcription + answer + cloned audio. */
export async function askVoice(
  cfg: ServerConfig,
  blob: Blob,
  filename = "recording.webm"
): Promise<VoiceResult> {
  const fd = new FormData();
  fd.append("file", blob, filename);
  const res = await fetch(httpUrl(cfg, "/ask_voice"), {
    method: "POST",
    headers: authHeaders(cfg), // no Content-Type — the browser sets the boundary
    body: fd,
  });
  if (res.status === 401) throw new Error("Unauthorized — check the API token.");
  if (!res.ok) throw new Error(`Server error ${res.status}`);
  return (await res.json()) as VoiceResult;
}

/** GET /health (no token required) — used for the connection indicator. */
export async function checkHealth(cfg: ServerConfig): Promise<boolean> {
  try {
    const res = await fetch(httpUrl(cfg, "/health"), {
      headers: { "ngrok-skip-browser-warning": "true" },
    });
    return res.ok;
  } catch {
    return false;
  }
}
