// ============================================================================
//  V6 frontend — client for v6/server.py
//
//  Two paths, mapping 1:1 to the server's contract:
//    • streamAsk()  → WS  /ws       text in, fully streamed out
//                                   (thinking → meta → tokens → audio → done)
//    • askVoice()   → POST /ask_voice  audio in, STT + answer + cloned audio
//    • checkHealth()→ GET  /health     "is Colab up?" probe
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
 * Open a WebSocket to /ws, send the question, and dispatch streamed events.
 * Returns a `close()` you can call to abort. The socket closes itself on
 * `done`. Auth + token live in the URL query (browsers can't set WS headers).
 */
export function streamAsk(
  cfg: ServerConfig,
  question: string,
  thread: string,
  h: StreamHandlers
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

  ws.onopen = () => {
    h.onOpen?.();
    ws.send(JSON.stringify({ question, thread }));
  };

  ws.onmessage = (e) => {
    let m: any;
    try {
      m = JSON.parse(e.data);
    } catch {
      return;
    }
    switch (m.type) {
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
