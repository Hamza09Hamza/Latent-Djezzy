// ============================================================================
//  V6 frontend — backend connection config
//
//  Each Colab session prints a NEW ngrok URL + API token (the notebook's
//  "Serve" cell). When Colab is ready, point the frontend at the live backend
//  in ONE of three ways, in increasing precedence:
//
//   1. Edit DEFAULT_CONFIG below (committed default — fine for a fixed host).
//   2. Set NEXT_PUBLIC_NGROK_URL / NEXT_PUBLIC_API_TOKEN in `.env.local`
//      (see .env.local.example) and restart `npm run dev`.
//   3. Use the in-app Settings panel (gear icon) — saved to localStorage,
//      overrides everything, no rebuild. Best while Colab is warming up.
//
//  loadConfig() merges these so the app always has a single source of truth.
// ============================================================================

export interface ServerConfig {
  /** ngrok HTTPS base, e.g. https://1a2b-34-56.ngrok-free.app (no trailing /) */
  serverUrl: string;
  /** V6_API_TOKEN printed by the serve cell — every endpoint requires it */
  apiToken: string;
}

// ── EDIT THESE when you have a long-lived host; otherwise use .env.local ─────
export const DEFAULT_CONFIG: ServerConfig = {
  serverUrl:
    process.env.NEXT_PUBLIC_NGROK_URL ??
    "https://YOUR-NGROK-SUBDOMAIN.ngrok-free.app",
  apiToken: process.env.NEXT_PUBLIC_API_TOKEN ?? "",
};

const STORAGE_KEY = "v6.serverConfig";

/** Strip whitespace + any trailing slashes from a base URL. */
export function normalizeBase(url: string): string {
  return (url || "").trim().replace(/\/+$/, "");
}

/** Effective config: localStorage override (if any) on top of DEFAULT_CONFIG. */
export function loadConfig(): ServerConfig {
  if (typeof window === "undefined") return DEFAULT_CONFIG;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_CONFIG;
    const saved = JSON.parse(raw) as Partial<ServerConfig>;
    return {
      serverUrl: saved.serverUrl?.trim() || DEFAULT_CONFIG.serverUrl,
      apiToken: saved.apiToken?.trim() ?? DEFAULT_CONFIG.apiToken,
    };
  } catch {
    return DEFAULT_CONFIG;
  }
}

export function saveConfig(cfg: ServerConfig): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify({
      serverUrl: normalizeBase(cfg.serverUrl),
      apiToken: cfg.apiToken.trim(),
    })
  );
}

export function clearConfig(): void {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(STORAGE_KEY);
}

export function isConfigured(cfg: ServerConfig): boolean {
  const base = normalizeBase(cfg.serverUrl);
  return (
    base.startsWith("http") &&
    !base.includes("YOUR-NGROK-SUBDOMAIN") &&
    cfg.apiToken.trim().length > 0
  );
}

// ── URL helpers ──────────────────────────────────────────────────────────────

/** ws(s):// .../ws?token=... — derived from the http(s) base. */
export function wsUrl(cfg: ServerConfig): string {
  const base = normalizeBase(cfg.serverUrl);
  const wsBase = base.startsWith("https")
    ? "wss" + base.slice(5)
    : base.startsWith("http")
    ? "ws" + base.slice(4)
    : base;
  return `${wsBase}/ws?token=${encodeURIComponent(cfg.apiToken)}`;
}

export function httpUrl(cfg: ServerConfig, path: string): string {
  const base = normalizeBase(cfg.serverUrl);
  return `${base}${path.startsWith("/") ? path : "/" + path}`;
}

/** Absolute URL for a chart artifact (server returns a relative "/chart/..."). */
export function chartUrl(cfg: ServerConfig, rel: string): string {
  const base = normalizeBase(cfg.serverUrl);
  const path = rel.startsWith("http")
    ? rel
    : base + (rel.startsWith("/") ? rel : "/" + rel);
  const sep = path.includes("?") ? "&" : "?";
  return `${path}${sep}token=${encodeURIComponent(cfg.apiToken)}`;
}

/** Bearer token + the header that skips ngrok's free-tier browser warning. */
export function authHeaders(cfg: ServerConfig): Record<string, string> {
  return {
    Authorization: `Bearer ${cfg.apiToken}`,
    "ngrok-skip-browser-warning": "true",
  };
}
