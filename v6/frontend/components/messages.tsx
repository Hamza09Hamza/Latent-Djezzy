"use client";

import { useRef, useState } from "react";
import { toast } from "sonner";
import {
  Volume2,
  Loader2,
  XCircle,
  Sparkles,
  ChevronDown,
  ChevronRight,
  BarChart3,
  FileText,
  Mail,
  Play,
  Pause,
  ExternalLink,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ChartView } from "@/components/ChartView";
import { ReportView } from "@/components/ReportView";
import { chartUrl, httpUrl, ServerConfig } from "@/lib/config";
import { Artifact } from "@/lib/api";

export interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  thinking: string[];
  intent?: string;
  polishRole?: string;
  artifacts: Artifact[];
  audioChunks?: number;
  audioSrc?: string; // demo mode: a ready-made clip to replay (real mode streams)
  error?: boolean;
  streaming?: boolean;
}

export function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center py-16">
      <div className="relative w-24 h-24 rounded-full bg-gradient-to-br from-primary/20 to-accent/20 flex items-center justify-center mb-6 glow-effect">
        <div className="absolute inset-0 rounded-full bg-gradient-to-br from-primary/10 to-accent/10 pulse-ring" />
        <Sparkles className="w-11 h-11 text-primary" />
      </div>
      <h2 className="text-2xl font-semibold mb-2 bg-gradient-to-r from-foreground to-foreground/60 bg-clip-text text-transparent">
        Ask the Djezzy analytics brain
      </h2>
      <p className="text-muted-foreground max-w-md leading-relaxed text-sm">
        Type or speak a KPI question. You&apos;ll watch it think, see the answer
        stream in, and hear it back in a cloned voice — plus charts, reports, or
        an email draft when you ask.
      </p>
    </div>
  );
}

function PlayVoiceButton({ src }: { src: string }) {
  const [playing, setPlaying] = useState(false);
  const ref = useRef<HTMLAudioElement | null>(null);

  function toggle() {
    const a = ref.current;
    if (!a) return;
    if (playing) {
      a.pause();
      setPlaying(false);
    } else {
      a.currentTime = 0;
      a.play().then(() => setPlaying(true)).catch(() => setPlaying(false));
    }
  }

  return (
    <span className="inline-flex items-center">
      {/* React owns this element — src prop wires directly to the DOM attribute */}
      <audio ref={ref} src={src} onEnded={() => setPlaying(false)} preload="auto" />
      <button
        onClick={toggle}
        className="inline-flex items-center gap-1.5 text-xs text-primary hover:text-primary/80 mt-2"
      >
        {playing ? <Pause className="w-3.5 h-3.5" /> : <Play className="w-3.5 h-3.5" />}
        {playing ? "Playing…" : "Play voice"}
      </button>
    </span>
  );
}

export function MessageBubble({ m, cfg }: { m: Message; cfg: ServerConfig }) {
  const isUser = m.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <Card
        className={`max-w-[85%] p-4 backdrop-blur-sm ${
          isUser
            ? "bg-gradient-to-br from-primary to-primary/80 text-background border-primary/30"
            : m.error
            ? "bg-destructive/10 border-destructive/30"
            : "bg-card/80 border-border/50"
        }`}
      >
        <div className="flex items-start gap-3">
          {!isUser && (
            <div
              className={`w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5 ${
                m.error
                  ? "bg-destructive/20"
                  : "bg-gradient-to-br from-primary/20 to-accent/20"
              }`}
            >
              {m.error ? (
                <XCircle className="w-4 h-4 text-destructive" />
              ) : m.audioChunks || m.audioSrc ? (
                <Volume2 className="w-4 h-4 text-primary" />
              ) : (
                <Sparkles className="w-4 h-4 text-primary" />
              )}
            </div>
          )}
          <div className="flex-1 min-w-0">
            {!isUser && (m.intent || m.polishRole) && (
              <div className="flex items-center gap-2 mb-2">
                {m.intent && <Badge>{m.intent}</Badge>}
                {m.polishRole && m.polishRole !== "verbatim" && (
                  <Badge variant="muted">{m.polishRole}</Badge>
                )}
              </div>
            )}

            {!isUser && m.thinking.length > 0 && (
              <ThinkingTrace lines={m.thinking} live={m.streaming} />
            )}

            <p className="text-sm leading-relaxed whitespace-pre-wrap">
              {m.text}
              {m.streaming && !m.text && (
                <span className="inline-flex items-center gap-2 text-muted-foreground">
                  <Loader2 className="w-3.5 h-3.5 animate-spin" /> thinking…
                </span>
              )}
              {m.streaming && m.text && (
                <span className="inline-block w-1.5 h-4 align-middle ml-0.5 bg-primary/70 animate-pulse" />
              )}
            </p>

            {m.artifacts.length > 0 && (
              <div className="mt-3 space-y-2">
                {m.artifacts.map((a, i) => (
                  <ArtifactView key={i} a={a} cfg={cfg} />
                ))}
              </div>
            )}

            {!isUser && m.audioSrc ? (
              <PlayVoiceButton src={m.audioSrc} />
            ) : !isUser && !!m.audioChunks ? (
              <p className="text-xs text-muted-foreground mt-2 flex items-center gap-1.5">
                <Volume2 className="w-3.5 h-3.5" /> spoken in the cloned voice
              </p>
            ) : null}
          </div>
        </div>
      </Card>
    </div>
  );
}

function ThinkingTrace({ lines, live }: { lines: string[]; live?: boolean }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="mb-2 rounded-lg border border-border/40 bg-background/40">
      <button
        onClick={() => setOpen((o) => !o)}
        className="w-full flex items-center gap-1.5 px-3 py-1.5 text-xs text-muted-foreground hover:text-foreground"
      >
        {open ? (
          <ChevronDown className="w-3.5 h-3.5" />
        ) : (
          <ChevronRight className="w-3.5 h-3.5" />
        )}
        {live ? <Loader2 className="w-3 h-3 animate-spin text-accent" /> : null}
        Reasoning ({lines.length})
      </button>
      {open && (
        <div className="px-3 pb-2 space-y-1">
          {lines.map((l, i) => (
            <p key={i} className="text-xs text-muted-foreground leading-relaxed">
              <span className="text-accent/70">💭</span> {l}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

function EmailArtifactCard({ a }: { a: Artifact }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="rounded-lg border border-primary/20 bg-primary/5 overflow-hidden">
      <div className="flex items-start gap-2.5 px-3 py-2.5">
        <div className="w-7 h-7 rounded bg-primary/15 flex items-center justify-center flex-shrink-0 mt-0.5">
          <Mail className="w-3.5 h-3.5 text-primary" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-xs font-semibold text-primary mb-1">Email draft</p>
          {(a.to_name || a.to) && (
            <p className="text-xs text-muted-foreground">
              <span className="text-foreground/50 mr-1">To</span>
              {a.to_name ? `${a.to_name}` : ""}{a.to ? ` <${a.to}>` : ""}
            </p>
          )}
          {a.subject && (
            <p className="text-xs text-muted-foreground mt-0.5">
              <span className="text-foreground/50 mr-1">Subject</span>
              {a.subject}
            </p>
          )}
        </div>
        {a.body && (
          <button
            onClick={() => setOpen((o) => !o)}
            className="flex items-center gap-1 text-xs text-primary/70 hover:text-primary flex-shrink-0 mt-0.5"
          >
            {open ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
            {open ? "Hide" : "View"}
          </button>
        )}
      </div>
      {open && a.body && (
        <div className="mx-3 mb-3 rounded border border-primary/10 bg-background/60 px-3 py-2.5">
          <pre className="text-xs text-foreground/80 whitespace-pre-wrap font-sans leading-relaxed">
            {a.body}
          </pre>
        </div>
      )}
    </div>
  );
}

function ReportArtifactCard({ a, cfg }: { a: Artifact; cfg: ServerConfig }) {
  const [open, setOpen] = useState(false);
  const name = a.name || "report.md";
  const downloadHref = a.url ? httpUrl(cfg, a.url) : null;

  function handleEmail(to: string) {
    if (!cfg.serverUrl) {
      // demo mode — show a toast
      toast.success(`Report queued for delivery to ${to}`);
      return;
    }
    // production — fire-and-forget POST to /send_report
    fetch(`${cfg.serverUrl}/send_report`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${cfg.apiToken}`, "ngrok-skip-browser-warning": "true" },
      body: JSON.stringify({ to, report_name: name }),
    })
      .then((r) => r.ok ? toast.success(`Report sent to ${to}`) : toast.error("Failed to send report"))
      .catch(() => toast.error("Could not reach the server"));
  }

  // If we have a rich report spec, show it inline (expandable).
  if (a.reportSpec) {
    return (
      <div className="rounded-lg border border-accent/20 bg-accent/5 overflow-hidden">
        {/* Collapsed header */}
        <button
          onClick={() => setOpen((o) => !o)}
          className="w-full flex items-center gap-2.5 px-3 py-2.5 hover:bg-accent/5 transition-colors"
        >
          <div className="w-7 h-7 rounded bg-accent/15 flex items-center justify-center flex-shrink-0">
            <FileText className="w-3.5 h-3.5 text-accent" />
          </div>
          <div className="flex-1 text-left min-w-0">
            <p className="text-xs font-semibold text-accent">Report ready</p>
            <p className="text-xs text-muted-foreground font-mono truncate">{name}</p>
          </div>
          {open ? (
            <ChevronDown className="w-3.5 h-3.5 text-muted-foreground flex-shrink-0" />
          ) : (
            <ChevronRight className="w-3.5 h-3.5 text-muted-foreground flex-shrink-0" />
          )}
        </button>
        {/* Expanded: full ReportView */}
        {open && (
          <div className="border-t border-accent/10 p-3">
            <ReportView spec={a.reportSpec} onEmail={handleEmail} />
          </div>
        )}
      </div>
    );
  }

  // Fallback: no rich spec — show a minimal card with a download link.
  return (
    <div className="rounded-lg border border-accent/20 bg-accent/5">
      <div className="flex items-start gap-2.5 px-3 py-2.5">
        <div className="w-7 h-7 rounded bg-accent/15 flex items-center justify-center flex-shrink-0 mt-0.5">
          <FileText className="w-3.5 h-3.5 text-accent" />
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-xs font-semibold text-accent mb-1">Report saved</p>
          <p className="text-xs text-muted-foreground font-mono truncate">{name}</p>
        </div>
        {downloadHref && (
          <a href={downloadHref} target="_blank" rel="noreferrer"
            className="flex items-center gap-1 text-xs text-accent/70 hover:text-accent flex-shrink-0 mt-0.5">
            <ExternalLink className="w-3.5 h-3.5" /> Open
          </a>
        )}
      </div>
    </div>
  );
}

function ArtifactView({ a, cfg }: { a: Artifact; cfg: ServerConfig }) {
  if (a.kind === "chart") {
    if (a.spec) return <ChartView spec={a.spec} />;
    if (a.url) {
      const src = chartUrl(cfg, a.url);
      // eslint-disable-next-line @next/next/no-img-element
      return (
        <a href={src} target="_blank" rel="noreferrer">
          <img src={src} alt="chart" className="rounded-lg border border-border/50 max-w-full" />
        </a>
      );
    }
    return null;
  }
  if (a.kind === "email") return <EmailArtifactCard a={a} />;
  if (a.kind === "report") return <ReportArtifactCard a={a} cfg={cfg} />;
  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground">
      <BarChart3 className="w-4 h-4" /> {a.kind}
    </div>
  );
}
