"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import {
  FlaskConical,
  Play,
  RotateCcw,
  ArrowLeft,
  Square,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { MessageBubble, Message } from "@/components/messages";
import { Artifact } from "@/lib/api";
import { ServerConfig } from "@/lib/config";
import { DEMO_TURNS, DemoTurn, demoVoiceSrc } from "@/lib/demo";

// No backend in demo mode — charts render from the spec, so cfg is unused.
const DUMMY_CFG: ServerConfig = { serverUrl: "", apiToken: "" };

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
let _seq = 0;
const newId = () => `demo-${Date.now()}-${_seq++}`;

function turnArtifacts(turn: DemoTurn): Artifact[] {
  return [
    ...(turn.chart ? [{ kind: "chart" as const, spec: turn.chart }] : []),
    ...(turn.artifacts || []),
  ];
}

function staticMessages(): Message[] {
  const out: Message[] = [];
  DEMO_TURNS.forEach((turn, idx) => {
    out.push({
      id: newId(),
      role: "user",
      text: turn.user,
      thinking: [],
      artifacts: [],
    });
    out.push({
      id: newId(),
      role: "assistant",
      text: turn.answer,
      thinking: turn.thinking,
      intent: turn.intent,
      polishRole: turn.polishRole,
      artifacts: turnArtifacts(turn),
      audioSrc: turn.audioClip ?? demoVoiceSrc(idx),
    });
  });
  return out;
}

export default function DemoPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [busy, setBusy] = useState(false);
  const stopRef = useRef(false);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    setMessages(staticMessages());
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  function patch(id: string, fn: (m: Message) => Message) {
    setMessages((prev) => prev.map((m) => (m.id === id ? fn(m) : m)));
  }

  function playClip(src: string) {
    try {
      audioRef.current?.pause();
    } catch {
      /* noop */
    }
    const a = new Audio(src);
    audioRef.current = a;
    void a.play().catch(() => {});
  }

  function stopReplay() {
    stopRef.current = true;
    try {
      audioRef.current?.pause();
    } catch {
      /* noop */
    }
  }

  async function replay() {
    if (busy) return;
    setBusy(true);
    stopRef.current = false;
    setMessages([]);

    for (let idx = 0; idx < DEMO_TURNS.length; idx++) {
      const turn = DEMO_TURNS[idx];
      if (stopRef.current) break;
      setMessages((p) => [
        ...p,
        { id: newId(), role: "user", text: turn.user, thinking: [], artifacts: [] },
      ]);
      await sleep(450);

      const aid = newId();
      setMessages((p) => [
        ...p,
        { id: aid, role: "assistant", text: "", thinking: [], artifacts: [], streaming: true },
      ]);

      for (const th of turn.thinking) {
        if (stopRef.current) break;
        await sleep(420);
        patch(aid, (m) => ({ ...m, thinking: [...m.thinking, th] }));
      }
      await sleep(220);
      patch(aid, (m) => ({ ...m, intent: turn.intent, polishRole: turn.polishRole }));

      const tokens = turn.answer.match(/\S+\s*/g) || [turn.answer];
      for (const t of tokens) {
        if (stopRef.current) break;
        await sleep(26);
        patch(aid, (m) => ({ ...m, text: m.text + t }));
      }

      const arts = turnArtifacts(turn);
      if (arts.length) {
        await sleep(160);
        patch(aid, (m) => ({ ...m, artifacts: arts }));
      }

      const src = turn.audioClip ?? demoVoiceSrc(idx);
      patch(aid, (m) => ({ ...m, audioSrc: src, streaming: false }));
      if (!stopRef.current) playClip(src);
      await sleep(900);
    }

    setMessages((prev) => prev.map((m) => ({ ...m, streaming: false })));
    setBusy(false);
  }

  function reset() {
    stopReplay();
    setBusy(false);
    setMessages(staticMessages());
  }

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header */}
      <header className="border-b border-border/50 bg-card/30 backdrop-blur-xl sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-accent/30 to-primary/30 flex items-center justify-center">
              <FlaskConical className="w-5 h-5 text-accent" />
            </div>
            <div>
              <h1 className="text-lg font-semibold bg-gradient-to-r from-accent to-primary bg-clip-text text-transparent">
                Djezzy AI — UI Demo
              </h1>
              <div className="flex items-center gap-2 text-xs text-muted-foreground">
                <Badge variant="accent">fake data · no backend</Badge>
                voice = reference recordings
              </div>
            </div>
          </div>
          <Link href="/" title="Back to the live app">
            <Button variant="ghost" size="sm" className="text-muted-foreground hover:text-primary gap-1.5">
              <ArrowLeft className="w-4 h-4" /> Live app
            </Button>
          </Link>
        </div>
      </header>

      {/* Controls */}
      <div className="border-b border-border/50 bg-card/40">
        <div className="max-w-5xl mx-auto px-4 py-3 flex items-center gap-2">
          {busy ? (
            <Button size="sm" variant="destructive" onClick={stopReplay} className="gap-1.5">
              <Square className="w-3.5 h-3.5 fill-current" /> Stop
            </Button>
          ) : (
            <Button size="sm" onClick={replay} className="gap-1.5">
              <Play className="w-4 h-4" /> Replay (streaming)
            </Button>
          )}
          <Button size="sm" variant="outline" onClick={reset} className="gap-1.5" disabled={busy}>
            <RotateCcw className="w-4 h-4" /> Reset
          </Button>
          <span className="text-xs text-muted-foreground ml-auto">
            {busy
              ? "Streaming the canned conversation…"
              : "Showing the full conversation · press a message's ▶ to hear the voice"}
          </span>
        </div>
      </div>

      {/* Conversation */}
      <div className="flex-1 flex flex-col max-w-5xl w-full mx-auto px-4 py-6 min-h-0">
        <div ref={scrollRef} className="flex-1 overflow-y-auto space-y-4 min-h-0">
          {messages.map((m) => (
            <MessageBubble key={m.id} m={m} cfg={DUMMY_CFG} />
          ))}
        </div>
      </div>
    </div>
  );
}
