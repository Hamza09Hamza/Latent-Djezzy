"use client";

import { useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import Link from "next/link";
import { Mic, Send, Settings, Trash2, Sparkles, Square, FlaskConical } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useAudioRecorder } from "@/hooks/useAudioRecorder";
import { MessageBubble, EmptyState, Message } from "@/components/messages";
import { PcmPlayer } from "@/lib/audio";
import {
  loadConfig,
  saveConfig,
  isConfigured,
  ServerConfig,
  DEFAULT_CONFIG,
} from "@/lib/config";
import { streamAsk, askVoice, checkHealth } from "@/lib/api";

type ConnState = "unknown" | "checking" | "online" | "offline";

function newId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export default function DjezzyAssistant() {
  const [config, setConfig] = useState<ServerConfig>(DEFAULT_CONFIG);
  const [draft, setDraft] = useState<ServerConfig>(DEFAULT_CONFIG);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [conn, setConn] = useState<ConnState>("unknown");

  const recorder = useAudioRecorder();
  const playerRef = useRef<PcmPlayer | null>(null);
  const closeStreamRef = useRef<(() => void) | null>(null);
  const threadRef = useRef<string>("");
  const scrollRef = useRef<HTMLDivElement | null>(null);

  // ── init: load saved config + thread, probe health ────────────────────────
  useEffect(() => {
    const cfg = loadConfig();
    setConfig(cfg);
    setDraft(cfg);
    threadRef.current = `web-${newId()}`;
    if (!playerRef.current) playerRef.current = new PcmPlayer();
    if (!isConfigured(cfg)) setShowSettings(true);
    void probe(cfg);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  async function probe(cfg: ServerConfig) {
    if (!isConfigured(cfg)) {
      setConn("unknown");
      return;
    }
    setConn("checking");
    setConn((await checkHealth(cfg)) ? "online" : "offline");
  }

  function patch(id: string, fn: (m: Message) => Message) {
    setMessages((prev) => prev.map((m) => (m.id === id ? fn(m) : m)));
  }

  // ── text question → WebSocket stream ──────────────────────────────────────
  function handleSend() {
    const q = input.trim();
    if (!q || isProcessing) return;
    if (!isConfigured(config)) {
      toast.error("Set the ngrok URL + API token in Settings first.");
      setShowSettings(true);
      return;
    }
    setInput("");
    setIsProcessing(true);

    const userMsg: Message = {
      id: newId(),
      role: "user",
      text: q,
      thinking: [],
      artifacts: [],
    };
    const botId = newId();
    const botMsg: Message = {
      id: botId,
      role: "assistant",
      text: "",
      thinking: [],
      artifacts: [],
      streaming: true,
    };
    setMessages((prev) => [...prev, userMsg, botMsg]);

    const player = playerRef.current!;
    void player.resume();
    player.reset();

    closeStreamRef.current = streamAsk(config, q, threadRef.current, {
      onThinking: (t) =>
        patch(botId, (m) => ({ ...m, thinking: [...m.thinking, t] })),
      onMeta: (intent, role) =>
        patch(botId, (m) => ({ ...m, intent, polishRole: role })),
      onToken: (t) => patch(botId, (m) => ({ ...m, text: m.text + t })),
      onAudio: (b64, sr) => {
        player.push(b64, sr);
        patch(botId, (m) => ({ ...m, audioChunks: (m.audioChunks || 0) + 1 }));
      },
      onArtifact: (a) =>
        patch(botId, (m) => ({ ...m, artifacts: [...m.artifacts, a] })),
      onAnswer: (text) =>
        patch(botId, (m) => ({ ...m, text: text || m.text })),
      onError: (text) => {
        patch(botId, (m) => ({
          ...m,
          error: true,
          text: m.text || text,
        }));
        toast.error(text);
        if (/unauthorized|server url|colab/i.test(text)) setConn("offline");
      },
      onDone: () => {
        patch(botId, (m) => ({ ...m, streaming: false }));
        setIsProcessing(false);
        closeStreamRef.current = null;
      },
    });
  }

  function handleStop() {
    closeStreamRef.current?.();
    closeStreamRef.current = null;
    setIsProcessing(false);
    setMessages((prev) =>
      prev.map((m) => (m.streaming ? { ...m, streaming: false } : m))
    );
  }

  // ── voice question → /ask_voice (STT + answer + cloned audio) ──────────────
  async function handleVoice(blob: Blob) {
    if (!isConfigured(config)) {
      toast.error("Set the ngrok URL + API token in Settings first.");
      setShowSettings(true);
      return;
    }
    setIsProcessing(true);
    const player = playerRef.current!;
    await player.resume();
    player.reset();
    try {
      const res = await askVoice(config, blob);
      setMessages((prev) => [
        ...prev,
        {
          id: newId(),
          role: "user",
          text: res.question || "(could not transcribe)",
          thinking: [],
          artifacts: [],
        },
        {
          id: newId(),
          role: "assistant",
          text: res.answer,
          thinking: res.thinking || [],
          intent: res.intent,
          artifacts: res.artifacts || [],
          audioChunks: res.audio_pcm16_b64?.length || 0,
        },
      ]);
      if (res.audio_pcm16_b64?.length) {
        player.pushAll(res.audio_pcm16_b64, res.sample_rate);
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Voice request failed";
      toast.error(msg);
      if (/unauthorized/i.test(msg)) setConn("offline");
      setMessages((prev) => [
        ...prev,
        {
          id: newId(),
          role: "assistant",
          text: msg,
          thinking: [],
          artifacts: [],
          error: true,
        },
      ]);
    } finally {
      setIsProcessing(false);
    }
  }

  async function handleMicClick() {
    if (recorder.isRecording) {
      recorder.stopRecording();
      return;
    }
    try {
      await playerRef.current?.resume(); // unlock audio inside the gesture
      await recorder.startRecording(handleVoice);
    } catch {
      toast.error("Could not access the microphone — check permissions.");
    }
  }

  function clearHistory() {
    setMessages([]);
    threadRef.current = `web-${newId()}`;
  }

  function applySettings() {
    const cfg: ServerConfig = {
      serverUrl: draft.serverUrl.trim(),
      apiToken: draft.apiToken.trim(),
    };
    setConfig(cfg);
    saveConfig(cfg);
    toast.success("Saved — pointing at the new endpoint.");
    void probe(cfg);
  }

  const connDot =
    conn === "online"
      ? "bg-primary"
      : conn === "offline"
      ? "bg-destructive"
      : conn === "checking"
      ? "bg-accent animate-pulse"
      : "bg-muted-foreground";
  const connLabel =
    conn === "online"
      ? "Colab online"
      : conn === "offline"
      ? "Offline"
      : conn === "checking"
      ? "Checking…"
      : "Not configured";

  return (
    <div className="min-h-screen bg-background flex flex-col">
      {/* Header */}
      <header className="border-b border-border/50 bg-card/30 backdrop-blur-xl sticky top-0 z-10">
        <div className="max-w-5xl mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-primary/30 to-accent/30 flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h1 className="text-lg font-semibold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                Djezzy AI — LatentMind V6
              </h1>
              <button
                onClick={() => probe(config)}
                className="flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors"
                title="Click to re-check the backend"
              >
                <span className={`w-2 h-2 rounded-full ${connDot}`} />
                {connLabel}
              </button>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Link href="/demo" title="Open the no-backend demo">
              <Button
                variant="ghost"
                size="sm"
                className="text-muted-foreground hover:text-accent gap-1.5"
              >
                <FlaskConical className="w-4 h-4" /> Demo
              </Button>
            </Link>
            <Button
              variant="ghost"
              size="icon"
              onClick={() => setShowSettings((s) => !s)}
              className="text-muted-foreground hover:text-primary"
            >
              <Settings className="w-5 h-5" />
            </Button>
            {messages.length > 0 && (
              <Button
                variant="ghost"
                size="icon"
                onClick={clearHistory}
                className="text-muted-foreground hover:text-destructive"
              >
                <Trash2 className="w-5 h-5" />
              </Button>
            )}
          </div>
        </div>
      </header>

      {/* Settings panel */}
      {showSettings && (
        <div className="border-b border-border/50 bg-card/50 backdrop-blur-xl">
          <div className="max-w-5xl mx-auto px-4 py-4 space-y-3">
            <p className="text-xs text-muted-foreground">
              Paste the ngrok URL + token the Colab{" "}
              <span className="text-foreground font-medium">Serve</span> cell
              printed. Saved locally — overrides{" "}
              <code className="text-accent">lib/config.ts</code> /{" "}
              <code className="text-accent">.env.local</code>.
            </p>
            <div className="grid gap-3 sm:grid-cols-2">
              <div>
                <Label htmlFor="url" className="text-xs text-muted-foreground">
                  Server URL (ngrok)
                </Label>
                <Input
                  id="url"
                  value={draft.serverUrl}
                  onChange={(e) =>
                    setDraft((d) => ({ ...d, serverUrl: e.target.value }))
                  }
                  placeholder="https://xxxx.ngrok-free.app"
                  className="mt-1 font-mono text-xs"
                />
              </div>
              <div>
                <Label htmlFor="tok" className="text-xs text-muted-foreground">
                  API token
                </Label>
                <Input
                  id="tok"
                  value={draft.apiToken}
                  onChange={(e) =>
                    setDraft((d) => ({ ...d, apiToken: e.target.value }))
                  }
                  placeholder="V6_API_TOKEN"
                  className="mt-1 font-mono text-xs"
                />
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Button onClick={applySettings} size="sm">
                Save &amp; connect
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => probe(config)}
              >
                Test connection
              </Button>
              <span className="text-xs text-muted-foreground ml-auto flex items-center gap-1.5">
                <span className={`w-2 h-2 rounded-full ${connDot}`} />
                {connLabel}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Conversation */}
      <div className="flex-1 flex flex-col max-w-5xl w-full mx-auto px-4 py-6 min-h-0">
        <div
          ref={scrollRef}
          className="flex-1 overflow-y-auto mb-6 space-y-4 min-h-0"
        >
          {messages.length === 0 ? (
            <EmptyState />
          ) : (
            messages.map((m) => <MessageBubble key={m.id} m={m} cfg={config} />)
          )}
        </div>

        {/* Composer */}
        <div className="flex items-end gap-2">
          <div className="flex-1 relative">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              disabled={recorder.isRecording}
              placeholder="Ask in French, English, or Darija… e.g. Compare le revenu net entre Alger et Oran"
              className="pr-12 h-12"
            />
            {isProcessing ? (
              <Button
                size="icon"
                variant="ghost"
                onClick={handleStop}
                className="absolute right-1 top-1 h-10 w-10 text-destructive"
                title="Stop"
              >
                <Square className="w-4 h-4 fill-current" />
              </Button>
            ) : (
              <Button
                size="icon"
                variant="ghost"
                onClick={handleSend}
                disabled={!input.trim()}
                className="absolute right-1 top-1 h-10 w-10 text-primary"
                title="Send"
              >
                <Send className="w-5 h-5" />
              </Button>
            )}
          </div>

          {/* Mic */}
          <div className="relative">
            {recorder.isRecording && (
              <div className="absolute inset-0 rounded-full bg-destructive/30 pulse-ring" />
            )}
            <Button
              size="icon"
              onClick={handleMicClick}
              disabled={isProcessing && !recorder.isRecording}
              className={`relative h-12 w-12 rounded-full shadow-lg ${
                recorder.isRecording
                  ? "bg-gradient-to-br from-destructive to-destructive/80 glow-effect"
                  : "bg-gradient-to-br from-primary to-accent"
              }`}
              title={recorder.isRecording ? "Stop & send" : "Hold a question by voice"}
            >
              {recorder.isRecording ? (
                <div className="w-4 h-4 bg-background rounded-sm" />
              ) : (
                <Mic className="w-5 h-5 text-background" />
              )}
            </Button>
          </div>
        </div>

        <div className="h-5 mt-2 text-center text-xs text-muted-foreground">
          {recorder.isRecording
            ? `Recording ${recorder.recordingTime}s — tap the square to send`
            : isProcessing
            ? "Working… reasoning, querying, and synthesizing the voice"
            : "Enter to send · mic for voice · the answer streams with its reasoning + voice"}
        </div>
      </div>
    </div>
  );
}
