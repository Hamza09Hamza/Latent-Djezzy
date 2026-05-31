// ============================================================================
//  PcmPlayer — gapless playback of the server's PCM16 audio chunks.
//
//  The server streams the cloned voice as base64 little-endian PCM16. We decode
//  each chunk into an AudioBuffer and schedule it to start exactly where the
//  previous one ended (playAt), so there are no clicks or gaps between chunks.
//  (Ported from the inline browser client in v6/server.py.)
// ============================================================================

export class PcmPlayer {
  private ctx: AudioContext | null = null;
  private playAt = 0;

  private ensure(): AudioContext {
    if (!this.ctx) {
      const Ctor =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext })
          .webkitAudioContext;
      this.ctx = new Ctor();
      this.playAt = this.ctx.currentTime;
    }
    return this.ctx;
  }

  /** Call from a user gesture (click) so the browser allows audio. */
  async resume(): Promise<void> {
    const ctx = this.ensure();
    if (ctx.state === "suspended") await ctx.resume();
  }

  /** Reset the playback cursor to "now" — call before a new answer. */
  reset(): void {
    const ctx = this.ensure();
    this.playAt = ctx.currentTime;
  }

  /** Decode + schedule one base64 PCM16 chunk at the given sample rate. */
  push(b64: string, sampleRate: number): void {
    const ctx = this.ensure();
    const raw = atob(b64);
    const n = raw.length >> 1;
    if (n === 0) return;
    const buf = ctx.createBuffer(1, n, sampleRate);
    const ch = buf.getChannelData(0);
    for (let i = 0; i < n; i++) {
      let s = raw.charCodeAt(2 * i) | (raw.charCodeAt(2 * i + 1) << 8);
      if (s > 32767) s -= 65536;
      ch[i] = s / 32768;
    }
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    const t = Math.max(ctx.currentTime, this.playAt);
    src.start(t);
    this.playAt = t + buf.duration;
  }

  /** Push a whole batch of chunks (the /ask_voice REST path). */
  pushAll(chunks: string[], sampleRate: number): void {
    for (const c of chunks) this.push(c, sampleRate);
  }
}
