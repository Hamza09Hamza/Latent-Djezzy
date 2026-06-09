# 07 — The Voice Layer

> Code: `v6/speech.py` (STT + TTS + `speakable`), `v6/transcribe.py` (STT priming +
> fuzzy correction).

The voice layer wraps the text pipeline with a transcription front-end and a
synthesis back-end. `audio → STT → text → [the graph] → polished answer → TTS →
spoken audio`. The reasoning pipeline doesn't know or care whether the input was
spoken — voice is pure I/O around the text core.

---

## 1. STT — faster-whisper (`speech.STT`, `transcribe.py`)

- **Model:** faster-whisper **`large-v3`** (CTranslate2 backend). `float16` on GPU;
  `int8` / `int8_float16` fallback on CPU or to save ~1.5 GB VRAM on a T4.
- **Domain priming:** `transcribe()` primes Whisper with an `initial_prompt` seeded
  with **wilaya names + KPI terms**, biasing recognition toward the vocabulary it
  will actually hear ("Oum El Bouaghi," "ARPU," "churn").
- **Fuzzy correction:** the transcript is post-corrected against the known
  wilaya/KPI vocabulary (`transcribe.py`) — a mis-heard "Tizi Wuzu" snaps back to
  "Tizi Ouzou." `raw_text` keeps the **pre-correction** transcript so the benchmark
  can compute an honest WER.

This matters because Algerian wilaya names and telecom jargon are exactly the words
a generic ASR model gets wrong, and a single mis-heard wilaya silently changes the
SQL filter.

---

## 2. TTS — Coqui XTTS-v2 (`speech.TTS`)

- **Model:** Coqui **XTTS-v2**, driven through its **streaming** inference so
  speech starts ~one sentence after generation begins (it doesn't wait for the whole
  answer).
- **Sentence buffering:** `sentence_buffer()` groups the polisher's token stream
  into speakable sentences. A boundary is sentence punctuation **followed by
  whitespace**, so "253.4 million" is never split at the decimal.
- **Conditioning latents** are computed once per language and cached.
- **Voice cloning:** the default built-in voice is `Claribel Dervla` (clearer than
  the old `Daisy Studious`). Setting `V6_TTS_SPEAKER_WAV_EN`/`_FR` to a reference
  WAV **clones that voice** — the single biggest quality lever.
- **Quality knobs:** `V6_TTS_TEMPERATURE` (0.6 — lower = steadier),
  `V6_TTS_REPETITION_PENALTY` (2.5 — reduces slurring/artifacts), `TTS_TOP_K` /
  `TTS_TOP_P`, `TTS_ENABLE_SPLITTING`, `TTS_SPEED`.

> XTTS-v2 is a *local* model and won't match a hosted service like ElevenLabs. The
> closest path is a reference-WAV clone of a professional voice, or swapping the TTS
> backend behind the `get_tts()` seam.

---

## 3. `speakable()` — the spoken-text normalizer

XTTS reads text **literally**: "DZD" becomes "D-Z-D," "%" is hit or miss, and a file
path in a chart note gets read aloud. So before synthesis, each sentence is
rewritten into **what it should sound like** (applied per sentence inside the TTS
stream, so the on-screen text is unchanged):

| Transform | Example |
|---|---|
| **Currency → words** | `DZD` / `DA` → "dinars" (en/fr), "دينار" (ar) |
| **Percent → words** | `%` → " percent" / " pour cent" / " بالمئة" |
| **Drop artifact lines** | a pure chart/report/email note or file path (`📊 Chart saved: …`, `[chart]`) → `""` (skipped) |
| **Long-number safety net** | any 7-digit / comma-grouped number ≥ 1,000,000 that escaped `numfmt` → collapsed to a scale phrase ("1.09 billion") |

That last rule is the voice-side partner of the `numfmt` trust boundary: even if a
raw long number reaches the TTS layer, it is never read digit-by-digit. (See
[Data, Schema & Numbers](06-data-schema-numbers.md).)

---

## 4. Language consistency

`speech.language_for(text)` delegates to `slm.lang_code` — the **single source of
truth** for language (`ar` / `fr` / `en`). The written answer, the persona, the
off-topic deflection, and the spoken voice all derive their language from the same
function, so the voice never speaks French over an English answer. The voice for
each language is independently configurable (`V6_TTS_SPEAKER_*` / `_WAV_*`).

---

## 5. The end-to-end voice path

```
mic → recorded audio
   │  STT (faster-whisper large-v3, primed + fuzzy-corrected)
   ▼
transcript  ──►  [the V6 graph: brain → rag/sql/… → communicator]
   │
   ▼  polished answer (numbers already frozen by numfmt)
sentence_buffer() groups tokens into sentences
   │  speakable(sentence, lang): DZD→dinars, %→percent, drop paths, collapse longs
   ▼
XTTS-v2 streaming synthesis (cloned voice, per-language latents cached)
   ▼
spoken audio (starts ~1 sentence behind the text)
```

In the server, these audio chunks stream to the browser as PCM16 over a WebSocket,
~one sentence behind the streamed text. See [Serving & Frontend](08-serving-frontend.md).

→ Next: [Serving & Frontend](08-serving-frontend.md).
