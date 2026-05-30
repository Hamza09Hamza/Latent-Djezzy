"""v6/speech.py — The voice layer: STT in front, TTS at the back.

The text pipeline is unchanged; this module wraps it with speech:

    audio question ──▶ STT (faster-whisper) ──▶ text ──▶ [graph] ──▶
    polished answer (token stream) ──▶ TTS (XTTS-v2) ──▶ spoken audio

Two design points worth knowing:

  - **STT** is faster-whisper (CTranslate2). `large-v3` is the strongest
    French/English/Arabic model; on GPU it runs in float16, on CPU it
    falls back to int8 so it still works for local smoke tests.

  - **TTS** is Coqui XTTS-v2, driven through its *streaming* inference so
    speech starts before the whole answer is written. The polisher yields
    tokens; `sentence_buffer` groups them into speakable sentences, and
    each sentence is synthesised as it completes. Time-to-first-audio is
    therefore one sentence, not the whole paragraph.

Both models follow the project's lazy-singleton pattern (`get_stt`,
`get_tts`) so they load once and stay resident.

Colab install (see requirements.txt / the notebook):
    pip install faster-whisper coqui-tts soundfile
XTTS-v2 needs the Coqui model licence agreed non-interactively — this
module sets COQUI_TOS_AGREED=1 on import.
"""

from __future__ import annotations
import os
import re
import time
from typing import Iterable, Iterator

import torch

from .config import V6Config

# XTTS-v2 ships under the Coqui Public Model Licence; agreeing here keeps the
# loader from blocking on an interactive y/n prompt in a notebook.
os.environ.setdefault("COQUI_TOS_AGREED", "1")


# ── language helper ───────────────────────────────────────────────────────
def language_for(text: str) -> str:
    """Map free text to an XTTS / Whisper language code: 'fr', 'ar', or 'en'.

    Delegates to slm.lang_code — the single source of truth — so the spoken
    language always matches the language the chat persona and the off-topic
    deflection were written in.
    """
    from .slm import lang_code
    return lang_code(text)


# ── speakable normalization ─────────────────────────────────────────────────
# XTTS reads raw text literally: "DZD" becomes the letters "D-Z-D", "%" is hit
# or miss, and a long grouped number like "1,087,355,290.78" is read digit by
# digit for ten seconds. speakable() rewrites a sentence into what it should
# SOUND like just before synthesis: currency codes and symbols become words,
# artifact/path lines are dropped, and any long number that slipped past the
# deterministic formatter (numfmt) is collapsed to a scale phrase as a net.
_CURRENCY_WORD = {"en": "dinars", "fr": "dinars", "ar": "دينار"}
_PERCENT_WORD = {"en": " percent", "fr": " pour cent", "ar": " بالمئة"}
# A line that is purely an artifact note (chart/report/email path) — never spoken.
_ARTIFACT_LINE = re.compile(
    r"^\s*(?:[📊📄📧🎙🔊]|chart saved|report saved|email (?:draft|saved)|saved\s*[→:-])",
    re.IGNORECASE)
_INLINE_TAG = re.compile(r"\[(?:chart|email[:\s]\w*|report|draft)\]", re.IGNORECASE)
# A grouped or long bare integer (with optional decimals): 1,087,355,290.78
_LONG_NUM = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{7,}(?:\.\d+)?")


def _collapse_long_numbers(text: str, lang: str) -> str:
    """Safety net: turn any 7-digit-or-grouped number ≥ 1,000,000 into a scale
    phrase ("1.09 billion") so TTS never reads a number digit by digit."""
    from .numfmt import humanize

    def repl(m: re.Match) -> str:
        try:
            value = float(m.group(0).replace(",", ""))
        except ValueError:
            return m.group(0)
        if abs(value) < 1_000_000:
            return m.group(0)          # short enough to read as-is
        return humanize(value, None, lang)

    return _LONG_NUM.sub(repl, text)


def speakable(text: str, lang: str = "en") -> str:
    """Rewrite one sentence into a clean, TTS-friendly form.

    Drops artifact/path lines, spells out currency codes and the percent sign,
    and collapses any stray long number. Returns "" for lines that are pure
    artifact notes (so the caller skips them)."""
    if not text or not text.strip():
        return ""
    if _ARTIFACT_LINE.match(text):
        return ""
    out = _INLINE_TAG.sub("", text)
    out = _collapse_long_numbers(out, lang)
    # currency codes → spoken word (DZD everywhere; DA only as a bare token)
    cur = _CURRENCY_WORD.get(lang, "dinars")
    out = re.sub(r"\bDZD\b", cur, out)
    out = re.sub(r"\bDA\b", cur, out)
    # percent sign → word (handles "42.42%" and "42 %")
    out = re.sub(r"\s*%", _PERCENT_WORD.get(lang, " percent"), out)
    return re.sub(r"\s{2,}", " ", out).strip()


# ── sentence buffering for streaming TTS ──────────────────────────────────
# A boundary is sentence punctuation FOLLOWED by whitespace. Requiring the
# trailing space is what stops "491.9" or "432.7 million" from being split at
# the decimal point — the period there is followed by a digit, not a space.
_BOUNDARY_RE = re.compile(r"[.!?…]['\"\)\]]*\s")
_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def sentence_split(text: str) -> list[str]:
    """Split a finished string into sentences for synthesis."""
    return [s.strip() for s in _SPLIT_RE.split(text or "") if s.strip()]


def sentence_buffer(token_iter: Iterable[str],
                    min_chars: int = 30) -> Iterator[str]:
    """Group a token stream into speakable sentences.

    Flushes at the first sentence boundary whose preceding text is at least
    `min_chars` long — so a tiny lead like "Hi." merges with the next
    sentence instead of being spoken as a choppy fragment. Decimals are safe
    (see `_BOUNDARY_RE`). Whatever remains is flushed when the stream ends.
    """
    buf = ""
    for tok in token_iter:
        buf += tok
        while True:
            cut = -1
            for m in _BOUNDARY_RE.finditer(buf):
                if len(buf[:m.end()].strip()) >= min_chars:
                    cut = m.end()
                    break
            if cut == -1:
                break
            yield buf[:cut].strip()
            buf = buf[cut:]
    if buf.strip():
        yield buf.strip()


# ── STT: faster-whisper ───────────────────────────────────────────────────
class STT:
    """faster-whisper speech-to-text. Transcribes an audio file to text."""

    def __init__(self):
        from faster_whisper import WhisperModel

        if torch.cuda.is_available():
            device, compute = "cuda", V6Config.STT_COMPUTE
        else:
            device, compute = "cpu", "int8"   # CPU can't do float16
        self.model = WhisperModel(
            V6Config.STT_MODEL, device=device, compute_type=compute)
        self.device = device

    def transcribe(self, audio_path: str,
                   language: str | None = None) -> dict:
        """Audio file → {text, raw_text, language, language_prob, segments, ms}.

        Whisper is primed with an `initial_prompt` of the real wilaya + KPI
        vocabulary, then the raw transcript is fuzzy-corrected (wilaya names
        + KPI acronyms) — see transcribe.py. `raw_text` keeps the pre-
        correction output so the benchmark can show what was fixed.

        `language` forces a language ('fr'/'en'/'ar'); None auto-detects
        unless V6Config.STT_LANGUAGE is set.
        """
        from .transcribe import build_bias_prompt, correct_transcript

        t0 = time.time()
        lang = language or (V6Config.STT_LANGUAGE or None)
        segments, info = self.model.transcribe(
            audio_path, beam_size=V6Config.STT_BEAM_SIZE, language=lang,
            initial_prompt=build_bias_prompt())
        seg_list, parts = [], []
        for s in segments:                     # generator — consume to run ASR
            parts.append(s.text)
            seg_list.append({"start": round(s.start, 2),
                             "end": round(s.end, 2), "text": s.text})
        raw = "".join(parts).strip()
        return {
            "text": correct_transcript(raw),
            "raw_text": raw,
            "language": info.language,
            "language_prob": round(float(info.language_probability), 3),
            "segments": seg_list,
            "ms": round((time.time() - t0) * 1000, 1),
        }


# ── TTS: Coqui XTTS-v2 (streaming) ────────────────────────────────────────
class TTS:
    """XTTS-v2 text-to-speech with sentence-level streaming.

    Built-in studio speakers are used by default (clean female voices); set
    a reference WAV in config to clone a specific voice instead. Conditioning
    latents are computed once per (language→speaker) and cached.
    """

    def __init__(self):
        self._patch_torch_load()
        from TTS.api import TTS as CoquiTTS

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        api = CoquiTTS(V6Config.TTS_MODEL).to(self.device)
        # Reach through the high-level API to the underlying Xtts model so we
        # can call its streaming inference and reuse its built-in speakers.
        self.xtts = api.synthesizer.tts_model
        self.sample_rate = V6Config.TTS_SAMPLE_RATE
        self._latent_cache: dict = {}          # lang → (gpt_latent, spk_emb)

    @staticmethod
    def _patch_torch_load() -> None:
        """PyTorch >= 2.6 defaults torch.load to weights_only=True, which
        rejects XTTS's pickled config classes. Register them as safe so the
        checkpoint loads. No-op on older torch / if the classes move."""
        try:
            from torch.serialization import add_safe_globals
            from TTS.tts.configs.xtts_config import XttsConfig
            from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
            from TTS.config.shared_configs import BaseDatasetConfig
            add_safe_globals(
                [XttsConfig, XttsAudioConfig, XttsArgs, BaseDatasetConfig])
        except Exception:  # noqa: BLE001 — older torch has no safe-globals API
            pass

    def _latents(self, language: str):
        """Conditioning latents for a language's configured voice (cached)."""
        if language in self._latent_cache:
            return self._latent_cache[language]
        if language == "fr":
            wav, name = V6Config.TTS_SPEAKER_WAV_FR, V6Config.TTS_SPEAKER_FR
        else:
            wav, name = V6Config.TTS_SPEAKER_WAV_EN, V6Config.TTS_SPEAKER_EN
        if wav and os.path.isfile(wav):
            gpt_latent, spk_emb = self.xtts.get_conditioning_latents(
                audio_path=[wav])
        else:
            sp = self.xtts.speaker_manager.speakers[name]
            gpt_latent, spk_emb = sp["gpt_cond_latent"], sp["speaker_embedding"]
        self._latent_cache[language] = (gpt_latent, spk_emb)
        return gpt_latent, spk_emb

    def available_speakers(self) -> list[str]:
        """Built-in studio speaker names (for picking a voice)."""
        try:
            return list(self.xtts.speaker_manager.speakers.keys())
        except Exception:  # noqa: BLE001
            return []

    @torch.no_grad()
    def stream(self, source, language: str = "en") -> Iterator:
        """Yield float32 audio chunks (numpy 1-D) as speech is synthesised.

        `source` is either a finished string or an iterator of text tokens
        (e.g. the polisher's stream). XTTS is fed one sentence at a time and
        itself streams sub-sentence audio chunks, so the first chunk arrives
        about one sentence after generation starts.
        """
        xtts_lang = "fr" if str(language).startswith("fr") else (
            "ar" if str(language).startswith("ar") else "en")
        gpt_latent, spk_emb = self._latents(xtts_lang)
        sentences = (sentence_split(source) if isinstance(source, str)
                     else sentence_buffer(source))
        for sent in sentences:
            # Normalize to a clean, TTS-friendly form (currency words, no paths,
            # no digit-by-digit numbers) right before synthesis.
            sent = speakable(sent, xtts_lang)
            if not sent.strip():
                continue
            for chunk in self.xtts.inference_stream(
                    sent, xtts_lang, gpt_latent, spk_emb,
                    speed=V6Config.TTS_SPEED,
                    temperature=V6Config.TTS_TEMPERATURE,
                    length_penalty=V6Config.TTS_LENGTH_PENALTY,
                    repetition_penalty=V6Config.TTS_REPETITION_PENALTY,
                    top_k=V6Config.TTS_TOP_K,
                    top_p=V6Config.TTS_TOP_P,
                    enable_text_splitting=V6Config.TTS_ENABLE_SPLITTING):
                yield chunk.detach().cpu().numpy()

    def synthesize(self, text: str, language: str = "en",
                   out_path: str | None = None) -> dict:
        """Synthesise a full string to a WAV file (used for fixtures + bench).

        Returns {path, ms, seconds, first_chunk_ms} — first_chunk_ms is the
        time to the first audio chunk, i.e. the streaming latency.
        """
        import numpy as np
        import soundfile as sf

        t0 = time.time()
        chunks, first_ms = [], None
        for ch in self.stream(text, language):
            if first_ms is None:
                first_ms = round((time.time() - t0) * 1000, 1)
            chunks.append(ch)
        audio = (np.concatenate(chunks) if chunks
                 else np.zeros(1, dtype="float32"))
        if out_path is None:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(V6Config.audio_dir(), f"tts_{stamp}.wav")
        sf.write(out_path, audio, self.sample_rate)
        return {
            "path": out_path,
            "ms": round((time.time() - t0) * 1000, 1),
            "first_chunk_ms": first_ms,
            "seconds": round(len(audio) / self.sample_rate, 2),
        }


# ── singletons ─────────────────────────────────────────────────────────────
_stt: STT | None = None
_tts: TTS | None = None


def get_stt() -> STT:
    global _stt
    if _stt is None:
        _stt = STT()
    return _stt


def get_tts() -> TTS:
    global _tts
    if _tts is None:
        _tts = TTS()
    return _tts


# ── convenience module-level functions ─────────────────────────────────────
def transcribe(audio_path: str, language: str | None = None) -> dict:
    return get_stt().transcribe(audio_path, language)


def speak(text: str, language: str | None = None,
          out_path: str | None = None) -> dict:
    """Synthesise `text` to a WAV; auto-detects language when not given."""
    lang = language or language_for(text)
    return get_tts().synthesize(text, lang, out_path)
