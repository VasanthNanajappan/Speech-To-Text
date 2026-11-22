"""
Speech-to-text engines behind one streaming interface.

Both engines expose the same session API, so the server and the benchmark
can swap them freely:

    session = engine.new_session(on_text)   # on_text(dict) is called from a worker thread
    session.feed(float32_audio_16khz)       # call as often as audio arrives
    text = session.finish()                 # flush and return the final text
    session.cancel()                        # throw everything away

Engines
-------
nemotron : NVIDIA Nemotron Speech Streaming EN 0.6B (cache-aware, RNN-T).
           Reads each chunk of audio ONCE and emits words as it goes.
whisper  : faster-whisper with "LocalAgreement" streaming.
           Re-reads recent audio every STEP seconds and commits a word
           once two passes agree. Kept as the baseline to compare against.
"""

from __future__ import annotations

import os
import queue
import threading
import time

import numpy as np

SAMPLE_RATE = 16000
SILENCE_RMS = 0.006          # below this loudness we treat audio as silence


class EngineBusy(Exception):
    """All model copies are in use; the caller should ask the user to retry."""


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(audio ** 2))) if len(audio) else 0.0


class _Stats:
    """Per-session numbers the server and benchmark report."""

    def __init__(self):
        self.started = time.perf_counter()
        self.audio_samples = 0
        self.compute_s = 0.0
        self.first_text_at = None     # perf_counter() when the first word appeared

    def as_dict(self):
        audio_s = self.audio_samples / SAMPLE_RATE
        return {
            "audio_s": round(audio_s, 2),
            "compute_ms": int(self.compute_s * 1000),
            "rtf": round(self.compute_s / audio_s, 3) if audio_s else None,
        }


# =====================================================================
# Nemotron Speech Streaming (recommended for English, low latency)
# =====================================================================
class NemotronEngine:
    name = "nemotron"

    def __init__(self,
                 model_id: str = "nvidia/nemotron-speech-streaming-en-0.6b",
                 lookahead: int = 6,
                 workers: int = 1,
                 device: str | None = None):
        import torch
        from transformers import AutoModelForRNNT, AutoProcessor, TextStreamer

        self.torch = torch
        self.TextStreamer = TextStreamer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.lookahead = lookahead

        # The processor decides chunk sizes. lookahead (right context, in 80 ms
        # encoder frames) is the latency knob: 13 -> 1.12 s, 6 -> 0.56 s,
        # 1 -> 0.16 s, 0 -> 0.08 s.
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.processor.set_num_lookahead_tokens(lookahead)

        # Generation keeps per-call state on the model object, so one model
        # copy can serve only one live session at a time. `workers` copies
        # allow that many people to dictate at once.
        self.pool: queue.Queue = queue.Queue()
        for _ in range(max(1, workers)):
            model = AutoModelForRNNT.from_pretrained(model_id).to(self.device).eval()
            self.pool.put(model)
        self.workers = max(1, workers)

    def info(self):
        return {
            "engine": self.name,
            "device": self.device,
            "latency_ms": self.processor.streaming_latency_ms,
            "workers": self.workers,
        }

    def new_session(self, on_text, acquire_timeout: float = 2.0):
        try:
            model = self.pool.get(timeout=acquire_timeout)
        except queue.Empty:
            raise EngineBusy("All speech workers are busy.")
        return NemotronSession(self, model, on_text)


class NemotronSession:
    def __init__(self, engine: NemotronEngine, model, on_text):
        self.e = engine
        self.model = model
        self.on_text = on_text
        self.stats = _Stats()

        self.q: queue.Queue = queue.Queue()  # float32 chunks, or None = end of audio
        self.buf = np.zeros(0, dtype=np.float32)
        self.base = 0                        # absolute sample index of buf[0]
        self.ended = False
        self.real_total = 0                  # samples of real (non-padding) audio
        self.wait_s = 0.0                    # time spent waiting for audio (not compute)

        self.text = ""
        self.error: Exception | None = None
        self.cancelled = threading.Event()
        self._released = False
        self._lock = threading.Lock()

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    # ---------- public API ----------
    def feed(self, audio: np.ndarray) -> None:
        if len(audio):
            self.stats.audio_samples += len(audio)
            self.q.put(audio.astype(np.float32, copy=False))

    def finish(self, timeout: float = 60.0) -> str:
        self.q.put(None)
        self.thread.join(timeout)
        self._release()
        if self.error:
            raise self.error
        return self.text.strip()

    def cancel(self) -> None:
        self.cancelled.set()
        self.q.put(None)
        self.thread.join(10)
        self._release()

    # ---------- audio plumbing ----------
    def _ensure(self, end_abs: int) -> None:
        """Block until we have audio up to end_abs; pad with zeros after the end."""
        while self.base + len(self.buf) < end_abs:
            if self.ended:
                pad = end_abs - (self.base + len(self.buf))
                self.buf = np.concatenate([self.buf, np.zeros(pad, dtype=np.float32)])
                return
            t = time.perf_counter()
            item = self.q.get()
            self.wait_s += time.perf_counter() - t
            if item is None or self.cancelled.is_set():
                self.ended = True
                self.real_total = self.base + len(self.buf)
            else:
                self.buf = np.concatenate([self.buf, item])

    def _take(self, start: int, end: int) -> np.ndarray:
        """Slice [start, end) in absolute samples; zero-pad before the start of audio."""
        lead = max(0, -start)
        start = max(start, 0)
        piece = self.buf[start - self.base:end - self.base]
        return np.concatenate([np.zeros(lead, dtype=np.float32), piece]) if lead else piece

    def _trim(self, keep_from: int) -> None:
        """Drop audio we'll never read again so memory stays flat on long dictation."""
        cut = keep_from - self.base
        if cut > SAMPLE_RATE:  # trim in 1 s steps
            self.buf = self.buf[cut:]
            self.base = keep_from

    # ---------- the streaming loop ----------
    def _run(self) -> None:
        P = self.e.processor
        hop = P.feature_extractor.hop_length
        n_fft = P.feature_extractor.n_fft
        step = P.num_mel_frames_per_audio_chunk * hop
        t_start = time.perf_counter()

        try:
            first_n = P.num_samples_first_audio_chunk
            self._ensure(first_n)
            if self.cancelled.is_set() or (self.ended and self.real_total == 0):
                return

            first = P(self._take(0, first_n), sampling_rate=SAMPLE_RATE,
                      is_streaming=True, is_first_audio_chunk=True, return_tensors="pt")
            first = first.to(self.model.device, dtype=self.model.dtype)

            def chunks():
                # First chunk: exactly num_mel_frames_first_audio_chunk frames.
                yield first.input_features[:, :P.num_mel_frames_first_audio_chunk, :]
                mel = P.num_mel_frames_first_audio_chunk
                while True:
                    start = mel * hop - n_fft // 2
                    end = start + P.num_samples_per_audio_chunk
                    # One extra zero-padded chunk after the end flushes the last words.
                    if self.cancelled.is_set() or (self.ended and start >= self.real_total + step):
                        return
                    self._ensure(end)
                    if self.cancelled.is_set() or (self.ended and start >= self.real_total + step):
                        return
                    feats = P(self._take(start, end), sampling_rate=SAMPLE_RATE,
                              is_streaming=True, is_first_audio_chunk=False,
                              return_tensors="pt").input_features
                    yield feats
                    mel += P.num_mel_frames_per_audio_chunk
                    self._trim(start)

            session = self

            class _Streamer(self.e.TextStreamer):
                # Called with each finished word (TextStreamer flushes on spaces).
                def on_finalized_text(self, text, stream_end=False):
                    if text and not session.cancelled.is_set():
                        with session._lock:
                            session.text += text
                            if session.stats.first_text_at is None and session.text.strip():
                                session.stats.first_text_at = time.perf_counter()
                        session.on_text({"committed": session.text.strip(), "tentative": ""})

            streamer = _Streamer(P.tokenizer, skip_special_tokens=True)
            kwargs = dict(first)
            kwargs["input_features"] = chunks()
            kwargs["num_lookahead_tokens"] = self.e.lookahead
            kwargs["streamer"] = streamer
            with self.e.torch.inference_mode():
                self.model.generate(**kwargs)
        except Exception as exc:          # surfaced by finish()
            self.error = exc
        finally:
            self.stats.compute_s = max(0.0, time.perf_counter() - t_start - self.wait_s)

    def _release(self) -> None:
        # Only return the model once its generate() has fully stopped;
        # handing a still-running model to someone else would corrupt both sessions.
        if self._released or self.thread.is_alive():
            return
        self._released = True
        self.e.pool.put(self.model)


# =====================================================================
# Whisper baseline (faster-whisper + LocalAgreement)
# =====================================================================
def _norm_word(w: str) -> str:
    return w.strip().lower().strip(".,!?;:\"'")


class WhisperEngine:
    name = "whisper"

    def __init__(self, model_size: str = "small.en", device: str = "cpu",
                 step_s: float = 0.8, max_sessions: int = 2):
        from faster_whisper import WhisperModel
        compute_type = "int8" if device == "cpu" else "float16"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.model_size = model_size
        self.device = device
        self.step_s = step_s
        self.model_lock = threading.Lock()   # one pass at a time on the CPU
        self.slots = threading.BoundedSemaphore(max(1, max_sessions))
        self.max_sessions = max(1, max_sessions)

    def info(self):
        return {"engine": self.name, "model": self.model_size, "device": self.device,
                "step_ms": int(self.step_s * 1000), "workers": self.max_sessions}

    def new_session(self, on_text, acquire_timeout: float = 2.0):
        if not self.slots.acquire(timeout=acquire_timeout):
            raise EngineBusy("All speech workers are busy.")
        return WhisperSession(self, on_text)


class WhisperSession:
    MAX_BUFFER_S = 15

    def __init__(self, engine: WhisperEngine, on_text):
        self.e = engine
        self.on_text = on_text
        self.stats = _Stats()
        self.audio = np.zeros(0, dtype=np.float32)
        self.offset = 0.0
        self.new_samples = 0
        self.committed: list = []   # (start, end, text) in absolute seconds
        self.tentative: list = []
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self._released = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def feed(self, audio: np.ndarray) -> None:
        with self.lock:
            self.audio = np.concatenate([self.audio, audio.astype(np.float32, copy=False)])
            self.new_samples += len(audio)
            self.stats.audio_samples += len(audio)

    def finish(self, timeout: float = 60.0) -> str:
        self.stop.set()
        self.thread.join(timeout)
        try:
            self._process(final=True)
        finally:
            self._release()
        return self._text(self.committed)

    def cancel(self) -> None:
        self.stop.set()
        self.thread.join(10)
        self._release()

    def _release(self):
        if not self._released:
            self._released = True
            self.e.slots.release()

    @staticmethod
    def _text(words) -> str:
        return "".join(w[2] for w in words).strip()

    def _loop(self):
        while not self.stop.wait(self.e.step_s):
            if self.new_samples >= SAMPLE_RATE * 0.2:
                if self._process():
                    self.on_text({"committed": self._text(self.committed),
                                  "tentative": self._text(self.tentative)})

    def _process(self, final: bool = False) -> bool:
        with self.lock:
            audio = self.audio.copy()
            offset = self.offset
            new = self.new_samples
            self.new_samples = 0

        if len(audio) < SAMPLE_RATE * 0.5 and not final:
            return False
        # Silence guard: Whisper is known to invent words on silence/noise-only audio.
        if final and _rms(audio) < SILENCE_RMS:
            self.committed += self.tentative
            self.tentative = []
            return False
        if not final and new and _rms(audio[-new:]) < SILENCE_RMS and not self.tentative:
            return False

        context = self._text(self.committed[-30:]) or None
        t = time.perf_counter()
        with self.e.model_lock:
            segments, _ = self.e.model.transcribe(
                audio, language="en", beam_size=1, word_timestamps=True,
                condition_on_previous_text=False, initial_prompt=context, vad_filter=False)
            words = [(w.start + offset, w.end + offset, w.word)
                     for s in segments for w in (s.words or [])]
        self.stats.compute_s += time.perf_counter() - t

        cutoff = self.committed[-1][1] if self.committed else self.offset
        words = [w for w in words if (w[0] + w[1]) / 2 > cutoff]
        if final:
            self.committed += words
            self.tentative = []
        else:
            n = 0
            while (n < len(words) and n < len(self.tentative)
                   and _norm_word(words[n][2]) == _norm_word(self.tentative[n][2])):
                n += 1
            self.committed += words[:n]
            self.tentative = words[n:]
        if self.stats.first_text_at is None and (self.committed or self.tentative):
            self.stats.first_text_at = time.perf_counter()
        self._trim()
        return True

    def _trim(self):
        with self.lock:
            if len(self.audio) / SAMPLE_RATE <= self.MAX_BUFFER_S:
                return
            if not self.committed:
                self.committed += self.tentative
                self.tentative = []
            end = self.committed[-1][1] if self.committed else self.offset
            cut = max(0, min(int((end - self.offset) * SAMPLE_RATE), len(self.audio)))
            self.audio = self.audio[cut:]
            self.offset += cut / SAMPLE_RATE


# =====================================================================
# Loading
# =====================================================================
def load_engines(names: list[str]) -> tuple[dict, dict]:
    """Load the requested engines; return (engines, errors) so one failing doesn't stop the other."""
    engines, errors = {}, {}
    for name in names:
        name = name.strip().lower()
        try:
            if name == "nemotron":
                engines[name] = NemotronEngine(
                    lookahead=int(os.getenv("NEMOTRON_LOOKAHEAD", "6")),
                    workers=int(os.getenv("NEMOTRON_WORKERS", "1")),
                    device=os.getenv("NEMOTRON_DEVICE") or None)
            elif name == "nemotron-batched":
                import sys as _sys
                _sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "batching"))
                from batched_engine import BatchedNemotronEngine
                engines[name] = BatchedNemotronEngine(
                    lookahead=int(os.getenv("NEMOTRON_LOOKAHEAD", "6")),
                    max_users=int(os.getenv("NEMOTRON_BATCH_MAX_USERS", "4")),
                    max_batch=int(os.getenv("NEMOTRON_BATCH_MAX", "32")),
                    device=os.getenv("NEMOTRON_DEVICE") or None,
                    small=os.getenv("NEMOTRON_TEST_SMALL") == "1")
            elif name == "whisper":
                engines[name] = WhisperEngine(
                    model_size=os.getenv("WHISPER_MODEL", "small.en"),
                    device=os.getenv("WHISPER_DEVICE", "cpu"),
                    step_s=float(os.getenv("WHISPER_STEP_SECONDS", "0.8")),
                    max_sessions=int(os.getenv("WHISPER_WORKERS", "2")))
            elif name:
                errors[name] = "Unknown engine"
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
    return engines, errors
