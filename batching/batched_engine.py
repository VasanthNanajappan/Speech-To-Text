"""
Phase 4: a "nemotron-batched" engine for server.py.

Same interface as the engines in engines.py, so server.py and client.html don't change:

    session = engine.new_session(on_text)   # on_text(dict) called from the scheduler thread
    session.feed(float32_audio_16khz)
    text = session.finish()
    session.cancel()

How it differs from the "nemotron" engine
-----------------------------------------
nemotron          one model copy per user (a pool); each user runs separately
nemotron-batched  ONE model for everyone; a scheduler thread runs every tick (one chunk,
                  0.56 s at the default latency) and batches all users with the Phase 3
                  TickScheduler (aligned starts, catch-up, full-cache merging)

Threads
-------
Server (asyncio) thread: feed()/finish()/cancel() only put messages on a queue. Cheap,
never blocks the event loop.
Scheduler thread: owns all model state. Each tick it drains the queue, turns audio into
chunks (MelChunker), runs the TickScheduler, and sends each user their updated text.

Latency tradeoff
----------------
Work happens on tick boundaries, so a finished chunk waits on average half a tick
(~0.28 s at the default setting) before it's processed. That's the price of batching.
A smaller lookahead (shorter chunks) shortens the tick.
"""

from __future__ import annotations

import itertools
import os
import queue
import sys
import threading
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for p in (_HERE, _PARENT):
    if p not in sys.path:
        sys.path.insert(0, p)

from batched_stream import MelChunker          # noqa: E402
from scheduler import TickScheduler            # noqa: E402

SAMPLE_RATE = 16000


class EngineBusy(Exception):
    pass


try:  # use the server's exception type so server.py's "busy" handling works unchanged
    from engines import EngineBusy  # type: ignore  # noqa: F811
except Exception:
    pass


class _Stats:
    """Same fields as engines._Stats, so server.py can report them."""

    def __init__(self):
        self.audio_samples = 0
        self.compute_s = 0.0
        self.first_text_at = None

    def as_dict(self):
        audio_s = self.audio_samples / SAMPLE_RATE
        return {"audio_s": round(audio_s, 2), "compute_ms": int(self.compute_s * 1000),
                "rtf": round(self.compute_s / audio_s, 3) if audio_s else None}


class BatchedSession:
    def __init__(self, engine: "BatchedNemotronEngine", uid: int, on_text):
        self.e = engine
        self.uid = uid
        self.on_text = on_text
        self.stats = _Stats()
        self.chunker = MelChunker(engine.feature_extractor, engine.lookahead)
        self.text = ""
        self.done = threading.Event()
        self.finish_requested = False
        self.finish_sent = False
        self.cancelled = False

    # ---- called from the server thread: only enqueue, never touch model state ----
    def feed(self, audio) -> None:
        if len(audio):
            self.stats.audio_samples += len(audio)
            self.e.inbox.put(("audio", self.uid, audio))

    def finish(self, timeout: float = 60.0) -> str:
        self.e.inbox.put(("finish", self.uid, None))
        if not self.done.wait(timeout):
            self.cancel()
            raise TimeoutError("Transcription did not finish in time")
        return self.text.strip()

    def cancel(self) -> None:
        if not self.done.is_set():
            self.e.inbox.put(("cancel", self.uid, None))
            self.done.wait(10)


class BatchedNemotronEngine:
    name = "nemotron-batched"

    def __init__(self, lookahead: int = 6, max_users: int = 4, max_batch: int = 32,
                 device: str | None = None, small: bool = False, tick_seconds: float | None = None):
        from common import load_model
        self.model, self.feature_extractor, self.tokenizer = load_model(small, device)
        self.lookahead = lookahead
        self.max_users = max_users
        self.sched = TickScheduler(self.model, lookahead, max_batch=max_batch)
        self.chunk_seconds = self.sched.streamer.chunk_frames * 0.01
        self.tick_seconds = tick_seconds or self.chunk_seconds   # tests may run ticks faster
        self.inbox: queue.Queue = queue.Queue()
        self.sessions: dict = {}             # uid -> BatchedSession (scheduler thread only)
        self._ids = itertools.count()
        self._slots = threading.BoundedSemaphore(max_users)
        self._active = 0
        self._count_lock = threading.Lock()
        self._ticks = 0
        self._late_ticks = 0
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._loop, name="batched-scheduler", daemon=True)
        self.thread.start()

    # ---------------- public ----------------
    def info(self):
        st = self.sched.stats()
        return {
            "engine": self.name,
            "device": str(next(self.model.parameters()).device),
            "latency_ms": int(round((self.lookahead + 1) * 80)),
            "tick_ms": int(self.tick_seconds * 1000),
            "workers": self.max_users,
            "active_users": self._active,
            "avg_users_per_call": round(st["avg_users_per_call"], 2),
            "late_ticks_pct": round(100 * self._late_ticks / self._ticks, 1) if self._ticks else 0.0,
        }

    def new_session(self, on_text, acquire_timeout: float = 2.0):
        if not self._slots.acquire(timeout=acquire_timeout):
            raise EngineBusy("All speech slots are busy.")
        with self._count_lock:
            self._active += 1
        s = BatchedSession(self, next(self._ids), on_text)
        self.inbox.put(("add", s.uid, s))
        return s

    def close(self):
        self._stop.set()
        self.thread.join(5)

    # ---------------- scheduler thread ----------------
    def _decode(self, tokens) -> str:
        if self.tokenizer is None:                      # scaled-down test model: show raw ids
            return " ".join(map(str, tokens))
        return self.tokenizer.decode(tokens, skip_special_tokens=True)

    def _release(self, s: BatchedSession):
        if s.done.is_set():
            return
        with self._count_lock:
            self._active -= 1
        self._slots.release()
        s.done.set()

    def _drain_inbox(self):
        while True:
            try:
                kind, uid, payload = self.inbox.get_nowait()
            except queue.Empty:
                return
            if kind == "add":
                self.sessions[uid] = payload
                self.sched.add_user(uid)
                continue
            s = self.sessions.get(uid)
            if s is None:
                continue
            if kind == "audio":
                s.chunker.add(payload)
            elif kind == "finish":
                s.finish_requested = True
                s.chunker.finish()
            elif kind == "cancel":
                self.sched.remove(uid)
                self.sessions.pop(uid, None)
                s.cancelled = True
                self._release(s)

    def _tick(self):
        self._drain_inbox()

        # Audio -> exact-size chunks -> scheduler queues.
        for uid, s in self.sessions.items():
            for chunk in s.chunker.pop():
                self.sched.push(uid, chunk)
            if s.chunker.done and not s.finish_sent and uid in self.sched.users:
                self.sched.finish(uid)
                s.finish_sent = True

        if not self.sched.users:
            return False

        new = self.sched.run_tick()
        for uid, toks in new.items():
            s = self.sessions.get(uid)
            if s is None or not toks:
                continue
            slot = self.sched.users.get(uid) or self.sched.done.get(uid)
            s.text = self._decode(slot.state.tokens)
            if s.stats.first_text_at is None and s.text.strip():
                s.stats.first_text_at = time.perf_counter()
            try:
                s.on_text({"committed": s.text.strip(), "tentative": ""})
            except Exception:
                pass

        for uid in list(self.sched.done):
            slot = self.sched.done.pop(uid)
            s = self.sessions.pop(uid, None)
            if s is None:
                continue
            s.text = self._decode(slot.state.tokens)
            s.stats.compute_s = slot.compute_s
            self._release(s)
        return True

    def _loop(self):
        next_at = time.perf_counter()
        while not self._stop.is_set():
            busy = False
            try:
                busy = self._tick()
            except Exception as exc:                      # never let one bad tick kill the engine
                print(f"[nemotron-batched] tick failed: {exc!r}", file=sys.stderr)
                for s in list(self.sessions.values()):
                    self.sched.remove(s.uid)
                    self._release(s)
                self.sessions.clear()
            now = time.perf_counter()
            if busy:
                self._ticks += 1
                if now - next_at > self.tick_seconds:     # this tick overran its budget
                    self._late_ticks += 1
            next_at += self.tick_seconds
            if next_at < now:                             # fell behind: don't try to "make up" ticks
                next_at = now
            time.sleep(max(0.0, next_at - time.perf_counter()))
