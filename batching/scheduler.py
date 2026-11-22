"""
Phase 3: a tick scheduler that keeps users batchable while they arrive at random times.

The problem (from Phase 2, scenario C)
--------------------------------------
Users can share a model call only when their caches are the same size. During
the first 5.6 s caches grow chunk by chunk, so:
  * users who start at different times can't share a batch, and
  * a user whose chunk arrives late falls one step behind their group forever.
Result: ~1.3 users per call.

The fixes
---------
1. Tick-aligned starts. The scheduler runs once per tick (one chunk = 0.56 s).
   Everyone who pressed the mic since the last tick starts together, as a cohort.
2. Catch-up. Within a tick, users who are furthest BEHIND run first. A user whose
   chunk was late has two chunks waiting next tick: it processes the first
   (catching up), then joins its cohort for the second, in the same tick.
3. Full-cache merging. Once a user's history is full (5.6 s), they merge with
   every other full user regardless of start time (proven exact in Phase 2, D).
4. max_batch caps how many users share one call.

Every option can be switched off, so tests can compare against the Phase 2 approach
on the same traffic.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import torch

from batched_stream import BatchedStreamer, StreamState


@dataclass
class _Slot:
    state: StreamState = field(default_factory=StreamState)
    queue: deque = field(default_factory=deque)     # (chunk, arrival_tick)
    start_tick: int | None = None                   # tick of the user's first processed chunk
    finished: bool = False                          # no more audio will arrive
    waits: list = field(default_factory=list)       # ticks each chunk waited after arriving
    compute_s: float = 0.0                          # this user's share of model time


class TickScheduler:
    def __init__(self, model, num_lookahead_tokens: int = 6, max_batch: int = 32,
                 catch_up: bool = True, merge_full: bool = True):
        self.streamer = BatchedStreamer(model, num_lookahead_tokens, align_full_caches=merge_full)
        self.max_batch = max_batch
        self.catch_up = catch_up
        self.tick = 0
        self.users: dict = {}
        self.done: dict = {}                 # finished users, kept for results/stats
        self.call_sizes: list[int] = []      # users per model call
        self.calls_per_tick: list[int] = []
        self.tick_seconds: list[float] = []  # wall time of each busy tick (must be < chunk length to run live)

    # ---------------- inputs ----------------
    def add_user(self, uid) -> None:
        self.users[uid] = _Slot()

    def push(self, uid, chunk) -> None:
        self.users[uid].queue.append((chunk, self.tick))

    def finish(self, uid) -> None:
        """No more audio for this user; they leave once their queue is empty."""
        self.users[uid].finished = True

    def remove(self, uid) -> None:
        """Drop a user immediately (cancelled or disconnected)."""
        self.users.pop(uid, None)
        self.done.pop(uid, None)

    # ---------------- the tick ----------------
    def _deficit(self, slot: _Slot) -> int:
        """How many chunks a user is behind where an on-time user would be."""
        if slot.start_tick is None:
            return 1
        expected = self.tick - slot.start_tick + 1
        return expected - slot.state.chunks_done

    def run_tick(self) -> dict:
        """Process this tick's audio. Returns {uid: new tokens} and removes finished users."""
        new_tokens: dict = {}
        calls = 0
        t_start = time.perf_counter()
        processed_this_tick: set = set()

        while True:
            ready = [u for u, s in self.users.items() if s.queue]
            if not self.catch_up:   # Phase 2 behavior: at most one chunk per user per tick
                ready = [u for u in ready if u not in processed_this_tick]
            if not ready:
                break

            if self.catch_up:
                # Furthest-behind users go first, so laggards catch up and rejoin their cohort.
                worst = max(self._deficit(self.users[u]) for u in ready)
                ready = [u for u in ready if self._deficit(self.users[u]) == worst]

            for group in self.streamer.group([self.users[u].state for u in ready]):
                members = [ready[g] for g in group]
                for i in range(0, len(members), self.max_batch):
                    batch = members[i:i + self.max_batch]
                    slots = [self.users[u] for u in batch]
                    chunks = []
                    for u, s in zip(batch, slots):
                        chunk, arrived = s.queue.popleft()
                        chunks.append(chunk)
                        s.waits.append(self.tick - arrived)
                        if s.start_tick is None:
                            s.start_tick = self.tick
                    out = self.streamer.step([s.state for s in slots], chunks)
                    timing = self.streamer.last_timing
                    share = (timing["encoder_ms"] + timing["decoder_ms"]) / 1000 / len(batch)
                    for u, s, toks in zip(batch, slots, out):
                        new_tokens.setdefault(u, []).extend(toks)
                        s.compute_s += share
                    self.call_sizes.append(len(batch))
                    calls += 1
            processed_this_tick.update(ready)

        self.calls_per_tick.append(calls)
        if calls:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.tick_seconds.append(time.perf_counter() - t_start)
        for u in [u for u, s in self.users.items() if s.finished and not s.queue]:
            self.done[u] = self.users.pop(u)
        self.tick += 1
        return new_tokens

    # ---------------- reporting ----------------
    def stats(self) -> dict:
        sizes = self.call_sizes or [0]
        return {
            "model_calls": len(self.call_sizes),
            "avg_users_per_call": sum(sizes) / max(1, len(self.call_sizes)),
            "max_users_per_call": max(sizes),
        }
