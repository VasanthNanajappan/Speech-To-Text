# Batching Nemotron Speech Streaming — Phases 0 to 4

## The idea in one sentence

`model.generate()` stores each user's state inside the call (and as flags on the model
object), so one model copy can serve only one user. We moved every piece of per-user state
**outside** the model, which turns it into a pure function `(chunk, state) → (tokens, new state)`,
so many users can be stacked into **one** model call.

## What a user's state is

| State | Shape | Notes |
|---|---|---|
| Attention keys/values, per layer | `[B, heads, ≤70, dim]` | Sliding window: grows 7 frames per chunk, caps at 70 (5.6 s) |
| Position counter, per layer | int | Keeps growing after the cap |
| Convolution history, per conv layer | `[B, channels, kernel−1]` | Fixed size |
| Decoder LSTM | hidden/cell `[layers, B, dim]`, output `[B, 1, dim]` | Fixed size |

Everything has a batch dimension, so batching = concatenate on it, run once, slice back.

## Files

| File | Purpose |
|---|---|
| `batched_stream.py` | `StreamState` (per-user state), cache merge/split, `BatchedStreamer.step()` (batched encoder + batched RNN-T greedy decoding), `group()` (who can share a batch) |
| `common.py` | Loads the real model, or the same architecture scaled down with random weights for testing |
| `bench_phase0.py` | Phase 0: speed per user at batch sizes 1…32 |
| `test_phase1_stateless.py` | Phase 1: our loop must equal `model.generate()` token for token |
| `test_phase2_batching.py` | Phase 2: batched users must equal their solo results token for token |
| `scheduler.py` | Phase 3: `TickScheduler` — aligned starts, catch-up for late chunks, full-cache merging, max batch |
| `test_phase3_arrivals.py` | Phase 3: same traffic through the Phase 2 approach and the scheduler; correctness, batching, delay, live check |
| `batched_engine.py` | Phase 4: `nemotron-batched` engine for `server.py` — one shared model, a scheduler thread ticking every chunk |
| `test_phase4_engine.py` | Phase 4: concurrent users, live updates, busy, cancel, empty, and the real server over WebSockets |

## Results so far (test machine: 1 CPU core, scaled-down model, random weights)

**Phase 1 — stateless loop: PASS.** 9/9 exact matches vs `model.generate()` across latency
settings 0.16 s, 0.56 s and 1.12 s, on 4–13 s clips (covering the 5.6 s cache cap). The test
model was calibrated so all three decoding paths occur: blank → next frame, single token,
and the 10-symbol forced advance.

**Phase 2 — batching: PASS.** 12/12 scenarios exact at all three latency settings:

| Scenario | Result |
|---|---|
| A. 4 users start together, lockstep | exact, 4 users per call |
| B. 5 users start together, leave at different times | exact |
| C. 10 users arriving at random times + 15% jitter, exact grouping | exact, but only ~1.3 users per call |
| D. Phase 3 preview: also merge users with full 5.6 s histories but different counters | exact, ~1.7 users per call (up to 6) |

Batched vs solo encoder outputs differ by at most ~5×10⁻⁷ (floating-point rounding).

**Phase 0 — speed (scaled-down model, 1 CPU core):**

| users | per call | per user | speedup |
|---|---|---|---|
| 1 | 6.9 ms | 6.90 ms | 1.0× |
| 4 | 10.1 ms | 2.53 ms | 2.7× |
| 16 | 26.5 ms | 1.66 ms | 4.2× |

⚠️ This is a pipeline check, not a real measurement: a tiny model is dominated by fixed
per-call overhead, which batching removes easily. The real 600M model does far more math per
call, so expect less gain on a CPU and more on a GPU. **Run it on your hardware.**

## Run on your machine (real model)

```bash
cd batching
python test_phase1_stateless.py                      # must print PHASE 1: PASS
python test_phase1_stateless.py --audio ../samples/*.wav   # also on your own recordings (prints text)
python test_phase2_batching.py                        # must print PHASE 2: PASS
python bench_phase0.py                                # GPU: try --batches 1 2 4 8 16 32 64
python bench_phase0.py --batches 1 2 4 8              # CPU: keep it small
```

Add `--small` to any of them to run the scaled-down model without downloading anything.

If a test reports a mismatch on a GPU, check the "first difference" line. The tests already
turn off GPU fast-math (TF32). A single flipped token at a near-tie is rounding; a mismatch
from the very first tokens is a real bug — send me the output.

## What we learned that shapes Phase 3

1. **The first 5.6 s is the only hard part.** Users can share a batch only if their caches are
   the same size, and caches grow for the first 10 chunks. Exact grouping works but batches
   poorly with random arrivals (scenario C: ~1.3 users per call).
2. **After 5.6 s, everyone can merge** (scenario D, exact). The counter only shifts positions by
   whole chunks, and both the attention mask and the position encoding are relative.
3. So Phase 3 needs to solve only the warm-up. Options to test, in order:
   - **Start-time alignment:** new users start on the next tick, so everyone who presses the
     mic in the same ~0.5 s window forms one cohort. Exact, adds ≤0.56 s start delay.
   - **Silence pre-fill:** run 5.6 s of silence for a new user so their cache starts full.
     Not exact (the model "hears" silence first); verify with the WER benchmark.
   - **Padding + masking** the short caches so they fit with full ones. Exact in theory, most code.

## Phase 3 — users arriving at random times

`TickScheduler` runs once per tick (one chunk, 0.56 s) and applies three rules:

1. **Aligned starts:** everyone who pressed the mic since the last tick starts as one cohort.
2. **Catch-up:** users furthest behind run first. A user whose chunk arrived late processes it,
   catches up, and rejoins their cohort within the same tick.
3. **Full-cache merging:** after 5.6 s, every user merges with every other full user.

Results (scaled-down model, ~15% late chunks, users arriving over 10 ticks, every run exact):

| Users | Starting per tick | Phase 2 users/call | Phase 3 users/call | Delayed chunks P2 → P3 |
|---|---|---|---|---|
| 12 | 1.1 | 1.61 | 1.68 | 87 → 0 |
| 40 | 3.6 | 2.56 | 3.18 | 308 → 0 |
| 100 | 9.1 | 5.42 | 6.87 | 801 → 0 |

**What we learned**

- **Zero added delay.** In the Phase 2 approach, a late user could never catch up, so about half
  of all chunks waited an extra tick (+0.56 s latency that never went away). Catch-up removes it.
- **Batching grows with traffic.** More users per tick → bigger cohorts → bigger batches.
- **Warm-up is the remaining limit.** Voice prompts are short (~9 s here) and warm-up is 5.6 s,
  so ~60% of all work happens while caches are still growing. During warm-up only users who
  started in the SAME tick can share a call — a property of the cache math, not the scheduler.

Run it:

```bash
python test_phase3_arrivals.py --audio ../samples                 # real model, light traffic (~5 min on CPU)
python test_phase3_arrivals.py --small --users 100                # heavy traffic, scaled-down model
```

## What's not done yet

- **Phase 3b (optional):** batch warm-up users at *different* stages by padding short caches and
  masking the padding in attention. Exact in theory; needs a custom attention mask. This is the
  only way to lift the warm-up limit without changing results. (Silence pre-fill is simpler but
  changes results and adds ~10 chunks of compute per user.)
- **Phase 5:** multi-user load test on a GPU.

## Phase 4 — the batched engine in the server

`server.py` and `client.html` are unchanged. The new engine has the same
`feed / finish / cancel` interface, so it simply appears in the engine dropdown.

```
browser tabs ─WebSocket─► server.py ─feed()─► inbox queue ─► scheduler thread (every tick):
                                                               1. MelChunker: audio → exact-size chunks
                                                               2. TickScheduler.run_tick()   (Phase 3)
                                                               3. send each user their new text
```

- **One model for everyone** instead of one copy per user. Memory no longer grows with users.
- **The server thread never touches the model:** `feed()` only puts audio on a queue.
- **`MelChunker`** turns audio arriving in any piece sizes into exactly the chunks the offline
  slicer makes (verified identical: 16/16 cases, piece sizes from 1 sample to 0.5 s). Memory stays
  flat: after 25 s it holds 0.05 s of audio.
- **Latency tradeoff:** work runs on tick boundaries, so a chunk waits on average half a tick
  (~0.28 s at the default setting) before processing. Smaller `NEMOTRON_LOOKAHEAD` = shorter ticks.
- **A failing tick never kills the engine:** affected users are released and new users can join.

Run the server with it:

```bash
ENGINES=nemotron-batched python server.py                       # Mac/Linux
$env:ENGINES="nemotron-batched"; python server.py               # Windows PowerShell
```

| Variable | Default | Meaning |
|---|---|---|
| `NEMOTRON_BATCH_MAX_USERS` | `4` | Users who can dictate at once (more get "busy") |
| `NEMOTRON_BATCH_MAX` | `32` | Max users in one model call |
| `NEMOTRON_LOOKAHEAD` | `6` | Latency setting; also sets the tick length |

Test results (scaled-down model, three arrival patterns, 5–8 users): all six checks pass —
concurrent users exact (8/8), live updates, busy + slot release, cancel mid-sentence, empty
session, and two browsers at once through the real server.

```bash
python test_phase4_engine.py --small                  # ~1 minute, no download
python test_phase4_engine.py --audio ../samples       # real model, 3 users at real speed
```

On a laptop CPU the real model serves about one live user, so with 3 users the engine falls
behind real time (results stay exact; text just arrives later). Watch `late_ticks_pct` in
`http://localhost:8000/health` to see how often ticks overrun.
