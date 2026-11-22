"""
Phase 3 test: users arriving at random times.

One traffic pattern is generated (users press the mic at random ticks, talk for
random lengths, and ~15% of chunks arrive one tick late), then replayed through:

  Phase 2 approach   group only exactly-matching users, one chunk per user per tick
  Phase 3 scheduler  aligned starts + catch-up + full-cache merging

For each it checks:
  correctness  every user's tokens equal their solo result (must be exact)
  batching     average / max users per model call (higher is better)
  delay        how many ticks chunks waited in the queue (0 is ideal)

Usage:
  python test_phase3_arrivals.py --audio ../samples            # real model + your speech
  python test_phase3_arrivals.py --small                       # scaled-down model, no download
  python test_phase3_arrivals.py --audio ../samples --users 20 --arrival-ticks 15
"""
import argparse
import random
import sys

import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

from batched_stream import BatchedStreamer, StreamState, mel_chunks
from common import expand_audio_paths, load_audio_or_noise, load_model
from scheduler import TickScheduler


def make_traffic(fe, L, files, users, arrival_ticks, jitter, seed):
    """Each user: start tick, their mel chunks, and the tick each chunk arrives."""
    rng = random.Random(seed)
    traffic = []
    for u in range(users):
        seconds = rng.choice([4, 6, 8, 10, 14])
        if files:
            a = files[u % len(files)]
            n = int(seconds * 16000)
            start = rng.randint(0, max(0, len(a) - n))
            audio = a[start:start + n]
        else:
            audio = load_audio_or_noise(None, seed=100 + u, seconds=seconds)
        chunks = list(mel_chunks(fe, audio, L))
        start_tick = rng.randint(0, arrival_ticks)
        arrive, t = [], start_tick
        for j in range(len(chunks)):
            late = j > 0 and rng.random() < jitter      # this chunk shows up one tick late
            arrive.append(max(t, start_tick + j + (1 if late else 0)))
            t = arrive[-1]
        traffic.append({"chunks": chunks, "arrive": arrive})
    return traffic


def solo(model, chunks, L):
    st, s = BatchedStreamer(model, L), StreamState()
    for c in chunks:
        st.step([s], [c])
    return s.tokens


def replay(model, traffic, L, catch_up, merge_full, max_batch):
    sch = TickScheduler(model, L, max_batch=max_batch, catch_up=catch_up, merge_full=merge_full)
    sent = [0] * len(traffic)
    added = set()
    while len(sch.done) < len(traffic):
        for u, tr in enumerate(traffic):
            while sent[u] < len(tr["chunks"]) and tr["arrive"][sent[u]] <= sch.tick:
                if u not in added:
                    sch.add_user(u)
                    added.add(u)
                sch.push(u, tr["chunks"][sent[u]])
                sent[u] += 1
            if u in added and sent[u] == len(tr["chunks"]) and u in sch.users:
                sch.finish(u)
        sch.run_tick()
        if sch.tick > 10_000:
            raise RuntimeError("scheduler did not finish")
    tokens = [sch.done[u].state.tokens for u in range(len(traffic))]
    waits = [w for u in range(len(traffic)) for w in sch.done[u].waits]
    return tokens, sch.stats(), waits, sch.calls_per_tick


def describe(name, tokens, refs, stats, waits):
    exact = all(tokens[i] == refs[i] for i in range(len(refs)))
    late = sum(1 for w in waits if w > 0)
    print(f"{name}")
    print(f"   correctness : {'EXACT for all ' + str(len(refs)) + ' users' if exact else 'MISMATCH'}")
    print(f"   batching    : {stats['model_calls']} model calls, "
          f"{stats['avg_users_per_call']:.2f} users per call on average, max {stats['max_users_per_call']}")
    print(f"   delay       : {late} of {len(waits)} chunks waited extra, worst {max(waits)} tick(s)")
    for i in range(len(refs)):
        if tokens[i] != refs[i]:
            j = next((k for k, (a, b) in enumerate(zip(tokens[i], refs[i])) if a != b),
                     min(len(tokens[i]), len(refs[i])))
            print(f"   user {i}: first difference at token {j}")
    return exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--audio", nargs="*", default=[])
    ap.add_argument("--lookahead", type=int, default=6)
    ap.add_argument("--users", type=int, default=12)
    ap.add_argument("--arrival-ticks", type=int, default=10, help="users arrive spread over this many ticks")
    ap.add_argument("--jitter", type=float, default=0.15, help="fraction of chunks that arrive one tick late")
    ap.add_argument("--max-batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.audio = expand_audio_paths(args.audio)

    model, fe, _ = load_model(args.small)
    L = args.lookahead
    files = [a for a in (load_audio_or_noise(p) for p in args.audio) if len(a) >= 16000]
    if not files and not args.small:
        sys.exit("With the real model, pass --audio with speech recordings (tones produce no tokens).")

    traffic = make_traffic(fe, L, files, args.users, args.arrival_ticks, args.jitter, args.seed)
    total_chunks = sum(len(t["chunks"]) for t in traffic)
    print(f"{args.users} users, arriving over {args.arrival_ticks} ticks, "
          f"{args.jitter:.0%} late chunks, {total_chunks} chunks total\n")

    print("Computing each user's solo result...")
    refs = [solo(model, t["chunks"], L) for t in traffic]
    n_tokens = sum(len(r) for r in refs)
    print(f"   {n_tokens} tokens\n")
    if n_tokens == 0:
        sys.exit("NOT TESTED: the model produced 0 tokens. Use --audio with real speech.")

    t2, s2, w2, _ = replay(model, traffic, L, catch_up=False, merge_full=False, max_batch=args.max_batch)
    ok2 = describe("Phase 2 approach (exact grouping only)", t2, refs, s2, w2)
    print()
    t3, s3, w3, cpt = replay(model, traffic, L, catch_up=True, merge_full=True, max_batch=args.max_batch)
    ok3 = describe("Phase 3 scheduler (aligned starts + catch-up + full merge)", t3, refs, s3, w3)

    gain = s3["avg_users_per_call"] / max(1e-9, s2["avg_users_per_call"])
    fewer = 1 - s3["model_calls"] / max(1, s2["model_calls"])
    print(f"\nPhase 3 vs Phase 2: {gain:.2f}x users per call, {fewer:.0%} fewer model calls")
    busy = [c for c in cpt if c]
    print(f"Model calls per tick (busy ticks): average {sum(busy)/max(1,len(busy)):.1f}, max {max(busy)}")

    print("\nPHASE 3:", "PASS" if ok3 else "FAIL")
    sys.exit(0 if ok3 else 1)


if __name__ == "__main__":
    main()
