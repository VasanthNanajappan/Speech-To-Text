"""
Phase 2 test: many users through ONE model call must give each user exactly
the tokens they'd get alone. (Solo results use our Phase 1 loop, which is
proven identical to model.generate().)

Scenarios
  A  4 users start together, same length, stepped in lockstep
  B  5 users start together, different lengths (users leave the batch)
  C  10 users arriving at random times: a scheduler groups whoever is
     compatible each tick (dynamic batching, exact mode)
  D  Phase 3 preview: same as C, but users with a FULL 5.6 s history are
     merged even though their position counters differ

Usage:
  python test_phase2_batching.py --audio ../samples/*.wav     # real model + real speech (meaningful)
  python test_phase2_batching.py --small                      # scaled-down model, no download

With the real model you MUST pass --audio: synthetic test tones contain no
speech, so the real model (correctly) writes nothing and the test would only
compare empty results.
"""
import argparse
import random
import sys

import torch

# Exact-match tests check LOGIC, not rounding: disable GPU fast-math (TF32) and
# batch-size-dependent algorithm choices that can change the last few bits.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

from batched_stream import BatchedStreamer, StreamState, mel_chunks
from common import expand_audio_paths, load_model, load_audio_or_noise


def solo(model, chunks, L):
    st, s = BatchedStreamer(model, L), StreamState()
    for c in chunks:
        st.step([s], [c])
    return s.tokens


def report(name, results, refs, batch_sizes=None):
    ok = all(results[i] == refs[i] for i in range(len(refs)))
    if sum(len(r) for r in refs) == 0:
        print(f"{name}: NOT TESTED - the model produced 0 tokens (no speech in the audio)")
        return None
    extra = ""
    if batch_sizes:
        extra = f" | model calls {len(batch_sizes)}, avg users per call {sum(batch_sizes)/len(batch_sizes):.2f}, max {max(batch_sizes)}"
    print(f"{name}: {'MATCH' if ok else 'MISMATCH'} for {len(refs)} users ({sum(len(r) for r in refs)} tokens){extra}")
    for i in range(len(refs)):
        if results[i] != refs[i]:
            j = next((k for k, (a, b) in enumerate(zip(results[i], refs[i])) if a != b), min(len(results[i]), len(refs[i])))
            print(f"   user {i}: first difference at token {j}: batched {results[i][j:j+4]} vs solo {refs[i][j:j+4]}")
    return ok


def lockstep(model, all_chunks, L):
    """Everyone starts at tick 0; each tick, users who still have chunks run together."""
    st = BatchedStreamer(model, L)
    states = [StreamState() for _ in all_chunks]
    sizes = []
    for t in range(max(len(c) for c in all_chunks)):
        live = [i for i, c in enumerate(all_chunks) if t < len(c)]
        st.step([states[i] for i in live], [all_chunks[i][t] for i in live])
        sizes.append(len(live))
    return [s.tokens for s in states], sizes


def scheduled(model, all_chunks, starts, L, align, seed=0):
    """Dynamic batching: each tick, users with a chunk ready are grouped and run together.

    A random ~15% of ready users "miss" each tick (network jitter), so their
    caches drift out of step with others, exactly as in real traffic.
    """
    rng = random.Random(seed)
    st = BatchedStreamer(model, L, align_full_caches=align)
    states = [StreamState() for _ in all_chunks]
    pos = [0] * len(all_chunks)
    sizes, tick = [], 0
    while any(p < len(c) for p, c in zip(pos, all_chunks)):
        ready = [i for i in range(len(all_chunks))
                 if starts[i] <= tick and pos[i] < len(all_chunks[i]) and rng.random() > 0.15]
        for group in st.group([states[i] for i in ready]):
            users = [ready[g] for g in group]
            st.step([states[u] for u in users], [all_chunks[u][pos[u]] for u in users])
            for u in users:
                pos[u] += 1
            sizes.append(len(users))
        tick += 1
    return [s.tokens for s in states], sizes


@torch.inference_mode()
def numeric_check(model, all_chunks, L):
    """How close are batched encoder outputs to solo ones (floating-point drift)?"""
    st = BatchedStreamer(model, L)
    batch = [StreamState() for _ in all_chunks]
    solos = [StreamState() for _ in all_chunks]
    worst = 0.0
    for t in range(min(len(c) for c in all_chunks)):
        chunks = [c[t] for c in all_chunks]
        # encoder outputs only, via the same merge path the streamer uses
        from batched_stream import merge_encoder_caches, merge_padding_caches
        x = torch.cat(chunks)
        if t == 0:
            ob = model.get_audio_features(input_features=x, num_lookahead_tokens=L, use_cache=True, output_attention_mask=False)
        else:
            ob = model.get_audio_features(input_features=x,
                                          cache=merge_encoder_caches([s.encoder_cache for s in batch]),
                                          padding_cache=merge_padding_caches([s.padding_cache for s in batch]),
                                          num_lookahead_tokens=L, use_cache=True, output_attention_mask=False)
        st.step(batch, chunks)
        for i, s in enumerate(solos):
            if t == 0:
                os_ = model.get_audio_features(input_features=chunks[i], num_lookahead_tokens=L, use_cache=True, output_attention_mask=False)
            else:
                os_ = model.get_audio_features(input_features=chunks[i],
                                               cache=merge_encoder_caches([s.encoder_cache]),
                                               padding_cache=merge_padding_caches([s.padding_cache]),
                                               num_lookahead_tokens=L, use_cache=True, output_attention_mask=False)
            worst = max(worst, (ob.pooler_output[i] - os_.pooler_output[0]).abs().max().item())
            st.step([s], [chunks[i]])
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--lookahead", type=int, default=6)
    ap.add_argument("--audio", nargs="*", default=[], help="speech recordings (16 kHz wav) to use as users")
    args = ap.parse_args()
    args.audio = expand_audio_paths(args.audio)
    L = args.lookahead
    model, fe, _ = load_model(args.small)

    files = []
    for path in args.audio:
        a = load_audio_or_noise(path)
        if len(a) >= 16000:
            files.append(a)
    if args.audio:
        print(f"Using {len(files)} recordings ({sum(len(a) for a in files) / 16000:.0f} s of speech)\n")
    elif not args.small:
        print("WARNING: no --audio given. Synthetic tones contain no speech, so the real model will\n"
              "write nothing and the token checks will be skipped. Pass --audio ../samples/*.wav\n")

    def get_audio(seed, sec):
        """A different piece of real speech per simulated user, or synthetic audio."""
        if not files:
            return load_audio_or_noise(None, seed=seed, seconds=sec)
        a = files[seed % len(files)]
        n = int(sec * 16000)
        if len(a) <= n:
            return a
        start = random.Random(seed).randint(0, len(a) - n)
        return a[start:start + n]

    clip = lambda seed, sec: list(mel_chunks(fe, get_audio(seed, sec), L))

    results = {}
    # A: same start, same length
    chunks = [clip(10 + i, 9) for i in range(4)]
    refs = [solo(model, c, L) for c in chunks]
    got, sizes = lockstep(model, chunks, L)
    results["A"] = report("A  start together, same length      ", got, refs, sizes)

    # B: same start, different lengths
    chunks = [clip(20 + i, sec) for i, sec in enumerate([3, 5, 8, 11, 7])]
    refs = [solo(model, c, L) for c in chunks]
    got, sizes = lockstep(model, chunks, L)
    results["B"] = report("B  start together, users leave       ", got, refs, sizes)

    # C & D: random arrivals
    rng = random.Random(1)
    chunks = [clip(30 + i, rng.choice([4, 6, 9, 12])) for i in range(10)]
    starts = [rng.randint(0, 12) for _ in chunks]
    refs = [solo(model, c, L) for c in chunks]
    got, sizes = scheduled(model, chunks, starts, L, align=False)
    results["C"] = report("C  random arrivals, exact grouping    ", got, refs, sizes)
    got, sizes = scheduled(model, chunks, starts, L, align=True)
    results["D"] = report("D  Phase 3 preview: merge full caches ", got, refs, sizes)

    worst = numeric_check(model, [clip(40 + i, 4) for i in range(3)], L)
    print(f"\nLargest difference between batched and solo encoder outputs: {worst:.2e}")

    core = [results[k] for k in "ABC"]
    if any(r is None for r in core):
        print("\nPHASE 2: NOT TESTED - no tokens were produced. Run with --audio and real speech.")
        sys.exit(2)
    passed = all(core)
    print("\nPHASE 2:", "PASS" if passed else "FAIL")
    print("Phase 3 preview (D):", "merging full caches is exact" if results["D"] else "merging full caches is NOT exact")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
