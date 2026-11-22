"""
Phase 1 test: our stateless loop (state held OUTSIDE the model) must produce
exactly the same tokens as the official model.generate() streaming path.

Usage:
  python test_phase1_stateless.py                 # real model (downloads weights)
  python test_phase1_stateless.py --small         # same architecture, scaled down, random weights
  python test_phase1_stateless.py --audio samples/clip.wav
"""
import argparse
import sys

import numpy as np
import torch

# Exact-match tests check LOGIC, not rounding: disable GPU fast-math (TF32) and
# batch-size-dependent algorithm choices that can change the last few bits.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

from batched_stream import BatchedStreamer, StreamState, mel_chunks
from common import expand_audio_paths, load_model, load_audio_or_noise


def official_tokens(model, chunks, L):
    out = model.generate(input_features=(c for c in chunks), num_lookahead_tokens=L)
    seq = out.sequences[0].tolist()[1:]                          # drop the start token
    special = {model.config.blank_token_id, model.config.pad_token_id}
    return [t for t in seq if t not in special]


def ours_tokens(model, chunks, L):
    streamer = BatchedStreamer(model, L)
    state = StreamState()
    for c in chunks:
        streamer.step([state], [c])
    return state.tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true", help="scaled-down random-weight model (no download)")
    ap.add_argument("--audio", nargs="*", default=[])
    ap.add_argument("--lookahead", type=int, nargs="*", default=[6, 13, 1])
    args = ap.parse_args()
    args.audio = expand_audio_paths(args.audio)

    model, fe, tok = load_model(args.small)
    audios = [load_audio_or_noise(p) for p in args.audio] or [load_audio_or_noise(None, seed=s, seconds=sec)
                                                              for s, sec in [(1, 4), (2, 9), (3, 13)]]
    failures = 0
    for L in args.lookahead:
        for i, audio in enumerate(audios):
            chunks = list(mel_chunks(fe, audio, L))
            a, b = official_tokens(model, chunks, L), ours_tokens(model, chunks, L)
            ok = a == b
            if not a and not b:
                print(f"lookahead={L:<2} clip {i}: NOT TESTED - 0 tokens (no speech). Use --audio with real speech.")
                failures += 1
                continue
            failures += not ok
            print(f"lookahead={L:<2} clip {i} ({len(audio)/16000:.0f}s, {len(chunks)} chunks): "
                  f"generate={len(a)} tokens, ours={len(b)} tokens -> {'MATCH' if ok else 'MISMATCH'}")
            if tok and ok and i == 0:
                print("   text:", tok.decode(b, skip_special_tokens=True)[:80])
            if not ok:
                first = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
                print(f"   first difference at token {first}: {a[first:first+5]} vs {b[first:first+5]}")
    print("\nPHASE 1:", "PASS" if not failures else f"FAIL ({failures} mismatches or untested clips)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
