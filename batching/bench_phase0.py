"""
Phase 0: is batching worth it on THIS machine?

Runs one streaming step for B users at once (B = 1, 2, 4, 8, ...) with full
5.6 s histories (the steady state of a real conversation) and measures:

  per call     time for one model call serving B users
  per user     per-call time / B           (lower = batching is helping)
  speedup      per-user cost at B=1 / per-user cost at B
  capacity     users one model can keep up with in real time
               = chunk duration / per-user time  (an upper bound: assumes the
                 model does nothing else, ignores network and Python overhead)

Usage:
  python bench_phase0.py                       # real model, auto GPU/CPU
  python bench_phase0.py --batches 1 2 4 8     # fewer sizes (e.g. on CPU)
  python bench_phase0.py --small               # scaled-down test model (pipeline check only)
"""
import argparse
import time

import torch

from batched_stream import BatchedStreamer, StreamState, mel_chunks
from common import load_model, load_audio_or_noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--lookahead", type=int, default=6)
    ap.add_argument("--batches", type=int, nargs="*", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--steps", type=int, default=10, help="timed steps per batch size")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    model, fe, _ = load_model(args.small, args.device)
    L = args.lookahead
    st = BatchedStreamer(model, L)
    chunk_s = st.chunk_frames * 0.01                     # 10 ms per mel frame
    warm_chunks = st.window() // (L + 1) + 1             # enough chunks to fill the 5.6 s history
    audio = load_audio_or_noise(None, seed=0, seconds=chunk_s * (warm_chunks + args.steps + 3) + 1)
    chunks = list(mel_chunks(fe, audio, L))
    dev = next(model.parameters()).device

    print(f"device={dev}  model={'small test' if args.small else 'nemotron 0.6B'}  "
          f"chunk={chunk_s:.2f}s  lookahead={L}\n")
    print(f"{'users':>5} {'per call':>10} {'encoder':>9} {'decoder':>9} {'per user':>10} {'speedup':>8} "
          f"{'capacity':>9} {'GPU mem':>9}")

    base = None
    for B in args.batches:
        try:
            if dev.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            states = [StreamState() for _ in range(B)]
            for t in range(warm_chunks + 1):                 # fill everyone's history (untimed)
                st.step(states, [chunks[t]] * B)
            times, enc, dec = [], [], []
            for t in range(warm_chunks + 1, warm_chunks + 1 + args.steps):
                t0 = time.perf_counter()
                st.step(states, [chunks[t]] * B)
                if dev.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
                enc.append(st.last_timing["encoder_ms"])
                dec.append(st.last_timing["decoder_ms"])
        except torch.cuda.OutOfMemoryError:
            print(f"{B:>5}  out of GPU memory, stopping here")
            break

        times.sort()
        call_ms = times[len(times) // 2] * 1000           # median is robust to hiccups
        per_user = call_ms / B
        base = base or per_user
        mem = f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if dev.type == "cuda" else "-"
        print(f"{B:>5} {call_ms:>8.1f}ms {sorted(enc)[len(enc)//2]:>7.1f}ms {sorted(dec)[len(dec)//2]:>7.1f}ms "
              f"{per_user:>8.2f}ms {base / per_user:>7.1f}x {int(chunk_s * 1000 / per_user):>9} {mem:>9}")

    print("\nRead it like this: if 'per user' keeps falling as users grow, batching multiplies capacity.\n"
          "If it stays flat (common on CPUs), batching won't help much on this machine.")


if __name__ == "__main__":
    main()
