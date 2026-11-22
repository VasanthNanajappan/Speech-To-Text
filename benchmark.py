"""
Benchmark speech-to-text engines on YOUR recordings, plus automatic edge-case tests.

Usage
-----
  python benchmark.py --data samples --engines nemotron,whisper
  python benchmark.py --data samples --engines nemotron,whisper --realtime   # true live latency (slower)

samples/ holds pairs:  anything.wav  +  anything.txt  (exactly what was said)
Record them in the browser with http://localhost:8000/?record=1, or convert any file:
  ffmpeg -i input.m4a -ar 16000 -ac 1 samples/name.wav

What it measures
----------------
WER                 word error rate vs your .txt (lower is better)
RTF                 compute time / audio length (below 1.0 = faster than real time)
first_words_s       how far into the audio the first word appeared (realtime mode)
finalize_ms         time from "user pressed stop" to final text
hallucinated_words  words produced for audio that contains no speech (should be 0)

Edge-case suite (generated automatically from your samples)
------------------------------------------------------------
silence_10s            pressed the mic and stayed silent
noise_10s              background noise, nobody speaking
speech_noise_10db      speech with loud background noise
leading_silence_5s     5 s of silence before the user starts talking
long_90s               90+ seconds of continuous dictation
"""

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from engines import SAMPLE_RATE, load_engines

CHUNK = SAMPLE_RATE // 10   # feed 100 ms at a time, like the browser does


# ---------------------------------------------------------------- audio io
def load_wav(path: Path) -> np.ndarray:
    import soundfile as sf
    audio, rate = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)                       # stereo -> mono
    if rate != SAMPLE_RATE:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(rate, SAMPLE_RATE)
        audio = resample_poly(audio, SAMPLE_RATE // g, rate // g).astype(np.float32)
    return audio


def load_samples(folder: Path):
    samples = []
    for wav in sorted(folder.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if not txt.exists():
            print(f"  skipping {wav.name}: no {txt.name}")
            continue
        samples.append((wav.stem, load_wav(wav), txt.read_text(encoding="utf-8").strip()))
    return samples


# ---------------------------------------------------------------- scoring
def normalize(text: str) -> str:
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def wer(reference: str, hypothesis: str):
    import jiwer
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return None
    return jiwer.wer(ref, hyp if hyp else "")


# ---------------------------------------------------------------- edge cases
def noise(seconds: float, level: float, rng) -> np.ndarray:
    return (rng.standard_normal(int(seconds * SAMPLE_RATE)) * level).astype(np.float32)


def mix_at_snr(speech: np.ndarray, snr_db: float, rng) -> np.ndarray:
    n = rng.standard_normal(len(speech)).astype(np.float32)
    p_speech = np.mean(speech ** 2) + 1e-12
    n *= np.sqrt(p_speech / (10 ** (snr_db / 10)) / (np.mean(n ** 2) + 1e-12))
    out = speech + n
    return (out / max(1.0, np.abs(out).max())).astype(np.float32)


def build_edge_cases(samples):
    rng = np.random.default_rng(0)
    cases = [
        ("silence_10s", noise(10, 0.0003, rng), ""),        # tiny dither, like a real quiet mic
        ("noise_10s", noise(10, 0.03, rng), ""),
    ]
    if samples:
        name, audio, ref = samples[0]
        cases.append(("speech_noise_10db", mix_at_snr(audio, 10, rng), ref))
        cases.append(("leading_silence_5s", np.concatenate([noise(5, 0.0003, rng), audio]), ref))
        long_audio, long_ref, gap = [], [], noise(0.5, 0.0003, rng)
        while sum(len(a) for a in long_audio) < 90 * SAMPLE_RATE:
            for _, a, r in samples:
                long_audio += [a, gap]
                long_ref.append(r)
        cases.append(("long_90s", np.concatenate(long_audio), " ".join(long_ref)))
    return cases


# ---------------------------------------------------------------- one run
def run_one(engine, audio: np.ndarray, realtime: bool):
    session = engine.new_session(lambda update: None, acquire_timeout=120)
    t0 = time.perf_counter()
    for i in range(0, len(audio), CHUNK):
        chunk = audio[i:i + CHUNK]
        session.feed(chunk)
        if realtime:   # wait as if a person were speaking
            target = t0 + (i + len(chunk)) / SAMPLE_RATE
            delay = target - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
    t_stop = time.perf_counter()
    text = session.finish()
    t_end = time.perf_counter()

    audio_s = len(audio) / SAMPLE_RATE
    first = session.stats.first_text_at
    return {
        "text": text,
        "audio_s": round(audio_s, 2),
        "rtf": round(session.stats.compute_s / audio_s, 3) if audio_s else None,
        # Only meaningful in realtime mode; in fast mode it includes the whole backlog.
        "finalize_ms": int((t_end - t_stop) * 1000) if realtime else None,
        # In realtime mode this is "seconds into the recording when words first appeared".
        "first_words_s": round(first - t0, 2) if (realtime and first) else None,
    }


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="samples", help="folder of .wav + .txt pairs")
    ap.add_argument("--engines", default="nemotron,whisper")
    ap.add_argument("--realtime", action="store_true", help="feed audio at real speed to measure live latency")
    ap.add_argument("--no-edge-cases", action="store_true")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    folder = Path(args.data)
    samples = load_samples(folder) if folder.exists() else []
    if not samples:
        print(f"No .wav/.txt pairs found in '{folder}'. Only the silence/noise edge cases will run.")
    cases = [(n, a, r, "sample") for n, a, r in samples]
    if not args.no_edge_cases:
        cases += [(n, a, r, "edge") for n, a, r in build_edge_cases(samples)]

    engines, errors = load_engines(args.engines.split(","))
    for name, err in errors.items():
        print(f"! {name} not loaded: {err}")
    if not engines:
        sys.exit("No engines loaded.")

    # Warm-up so model loading / first-call overhead doesn't skew the first result.
    for engine in engines.values():
        run_one(engine, np.zeros(SAMPLE_RATE, dtype=np.float32), realtime=False)

    rows = []
    for engine_name, engine in engines.items():
        print(f"\n=== {engine_name} ===")
        for case, audio, ref, kind in cases:
            r = run_one(engine, audio, args.realtime)
            score = wer(ref, r["text"])
            hallucinated = len(normalize(r["text"]).split()) if not ref else None
            row = {"engine": engine_name, "case": case, "kind": kind,
                   "wer": round(score, 4) if score is not None else None,
                   "hallucinated_words": hallucinated, **r, "reference": ref}
            rows.append(row)
            shown = f"WER {score:.1%}" if score is not None else f"hallucinated {hallucinated} words"
            fin = f"  finalize {r['finalize_ms']} ms" if r["finalize_ms"] is not None else ""
            print(f"  {case:<22} {shown:<24} RTF {r['rtf']}{fin}"
                  + (f"  first words at {r['first_words_s']} s" if r["first_words_s"] is not None else ""))

    # ---------------- summary ----------------
    print("\n=== Summary (your samples) ===")
    print(f"{'engine':<10} {'avg WER':>8} {'avg RTF':>8} {'finalize':>10}")
    if not args.realtime:
        print("(run with --realtime to measure live latency: first words and finalize time)")
    for engine_name in engines:
        mine = [r for r in rows if r["engine"] == engine_name and r["kind"] == "sample" and r["wer"] is not None]
        if mine:
            avg = lambda k: sum(r[k] for r in mine) / len(mine)
            fin = f"{avg('finalize_ms'):>8.0f}ms" if args.realtime else f"{'-':>10}"
            print(f"{engine_name:<10} {avg('wer'):>8.1%} {avg('rtf'):>8.3f} {fin}")

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    with open(out / f"benchmark-{stamp}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out / f"benchmark-{stamp}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved results to {out}/benchmark-{stamp}.csv and .json")


if __name__ == "__main__":
    main()
