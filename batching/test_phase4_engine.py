"""
Phase 4 test: the "nemotron-batched" engine, used the way server.py uses it.

  1  Concurrent users   several users talk at once from separate threads, starting at
                        different times; each final text must equal their solo result
  2  Live updates       users receive growing text while still talking
  3  Busy               more users than slots -> clean "busy" error, slots free up after
  4  Cancel             one user cancels mid-sentence; others unaffected, slot freed
  5  Empty              press and release with no audio -> empty text, no hang
  6  Server             the real server.py over WebSockets, two browsers at once

Audio is fed faster than real time and ticks run faster to keep the test short; the
math is the same as live.

Usage:
  python test_phase4_engine.py --small                 # no download (scaled-down model)
  python test_phase4_engine.py --audio ../samples      # real model + your speech
"""
import argparse
import json
import os
import random
import sys
import threading
import time

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False

from batched_engine import BatchedNemotronEngine, EngineBusy
from batched_stream import BatchedStreamer, StreamState, mel_chunks
from common import expand_audio_paths, load_audio_or_noise

PIECE = 1600            # 100 ms pieces, like the browser sends
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def solo_text(engine, audio):
    st, s = BatchedStreamer(engine.model, engine.lookahead), StreamState()
    for c in mel_chunks(engine.feature_extractor, audio, engine.lookahead):
        st.step([s], [c])
    return engine._decode(s.tokens).strip()


def talk(engine, audio, start_delay, speed, out, key, cancel_after=None):
    """One user: wait, open a session, stream audio in 100 ms pieces, then finish (or cancel)."""
    time.sleep(start_delay)
    updates = []
    s = engine.new_session(lambda u: updates.append(u["committed"]))
    for i in range(0, len(audio), PIECE):
        if cancel_after is not None and i >= cancel_after:
            s.cancel()
            out[key] = {"cancelled": True, "updates": updates}
            return
        s.feed(audio[i:i + PIECE])
        time.sleep(PIECE / 16000 / speed)
    out[key] = {"text": s.finish(), "updates": updates, "stats": s.stats.as_dict()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true")
    ap.add_argument("--audio", nargs="*", default=[])
    ap.add_argument("--users", type=int, default=0, help="default: 5 with --small, 3 with the real model")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--speed", type=float, default=0.0,
                    help="how many times faster than real time to feed audio (default: 10 small, 1 real)")
    args = ap.parse_args()
    args.audio = expand_audio_paths(args.audio)
    speed = args.speed or (10.0 if args.small else 1.0)
    args.users = args.users or (5 if args.small else 3)

    files = [a for a in (load_audio_or_noise(p) for p in args.audio) if len(a) >= 16000]
    if not files and not args.small:
        sys.exit("With the real model, pass --audio with speech recordings.")

    rng = random.Random(args.seed)
    def clip(i):
        sec = rng.choice([3, 5, 7, 9])
        if files:
            a = files[i % len(files)]
            n = int(sec * 16000)
            st = rng.randint(0, max(0, len(a) - n))
            return a[st:st + n]
        return load_audio_or_noise(None, seed=200 + i, seconds=sec)

    print("Loading engine...")
    probe = BatchedNemotronEngine(small=args.small, max_users=args.users, tick_seconds=0.56 / speed)
    tick = probe.tick_seconds
    print(f"tick {tick*1000:.0f} ms, audio fed {speed:g}x real time\n")

    # 1 + 2: concurrent users with staggered starts
    audios = [clip(i) for i in range(args.users)]
    refs = [solo_text(probe, a) for a in audios]
    out = {}
    threads = [threading.Thread(target=talk, args=(probe, a, rng.uniform(0, 1.5) / speed * 2, speed, out, i))
               for i, a in enumerate(audios)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    exact = sum(out[i]["text"] == refs[i] for i in range(len(audios)))
    check(f"1 concurrent users get their solo text", exact == len(audios),
          f"{exact}/{len(audios)} exact, avg {probe.info()['avg_users_per_call']} users per model call")
    for i in range(len(audios)):
        if out[i]["text"] != refs[i]:
            print(f"      user {i}: got {out[i]['text'][:60]!r}\n              want {refs[i][:60]!r}")
    if files and probe.tokenizer is not None:
        print(f"      e.g. user 0 heard: {out[0]['text'][:90]!r}")
    grew = all(len(out[i]["updates"]) >= 1 for i in range(len(audios)) if refs[i])
    check("2 live updates arrive while talking", grew,
          f"{sum(len(out[i]['updates']) for i in out)} updates in total")

    # 3: busy
    small_engine_slots = probe.max_users
    held = [probe.new_session(lambda u: None) for _ in range(small_engine_slots)]
    try:
        probe.new_session(lambda u: None, acquire_timeout=0.3)
        busy_ok = False
    except EngineBusy:
        busy_ok = True
    for h in held:
        h.finish()
    try:
        probe.new_session(lambda u: None, acquire_timeout=1.0).finish()
        freed = True
    except EngineBusy:
        freed = False
    check("3 extra user gets 'busy', slots free up afterwards", busy_ok and freed)

    # 4: cancel mid-sentence while others talk
    out = {}
    a_keep, a_cancel = clip(10), clip(11)
    ref_keep = solo_text(probe, a_keep)
    t1 = threading.Thread(target=talk, args=(probe, a_keep, 0, speed, out, "keep"))
    t2 = threading.Thread(target=talk, args=(probe, a_cancel, 0, speed, out, "cancel", len(a_cancel) // 2))
    t1.start(); t2.start(); t1.join(); t2.join()
    time.sleep(tick * 3)
    check("4 cancel mid-sentence: others unaffected, slot freed",
          out["keep"]["text"] == ref_keep and out["cancel"]["cancelled"] and probe.info()["active_users"] == 0)

    # 5: empty session
    t0 = time.time()
    txt = probe.new_session(lambda u: None).finish(timeout=10)
    check("5 empty session returns empty text quickly", txt == "" and time.time() - t0 < 5)
    probe.close()

    # 6: the real server over WebSockets
    os.environ["ENGINES"] = "nemotron-batched"
    os.environ["NEMOTRON_BATCH_MAX_USERS"] = "4"
    if args.small:
        os.environ["NEMOTRON_TEST_SMALL"] = "1"
    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent)
    try:
        import server
    except SystemExit:
        check("6 server.py: two browsers at once over WebSockets", False,
              "server could not load the engine; the WARNING line above says why. "
              "'Unknown engine' = engines.py next to server.py is the old version; "
              "'paging file is too small' / out of memory = not enough RAM for the model")
        print("\nPHASE 4: FAIL (1 failed)")
        sys.exit(1)
    from fastapi.testclient import TestClient
    client = TestClient(server.app)
    health = client.get("/health").json()
    srv_engine = server.ENGINES["nemotron-batched"]
    srv_engine.tick_seconds = 0.56 / speed
    audios = [clip(20), clip(21)]
    refs = [solo_text(srv_engine, a) for a in audios]
    finals = {}

    def browser(i):
        pcm = (np.clip(audios[i], -1, 1) * 32767).astype(np.int16)
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"engine": "nemotron-batched"}))
            assert ws.receive_json()["type"] == "ready"
            for k in range(0, len(pcm), PIECE):
                ws.send_bytes(pcm[k:k + PIECE].tobytes())
                time.sleep(PIECE / 16000 / speed)
            ws.send_text(json.dumps({"type": "stop"}))
            while True:
                m = ws.receive_json()
                if m["type"] == "final":
                    finals[i] = m
                    return

    # int16 conversion changes the audio slightly, so compare with the solo result of the same int16 audio
    refs = [solo_text(srv_engine, (np.clip(a, -1, 1) * 32767).astype(np.int16).astype(np.float32) / 32768.0)
            for a in audios]
    ts = [threading.Thread(target=browser, args=(i,)) for i in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    ok = all(finals[i]["text"] == refs[i] for i in range(2))
    check("6 server.py: two browsers at once over WebSockets", ok and "nemotron-batched" in health["engines"],
          f"final reason {finals[0]['reason']}, rtf {finals[0]['rtf']}")
    srv_engine.close()

    print("\nPHASE 4:", "PASS" if all(RESULTS) else f"FAIL ({RESULTS.count(False)} failed)")
    sys.exit(0 if all(RESULTS) else 1)


if __name__ == "__main__":
    main()
