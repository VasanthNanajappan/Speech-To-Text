# Low-latency English speech-to-text

A speech button for the video prompt composer. Words appear while you talk.

- **nemotron** (default): NVIDIA Nemotron Speech Streaming EN 0.6B. Reads each audio chunk once, emits words as it goes. ~0.56 s delay.
- **whisper** (baseline): faster-whisper re-reading recent audio every 0.8 s. Kept so you can benchmark the difference.

## Files

| File | What it does |
|---|---|
| `engines.py` | Both models behind one streaming interface (feed audio → get words). |
| `server.py` | WebSocket server. Session limits, idle timeout, busy handling. |
| `client.html` | The speech button. Silence/mute detection, auto-stop, limits, metrics. |
| `benchmark.py` | Accuracy, speed, latency and edge-case tests on your recordings. |

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or: pip install torch  (NVIDIA GPU)
pip install -r requirements.txt
python server.py
```

Open **http://localhost:8000** in Chrome or Edge. The first run downloads both models (Nemotron ~2.5 GB, Whisper small.en ~0.5 GB).

Only want one engine? `ENGINES=nemotron python server.py`

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `ENGINES` | `nemotron,whisper` | Which engines to load |
| `NEMOTRON_LOOKAHEAD` | `6` | Latency knob: `13`=1.12 s, `6`=0.56 s, `1`=0.16 s, `0`=0.08 s |
| `NEMOTRON_WORKERS` | `1` | Model copies = people who can dictate at the same time |
| `NEMOTRON_DEVICE` | auto | `cuda` or `cpu` |
| `WHISPER_MODEL` | `small.en` | `tiny.en`, `base.en`, `small.en`, `medium.en` |
| `WHISPER_STEP_SECONDS` | `0.8` | How often Whisper re-reads audio |
| `MAX_SESSION_SECONDS` | `180` | Hard cap per dictation (text is kept) |
| `IDLE_TIMEOUT_SECONDS` | `20` | No audio or ping for this long → finalize |

## Benchmark

1. Record test clips: open **http://localhost:8000/?record=1**, dictate, fix the text box to exactly what you said, download the `.wav` and `.txt` into `samples/`. Aim for 20–30 clips: short and long prompts, fast and slow speech, video-editing words ("B-roll", "jump cut", "9:16").
2. Run:

```bash
python benchmark.py --data samples --engines nemotron,whisper              # accuracy + speed
python benchmark.py --data samples --engines nemotron,whisper --realtime   # live latency (takes as long as the audio)
```

Results print as a table and save to `results/benchmark-*.csv` and `.json`.

To compare Nemotron latency settings, run it once per setting:

```bash
for L in 13 6 1 0; do NEMOTRON_LOOKAHEAD=$L python benchmark.py --engines nemotron --realtime --out results/lookahead-$L; done
```

**Reading the numbers**

| Metric | Good | Meaning |
|---|---|---|
| WER | lower | Words wrong vs your reference. 5% ≈ 1 word in 20. |
| RTF | below 1.0 | Compute ÷ audio length. Above 1.0 means it can't keep up live. |
| first_words_s | lower | (realtime) How far into the clip the first word appeared. |
| finalize_ms | lower | (realtime) Stop → final text. |
| hallucinated_words | 0 | Words invented for silence or noise. |

Note: write numbers in your `.txt` the way the model outputs them. If the model writes "30" and your reference says "thirty", it counts as an error.

## Edge cases handled

| Case | Where | What happens |
|---|---|---|
| Long dictation | client + server | Timer; warning at 90 s; auto-stop at ~170 s keeping the text; server hard cap 180 s. Nemotron's memory stays flat (it only keeps ~5.6 s of context). |
| Background noise | client | Browser noise suppression + echo cancellation + auto gain. Adaptive voice detection learns the room's noise level. |
| Video/audio playing on the page | client | Paused while the mic is on, resumed after, so the mic doesn't transcribe it. |
| Mic pressed, user silent | client | After 8 s with no speech: stops with "Didn't hear anything." |
| Mic muted / wrong device | client | Near-zero signal for 2 s: "Is your microphone muted?" |
| Finished speaking | client | Auto-stops after 2.5 s of silence (toggle "Stop when I pause"). |
| Accidental tap | client | Under 0.5 s with no speech: ignored silently. |
| Silence uploads | client | Silence isn't sent (300 ms pre-roll and 600 ms hangover keep word edges). Keep-alive pings instead. |
| Mic permission denied / missing / in use | client | Specific message for each. |
| Mic unplugged mid-dictation | client | Stops and keeps text so far. |
| Connection lost | client | Keeps text received so far. |
| Server busy | server | All model copies in use → "busy, try again" instead of hanging. |
| Idle connection | server | No audio/ping for 20 s → finalize. |
| Malformed or oversized frames | server | Ignored; session continues. |
| Disconnect / cancel | server | Session cancelled; model returned to the pool. |
| Whisper hallucinating on silence | engine | Final pass skipped when the audio is silent. |
| Empty result | client | Nothing inserted; asks the user to try again. |

## Privacy

Audio is never written to disk or browser storage. It lives in server memory only during the session. Server logs contain timings, never words. The `?record=1` mode keeps audio in browser memory only until you download it.

## Manual test checklist

Run each and note the result for your write-up:

- [ ] Dictate a normal prompt; note "speech → first words" and "stop → final text".
- [ ] Talk for 2+ minutes.
- [ ] Fan or music in the background.
- [ ] A video playing on the page while you press the mic.
- [ ] Press the mic and say nothing.
- [ ] Mute your mic (OS level) and press the mic.
- [ ] Tap the mic and immediately stop.
- [ ] Unplug a USB/Bluetooth mic mid-sentence.
- [ ] Stop `server.py` mid-sentence.
- [ ] Open two tabs and dictate in both (expect "busy" with 1 worker).
- [ ] Block mic permission in the address bar.
