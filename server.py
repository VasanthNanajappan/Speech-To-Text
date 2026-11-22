"""
Low-latency English speech-to-text server.

Browser -> WebSocket -> engine session -> live words back to the browser.

Protocol
--------
1. Client sends JSON config:   {"engine": "nemotron" | "whisper"}
2. Client sends binary frames:  16 kHz mono int16 PCM (about 100 ms each)
   and JSON {"type": "ping"} every few seconds while the user is silent
   (the browser doesn't send silent audio, see client.html).
3. Client ends with JSON:      {"type": "stop"}   -> keep the text
                           or  {"type": "cancel"} -> discard
Server sends:
   {"type": "ready", ...engine info}
   {"type": "partial", "committed": "...", "tentative": "..."}
   {"type": "final", "text": "...", "reason": "...", ...stats}
   {"type": "error", "code": "...", "message": "..."}

Edge cases handled here
-----------------------
- Long dictation: hard cap MAX_SESSION_SECONDS, then finalize and KEEP the text.
- Idle connection: no audio or ping for IDLE_TIMEOUT_SECONDS -> finalize.
- Busy server: all model copies in use -> clear "busy" error, not a hang.
- Bad input: wrong config, odd-length or oversized frames are rejected.
- Disconnects: the session is cancelled and the model is returned to the pool.
- Privacy: audio lives only in memory for the session; nothing is written to disk.

Run:  python server.py   then open http://localhost:8000
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from engines import SAMPLE_RATE, EngineBusy, load_engines

MAX_SESSION_SECONDS = int(os.getenv("MAX_SESSION_SECONDS", "180"))
IDLE_TIMEOUT_SECONDS = int(os.getenv("IDLE_TIMEOUT_SECONDS", "20"))
MAX_FRAME_BYTES = SAMPLE_RATE * 2 * 2          # at most 2 s of int16 audio per frame
FINALIZE_TIMEOUT_SECONDS = 60

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("stt")

ENGINE_NAMES = os.getenv("ENGINES", "nemotron,whisper").split(",")
log.info("Loading engines: %s", ENGINE_NAMES)
ENGINES, ENGINE_ERRORS = load_engines(ENGINE_NAMES)
for name, err in ENGINE_ERRORS.items():
    log.warning("Engine %s not available: %s", name, err)
if not ENGINES:
    raise SystemExit("No speech engine could be loaded. See the warnings above.")
DEFAULT_ENGINE = "nemotron" if "nemotron" in ENGINES else next(iter(ENGINES))
log.info("Ready. Engines: %s (default: %s)", list(ENGINES), DEFAULT_ENGINE)

app = FastAPI(title="Speech to text")


@app.get("/")
def index():
    return FileResponse(Path(__file__).parent / "client.html")


@app.get("/health")
def health():
    return {
        "ok": True,
        "default": DEFAULT_ENGINE,
        "engines": {name: e.info() for name, e in ENGINES.items()},
        "unavailable": ENGINE_ERRORS,
        "limits": {"max_session_s": MAX_SESSION_SECONDS, "idle_timeout_s": IDLE_TIMEOUT_SECONDS},
    }


async def _send_error(ws: WebSocket, code: str, message: str):
    try:
        await ws.send_json({"type": "error", "code": code, "message": message})
        await ws.close()
    except Exception:
        pass


@app.websocket("/ws")
async def stream(ws: WebSocket):
    await ws.accept()
    loop = asyncio.get_running_loop()

    # ---- 1. config ----
    try:
        config = json.loads(await asyncio.wait_for(ws.receive_text(), timeout=10))
    except Exception:
        await _send_error(ws, "bad_config", "Expected a JSON config message first.")
        return
    engine_name = config.get("engine") or DEFAULT_ENGINE
    engine = ENGINES.get(engine_name)
    if engine is None:
        await _send_error(ws, "bad_engine", f"Engine '{engine_name}' is not available.")
        return

    # ---- 2. session (may be busy) ----
    outbox: asyncio.Queue = asyncio.Queue()

    def on_text(update: dict):
        # Called from the engine's worker thread; hand over to the event loop safely.
        loop.call_soon_threadsafe(outbox.put_nowait, update)

    try:
        session = await asyncio.to_thread(engine.new_session, on_text)
    except EngineBusy:
        await _send_error(ws, "busy", "Voice input is busy right now. Please try again in a moment.")
        return

    await ws.send_json({"type": "ready", **engine.info()})
    started = time.perf_counter()
    finished = False

    async def sender():
        # Only the newest partial matters: skip stale ones so a slow
        # connection never builds up a backlog.
        while True:
            update = await outbox.get()
            while not outbox.empty():
                update = outbox.get_nowait()
            await ws.send_json({"type": "partial", **update})

    send_task = asyncio.create_task(sender())

    async def finalize(reason: str):
        nonlocal finished
        finished = True
        t = time.perf_counter()
        try:
            text = await asyncio.wait_for(asyncio.to_thread(session.finish), FINALIZE_TIMEOUT_SECONDS)
            error = None
        except Exception as exc:
            log.exception("Finalize failed")
            text, error = "", str(exc)
        send_task.cancel()
        stats = session.stats.as_dict()
        payload = {"type": "final", "text": text, "reason": reason,
                   "finalize_ms": int((time.perf_counter() - t) * 1000), **stats}
        if error:
            payload["error"] = error
        # Log numbers only, never the user's words.
        log.info("session engine=%s reason=%s audio=%ss compute=%sms rtf=%s",
                 engine_name, reason, stats["audio_s"], stats["compute_ms"], stats["rtf"])
        try:
            await ws.send_json(payload)
            await ws.close()
        except Exception:
            pass

    # ---- 3. receive loop ----
    try:
        while True:
            try:
                msg = await asyncio.wait_for(ws.receive(), timeout=IDLE_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                await finalize("idle_timeout")
                break

            if msg["type"] == "websocket.disconnect":
                break

            data = msg.get("bytes")
            if data is not None:
                if len(data) % 2 or len(data) > MAX_FRAME_BYTES:
                    continue  # malformed or oversized frame: ignore it, keep the session
                audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                session.feed(audio)
                if session.stats.audio_samples >= MAX_SESSION_SECONDS * SAMPLE_RATE:
                    await finalize("max_length")
                    break
                continue

            text = msg.get("text")
            if text:
                try:
                    kind = json.loads(text).get("type")
                except Exception:
                    continue
                if kind == "stop":
                    await finalize("stopped")
                    break
                if kind == "cancel":
                    break
                # "ping" just resets the idle timer.
    except WebSocketDisconnect:
        pass
    finally:
        send_task.cancel()
        if not finished:
            await asyncio.to_thread(session.cancel)   # frees the model for the next user
            try:
                await ws.close()                      # tell the client we're done (cancel path)
            except Exception:
                pass
        log.info("closed after %.1fs", time.perf_counter() - started)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8000")))
