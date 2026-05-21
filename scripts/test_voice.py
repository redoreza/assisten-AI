"""Phase 2 acceptance — voice-to-voice E2E test against /ws.

Pipeline under test:
    fake user audio (Edge TTS) -> WS audio_chunk/audio_end
    -> server STT (Groq Whisper) -> LLM (Groq Llama) -> TTS (Edge) -> viseme
    -> client receives transcript + ai_text + audio chunks + viseme events

Usage:
    # Terminal 1
    cd backend && uv run uvicorn app.main:app --port 8000

    # Terminal 2
    uv run --project backend python ../scripts/test_voice.py
    uv run --project backend python ../scripts/test_voice.py "Apa kabar, Aria?"
"""

from __future__ import annotations

import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import edge_tts
import websockets

WS_URL = "ws://127.0.0.1:8000/ws"
DEFAULT_USER_PROMPT = "Halo Aria, sebutkan tiga hal favoritmu dalam satu kalimat."
OUT_DIR = Path(__file__).resolve().parent / "_out"


async def synth_user_audio(text: str) -> bytes:
    """Render the 'user' prompt to MP3 — we send this as if it came from a microphone."""
    comm = edge_tts.Communicate(text, "id-ID-ArdiNeural")
    audio = bytearray()
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            audio.extend(chunk["data"])
    return bytes(audio)


def _save_audio(idx: int, b64: str, fmt: str) -> Path:
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / f"voice_test_{idx:02d}.{fmt}"
    path.write_bytes(base64.b64decode(b64))
    return path


async def run(user_text: str) -> int:
    print(f"[user prompt] {user_text!r}")
    user_audio = await synth_user_audio(user_text)
    print(f"  synthesized {len(user_audio)} bytes of mp3 to act as mic input")

    audio_b64 = base64.b64encode(user_audio).decode("ascii")

    t_connect = time.perf_counter()
    try:
        ws = await websockets.connect(WS_URL, max_size=10 * 1024 * 1024)
    except (OSError, websockets.exceptions.InvalidHandshake) as exc:
        print(f"\nCannot connect to {WS_URL}: {exc}")
        print("Start backend first: cd backend && uv run uvicorn app.main:app --port 8000")
        return 1

    async with ws:
        ready = json.loads(await ws.recv())
        print(f"[ready] {ready}")

        await ws.send(
            json.dumps({"type": "audio_chunk", "data": audio_b64, "format": "mp3"})
        )
        await ws.send(json.dumps({"type": "audio_end"}))
        t_sent = time.perf_counter()

        first_audio_t: float | None = None
        audio_count = 0
        transcript = ""
        full_text = ""

        while True:
            raw = await ws.recv()
            ev = json.loads(raw) if isinstance(raw, str) else json.loads(raw.decode())
            etype = ev.get("type")

            if etype == "transcript":
                transcript = ev.get("text", "")
                print(f"\n[transcript] ({ev.get('latency_ms')}ms) {transcript!r}")
            elif etype == "ai_text":
                if ev.get("is_final"):
                    full_text = ev.get("text", "")
                    print(f"\n[ai_text final] {full_text!r}")
                else:
                    print(f"  [ai_text #{ev.get('sentence_idx')}] {ev.get('text', '')}")
            elif etype == "audio":
                if first_audio_t is None:
                    first_audio_t = time.perf_counter()
                    print(f"  [first audio @ {(first_audio_t - t_sent) * 1000:.0f}ms wall]")
                audio_count += 1
                path = _save_audio(ev.get("sequence", audio_count), ev["data"], ev.get("format", "mp3"))
                print(
                    f"  [audio seq={ev.get('sequence')} sentence={ev.get('sentence_idx')} "
                    f"tts_ms={ev.get('tts_ms')} -> {path.name}]"
                )
            elif etype == "viseme":
                print(
                    f"  [viseme audio_seq={ev.get('audio_seq')} events={len(ev.get('events', []))}]"
                )
            elif etype == "timing":
                print(
                    f"\n[server timing] first_token={ev.get('first_token_ms')}ms "
                    f"first_audio={ev.get('first_audio_ms')}ms total={ev.get('total_ms')}ms "
                    f"sentences={ev.get('sentences')}"
                )
            elif etype == "error":
                print(f"\n[ERROR from server] {ev.get('message')}")
                return 2
            elif etype == "done":
                print("[done]")
                break
            elif etype == "ready":
                pass
            else:
                print(f"  [unknown event] {ev}")

        total_wall = (time.perf_counter() - t_connect) * 1000
        print(f"\nTotal wall clock: {total_wall:.0f}ms")
        print(f"Audio chunks saved to: {OUT_DIR}")
        if audio_count == 0:
            print("FAIL: no audio chunks received")
            return 3
        if not transcript:
            print("FAIL: no transcript")
            return 4
        return 0


def main() -> None:
    user_text = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_USER_PROMPT
    rc = asyncio.run(run(user_text))
    sys.exit(rc)


if __name__ == "__main__":
    main()
