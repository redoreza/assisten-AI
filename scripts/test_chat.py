"""Phase 1 smoke test — hits /api/chat and /api/chat/stream against a running backend.

Usage:
    # Terminal 1
    cd backend && uv run uvicorn app.main:app --reload --port 8000

    # Terminal 2
    uv run --project backend python ../scripts/test_chat.py
    uv run --project backend python ../scripts/test_chat.py "Apa kabar hari ini?"
"""

from __future__ import annotations

import json
import sys
import time

import httpx

BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MESSAGE = "Halo, perkenalkan dirimu dalam dua kalimat."


def test_health(client: httpx.Client) -> None:
    r = client.get(f"{BASE_URL}/health")
    r.raise_for_status()
    data = r.json()
    print(f"[health] {data}")
    if not data.get("groq_key_configured"):
        print("  WARNING: GROQ_API_KEY missing from .env — set it and restart backend.")
        sys.exit(2)


def test_personas(client: httpx.Client) -> None:
    r = client.get(f"{BASE_URL}/api/personas")
    r.raise_for_status()
    print(f"[personas] {r.json()}")


def test_chat_non_stream(client: httpx.Client, message: str) -> None:
    payload = {"message": message}
    t0 = time.perf_counter()
    r = client.post(f"{BASE_URL}/api/chat", json=payload, timeout=60)
    elapsed = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    data = r.json()
    print(f"\n[/api/chat] {elapsed:.0f}ms wall, {data['latency_ms']}ms server")
    print(f"  persona: {data['persona_id']}  model: {data['model']}")
    print(f"  reply  : {data['reply']}")


def test_chat_stream(client: httpx.Client, message: str) -> None:
    payload = {"message": message}
    t0 = time.perf_counter()
    first_token_t: float | None = None
    chars = 0
    print("\n[/api/chat/stream] streaming...")
    print("  ", end="", flush=True)
    with client.stream("POST", f"{BASE_URL}/api/chat/stream", json=payload, timeout=60) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw or not raw.startswith("data:"):
                continue
            body = raw[5:].strip()
            if body == "[DONE]":
                break
            try:
                ev = json.loads(body)
            except json.JSONDecodeError:
                continue
            if ev.get("event") == "token":
                if first_token_t is None:
                    first_token_t = time.perf_counter()
                txt = ev.get("text", "")
                chars += len(txt)
                print(txt, end="", flush=True)
            elif ev.get("event") == "done":
                print()
                print(
                    f"  server: first_token={ev.get('first_token_ms')}ms "
                    f"total={ev.get('total_ms')}ms"
                )
            elif ev.get("event") == "error":
                print(f"\n  ERROR: {ev.get('message')}")
                return
    total = (time.perf_counter() - t0) * 1000
    if first_token_t is not None:
        first_ms = (first_token_t - t0) * 1000
        print(f"  client: first_token={first_ms:.0f}ms total={total:.0f}ms chars={chars}")


def main() -> None:
    message = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MESSAGE
    with httpx.Client() as client:
        try:
            test_health(client)
        except httpx.ConnectError:
            print(
                "Cannot reach backend. Start it with:\n"
                "  cd backend && uv run uvicorn app.main:app --reload --port 8000"
            )
            sys.exit(1)
        test_personas(client)
        test_chat_non_stream(client, message)
        test_chat_stream(client, message)


if __name__ == "__main__":
    main()
