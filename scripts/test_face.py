"""Phase R1 smoke test — face recognize + enroll against running backend.

Uses the bundled InsightFace sample images (Tom_Hanks_54745.png, t1.jpg) so the
test is self-contained.

Usage:
    # Terminal 1
    cd backend && uv run uvicorn app.main:app --port 8000

    # Terminal 2
    uv run --project backend python ../scripts/test_face.py
"""

from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

import httpx

BASE_URL = "http://127.0.0.1:8000"
BACKEND_ROOT = Path(__file__).resolve().parent.parent / "backend"
INSIGHTFACE_IMAGES = (
    BACKEND_ROOT
    / ".venv"
    / "Lib"
    / "site-packages"
    / "insightface"
    / "data"
    / "images"
)


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def test_health(client: httpx.Client) -> None:
    r = client.get(f"{BASE_URL}/health")
    r.raise_for_status()
    print(f"[health] {r.json()}")


def test_warmup(client: httpx.Client) -> None:
    print("\n[warmup] triggering InsightFace model load — first run downloads ~300 MB, can take 30-90s...")
    t0 = time.perf_counter()
    r = client.post(f"{BASE_URL}/api/face/warmup", timeout=180)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    print(f"[warmup] {elapsed:.1f}s — {r.json()}")


def test_persons_empty(client: httpx.Client) -> None:
    r = client.get(f"{BASE_URL}/api/face/persons")
    r.raise_for_status()
    data = r.json()
    print(f"\n[persons before enroll] count={data['count']} persons={data['persons']}")


def test_recognize_no_db(client: httpx.Client, img: Path) -> None:
    print(f"\n[recognize before enroll] {img.name}")
    payload = {"image_base64": _b64(img)}
    t0 = time.perf_counter()
    r = client.post(f"{BASE_URL}/api/face/recognize", json=payload, timeout=60)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    print(f"  {elapsed * 1000:.0f}ms - count={data['count']}")
    for f in data["faces"]:
        print(
            f"  bbox={f['bbox']} det_score={f['det_score']:.3f} "
            f"match={f['match_name']!r} sim={f['similarity']:.3f}"
        )


def test_enroll(client: httpx.Client, name: str, imgs: list[Path]) -> int:
    print(f"\n[enroll] name={name!r} with {len(imgs)} images")
    payload = {"name": name, "images_base64": [_b64(p) for p in imgs]}
    t0 = time.perf_counter()
    r = client.post(f"{BASE_URL}/api/face/enroll", json=payload, timeout=60)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    print(f"  {elapsed * 1000:.0f}ms - {data}")
    return int(data["person_id"])


def test_recognize_after_enroll(client: httpx.Client, img: Path) -> None:
    print(f"\n[recognize after enroll] {img.name}")
    payload = {"image_base64": _b64(img)}
    t0 = time.perf_counter()
    r = client.post(f"{BASE_URL}/api/face/recognize", json=payload, timeout=60)
    elapsed = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    print(f"  {elapsed * 1000:.0f}ms - count={data['count']}")
    for f in data["faces"]:
        print(
            f"  bbox={f['bbox']} det_score={f['det_score']:.3f} "
            f"match={f['match_name']!r} sim={f['similarity']:.3f}"
        )


def test_persons_after(client: httpx.Client) -> None:
    r = client.get(f"{BASE_URL}/api/face/persons")
    r.raise_for_status()
    data = r.json()
    print(f"\n[persons after enroll] count={data['count']} persons={data['persons']}")


def test_cleanup(client: httpx.Client, person_id: int) -> None:
    print(f"\n[cleanup] deleting person_id={person_id}")
    r = client.delete(f"{BASE_URL}/api/face/persons/{person_id}")
    r.raise_for_status()
    print(f"  {r.json()}")


def main() -> int:
    multi = INSIGHTFACE_IMAGES / "t1.jpg"
    # Tom_Hanks_54745.png exists but is a 112x112 ArcFace crop — too small for
    # the SCRFD detector at det_size=640. Skip it as enrollment input.
    if not multi.exists():
        print(
            f"FAIL: missing bundled InsightFace samples in {INSIGHTFACE_IMAGES}. "
            "Re-run `uv sync` in backend/"
        )
        return 1

    with httpx.Client() as client:
        try:
            test_health(client)
        except httpx.ConnectError:
            print(
                f"FAIL: cannot reach backend at {BASE_URL}. Start it with:\n"
                "  cd backend && uv run uvicorn app.main:app --port 8000"
            )
            return 1

        test_warmup(client)
        test_persons_empty(client)
        # Recognize before enroll — should detect 6 faces, all with match=None
        test_recognize_no_db(client, multi)
        # Enroll the largest face under "PersonA" using the same image
        person_id = test_enroll(client, "PersonA", [multi])
        # Re-recognize — one of the 6 faces should now match PersonA with high similarity
        test_recognize_after_enroll(client, multi)
        test_persons_after(client)
        test_cleanup(client, person_id)
        test_persons_after(client)

    print("\nAll R1 face tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
