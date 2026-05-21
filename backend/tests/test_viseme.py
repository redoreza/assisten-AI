"""Viseme conversion unit tests — pure offline."""

from __future__ import annotations

from app.core.viseme import _tokenize_phonemes, boundaries_to_visemes, word_to_visemes


def test_tokenize_basic_vowels() -> None:
    assert _tokenize_phonemes("aiueo") == ["aa", "I", "U", "E", "O"]


def test_tokenize_digraph_ng() -> None:
    assert _tokenize_phonemes("ng") == ["nn"]
    assert _tokenize_phonemes("nganga") == ["nn", "aa", "nn", "aa"]


def test_tokenize_strips_punctuation() -> None:
    assert _tokenize_phonemes("Halo!") == ["kk", "aa", "RR", "O"]


def test_word_to_visemes_distributes_time() -> None:
    visemes = word_to_visemes({"text": "Halo", "offset": 1.0, "duration": 0.8})
    assert len(visemes) == 4  # H A L O → kk aa RR O
    assert visemes[0]["time"] == 1.0
    assert all(abs(v["duration"] - 0.2) < 1e-9 for v in visemes)
    assert abs((visemes[-1]["time"] + visemes[-1]["duration"]) - 1.8) < 1e-9


def test_empty_word_emits_silence() -> None:
    visemes = word_to_visemes({"text": "...", "offset": 0.5, "duration": 0.3})
    assert visemes == [{"phoneme": "sil", "time": 0.5, "duration": 0.3}]


def test_boundaries_to_visemes_fills_gaps() -> None:
    boundaries = [
        {"text": "Halo", "offset": 0.0, "duration": 0.4},
        {"text": "Aria", "offset": 0.8, "duration": 0.4},
    ]
    events = boundaries_to_visemes(boundaries)  # type: ignore[arg-type]
    sil_count = sum(1 for e in events if e["phoneme"] == "sil")
    assert sil_count >= 1
    for prev, nxt in zip(events[:-1], events[1:], strict=True):
        assert nxt["time"] >= prev["time"]


def test_no_gap_no_silence() -> None:
    boundaries = [
        {"text": "Halo", "offset": 0.0, "duration": 0.4},
        {"text": "Aria", "offset": 0.4, "duration": 0.4},
    ]
    events = boundaries_to_visemes(boundaries)  # type: ignore[arg-type]
    assert not any(e["phoneme"] == "sil" for e in events)
