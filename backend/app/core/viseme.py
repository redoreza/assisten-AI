"""Word boundary → viseme event conversion for VRM lip-sync.

Strategy: split each word into letter groups, map each group to an Oculus/VRM
viseme key, then time-distribute the visemes across the word's duration weighted
by letter count.

This is intentionally simple — for production-grade phoneme accuracy use
phonemizer + espeak-ng (Phase 6 polish). For Phase 2-4 the goal is "mouth opens
in roughly the right pattern and timing", which letter-based mapping achieves.
"""

from __future__ import annotations

import re
from typing import TypedDict

from app.services.tts_edge import WordBoundary


class VisemeEvent(TypedDict):
    phoneme: str
    time: float
    duration: float


_DIGRAPH_MAP: dict[str, str] = {
    "ch": "CH", "sh": "CH",
    "th": "TH",
    "ph": "FF",
    "ng": "nn", "ny": "nn",
    "qu": "kk",
}

_LETTER_MAP: dict[str, str] = {
    "a": "aa", "e": "E", "i": "I", "o": "O", "u": "U",
    "b": "PP", "p": "PP", "m": "PP",
    "f": "FF", "v": "FF",
    "d": "DD", "t": "DD",
    "k": "kk", "g": "kk", "c": "kk", "q": "kk",
    "s": "SS", "z": "SS", "x": "SS",
    "n": "nn",
    "r": "RR", "l": "RR",
    "j": "CH",
    "h": "kk", "w": "U", "y": "I",
}

_NON_LETTER = re.compile(r"[^a-zA-Z]+")


def _tokenize_phonemes(word: str) -> list[str]:
    cleaned = _NON_LETTER.sub("", word).lower()
    if not cleaned:
        return []
    out: list[str] = []
    i = 0
    while i < len(cleaned):
        if i + 1 < len(cleaned):
            di = cleaned[i : i + 2]
            if di in _DIGRAPH_MAP:
                out.append(_DIGRAPH_MAP[di])
                i += 2
                continue
        ch = cleaned[i]
        if ch in _LETTER_MAP:
            out.append(_LETTER_MAP[ch])
        i += 1
    # collapse runs of the same viseme — they'd otherwise jitter the mouth shape
    collapsed: list[str] = []
    for v in out:
        if not collapsed or collapsed[-1] != v:
            collapsed.append(v)
    return collapsed


def word_to_visemes(boundary: WordBoundary) -> list[VisemeEvent]:
    phonemes = _tokenize_phonemes(boundary["text"])
    if not phonemes:
        return [
            {"phoneme": "sil", "time": boundary["offset"], "duration": boundary["duration"]}
        ]
    per = boundary["duration"] / len(phonemes)
    return [
        {
            "phoneme": p,
            "time": boundary["offset"] + idx * per,
            "duration": per,
        }
        for idx, p in enumerate(phonemes)
    ]


def boundaries_to_visemes(boundaries: list[WordBoundary]) -> list[VisemeEvent]:
    """Build a single viseme timeline from a list of word boundaries.

    Gaps ≥ 50ms between words are filled with `sil` so the avatar's mouth closes
    between words rather than freezing on the last viseme.
    """
    events: list[VisemeEvent] = []
    prev_end = 0.0
    for b in boundaries:
        if b["offset"] - prev_end > 0.05:
            events.append(
                {
                    "phoneme": "sil",
                    "time": prev_end,
                    "duration": b["offset"] - prev_end,
                }
            )
        events.extend(word_to_visemes(b))
        prev_end = b["offset"] + b["duration"]
    return events
