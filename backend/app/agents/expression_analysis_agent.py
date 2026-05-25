"""ExpressionAnalysisAgent — analyse facial expression from a webcam frame.

Uses NVIDIA NIM vision LLM (llama-4-maverick, ~1.5 s) to infer the person's
mood from a base64-encoded image so the orchestrator can personalise greetings.

Returned moods: happy | focused | confused | tired | neutral
Timeout: 3 s hard cap — falls back to "neutral" so face engagement never stalls.
"""

from __future__ import annotations

import asyncio

from loguru import logger

MOOD_VALUES = ("happy", "focused", "confused", "tired", "neutral")
_VISION_MODEL = "meta/llama-4-maverick-17b-128e-instruct"

_SYSTEM = (
    "You are an expression classifier. "
    "Given a face image, reply with exactly ONE word from this list: "
    "happy, focused, confused, tired, neutral. "
    "No explanation, no punctuation — just the single word."
)


class ExpressionAnalysisAgent:
    def __init__(self) -> None:
        self._llm = self._build_llm()

    def _build_llm(self):
        try:
            from app.services.llm_nvidia import NvidiaLLM
            llm = NvidiaLLM(model=_VISION_MODEL)
            logger.info(f"[ExpressionAgent] vision LLM: {_VISION_MODEL}")
            return llm
        except Exception as exc:
            logger.warning(f"[ExpressionAgent] init failed ({exc}), expressions disabled")
            return None

    async def analyze(self, image_b64: str) -> str:
        """Return mood string. Always returns a valid mood — never raises."""
        if self._llm is None or not image_b64:
            return "neutral"

        # Preserve original MIME type from data URL; default to jpeg for raw base64
        if "," in image_b64 and image_b64.startswith("data:"):
            data_url = image_b64
        else:
            data_url = f"data:image/jpeg;base64,{image_b64}"

        messages = [
            {"role": "system", "content": _SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": "What is the mood?"},
                ],
            },
        ]

        try:
            raw = await asyncio.wait_for(
                self._llm.generate(messages, temperature=0.0, max_tokens=5),
                timeout=3.0,
            )
        except asyncio.TimeoutError:
            logger.warning("[ExpressionAgent] timeout — using neutral")
            return "neutral"
        except Exception as exc:
            logger.warning(f"[ExpressionAgent] LLM error: {exc} — using neutral")
            return "neutral"

        mood = raw.strip().lower().rstrip(".")
        if mood not in MOOD_VALUES:
            # Try first word if model returned extra text
            mood = mood.split()[0] if mood.split() else "neutral"
        if mood not in MOOD_VALUES:
            mood = "neutral"

        logger.info(f"[ExpressionAgent] mood detected: {mood}")
        return mood


_agent: ExpressionAnalysisAgent | None = None


def get_expression_analysis_agent() -> ExpressionAnalysisAgent:
    global _agent
    if _agent is None:
        _agent = ExpressionAnalysisAgent()
    return _agent
