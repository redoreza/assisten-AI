"""SpeechPrepAgent — naturalize LLM text for Indonesian TTS.

Two tasks in one LLM call:
  1. Formality → spoken style  ("tidak" → "nggak", "saya" → "aku", …)
  2. English phonetic respell  ("machine learning" → "masyin lerning")

Two entry points:
  • prepare(text)            — one sentence, ~200-400ms LLM call when needed
  • prepare_batch(sentences) — N sentences in a single LLM call (use this when
                               all sentences are known upfront, e.g. speak())

Both skip the LLM entirely when the input is already casual Indonesian-only
(heuristic regex gate → saves ~300ms per sentence on plain text).
"""

from __future__ import annotations

import re
from typing import Any

from loguru import logger

from app.services.llm_groq import ChatMessage

# ── Indonesian number → spoken word conversion ───────────────────────────────
_ONES = ["", "satu", "dua", "tiga", "empat", "lima", "enam", "tujuh", "delapan", "sembilan"]
_TEENS = [
    "sepuluh", "sebelas", "dua belas", "tiga belas", "empat belas", "lima belas",
    "enam belas", "tujuh belas", "delapan belas", "sembilan belas",
]


def _int_to_id(n: int) -> str:
    """Convert a non-negative integer to Indonesian spoken words."""
    if n == 0:
        return "nol"
    if n < 0:
        return "negatif " + _int_to_id(-n)
    parts: list[str] = []
    if n >= 1_000_000_000:
        parts.append(_int_to_id(n // 1_000_000_000) + " miliar")
        n %= 1_000_000_000
    if n >= 1_000_000:
        parts.append(_int_to_id(n // 1_000_000) + " juta")
        n %= 1_000_000
    if n >= 1000:
        th = n // 1000
        parts.append("seribu" if th == 1 else _int_to_id(th) + " ribu")
        n %= 1000
    if n >= 100:
        h = n // 100
        parts.append("seratus" if h == 1 else _ONES[h] + " ratus")
        n %= 100
    if n >= 20:
        parts.append(_ONES[n // 10] + " puluh")
        if n % 10:
            parts.append(_ONES[n % 10])
    elif n >= 10:
        parts.append(_TEENS[n - 10])
    elif n:
        parts.append(_ONES[n])
    return " ".join(parts)


def _expand_numbers_id(text: str) -> str:
    """Expand numbers in text to Indonesian spoken form before TTS.

    Order matters: percentages → Indonesian dot-thousands → comma-decimal → plain integers.
    """
    # "75%" → "tujuh puluh lima persen"
    text = re.sub(
        r"\b(\d+)%",
        lambda m: _int_to_id(int(m.group(1))) + " persen",
        text,
    )
    # "1.000.000" / "53.250" (Indonesian dot-thousands) → expanded words
    text = re.sub(
        r"\b(\d{1,3}(?:\.\d{3})+)\b",
        lambda m: _int_to_id(int(m.group(0).replace(".", ""))),
        text,
    )
    # "3,14" (Indonesian comma-decimal) → "tiga koma satu empat"
    def _comma_decimal(m: re.Match) -> str:
        iw = _int_to_id(int(m.group(1)))
        fw = " ".join(_int_to_id(int(d)) for d in m.group(2))
        return f"{iw} koma {fw}"
    text = re.sub(r"\b(\d+),(\d+)\b", _comma_decimal, text)
    # Plain integers and years: "2024" → "dua ribu dua puluh empat"
    text = re.sub(r"\b(\d+)\b", lambda m: _int_to_id(int(m.group(1))), text)
    return text


# ── Markdown / code stripping ────────────────────────────────────────────────
# These patterns fire BEFORE the LLM naturalization step to remove formatting
# that would sound broken when read aloud by TTS (backticks, code blocks, etc.)

_CODE_BLOCK_RE = re.compile(r"```[^\n]*\n?.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")
_MARKDOWN_HEADER_RE = re.compile(r"^#+\s+", re.MULTILINE)
_MARKDOWN_BOLD_RE = re.compile(r"\*\*([^*\n]+)\*\*")
_MARKDOWN_ITALIC_RE = re.compile(r"\*([^*\n]+)\*")
_MARKDOWN_BULLETS_RE = re.compile(r"^[\*\-]\s+", re.MULTILINE)


def _strip_for_speech(text: str) -> str:
    """Remove markdown/code formatting that sounds wrong when read aloud."""
    text = _CODE_BLOCK_RE.sub("", text)            # drop ``` blocks entirely
    text = _INLINE_CODE_RE.sub(r"\1", text)        # `var` → var
    text = _MARKDOWN_HEADER_RE.sub("", text)       # ## Heading → Heading
    text = _MARKDOWN_BOLD_RE.sub(r"\1", text)      # **word** → word
    text = _MARKDOWN_ITALIC_RE.sub(r"\1", text)    # *word* → word
    text = _MARKDOWN_BULLETS_RE.sub("", text)      # - item → item
    return text.strip()


# Heuristic gate: if ANY of these English patterns appear, the sentence is
# sent to the LLM for phonetic respelling before id-ID-GadisNeural reads it.
# A false-positive (calling LLM unnecessarily) costs ~300 ms; a false-negative
# (missing an English word) causes mispronunciation. Err on the inclusive side.
_LIKELY_ENGLISH = re.compile(
    r"\b("
    r"machine learning|deep learning|neural network|"
    r"computer vision|natural language|"
    r"object detection|image recognition|"
    r"algorithm|framework|library|dataset|"
    r"frontend|backend|fullstack|"
    r"python|javascript|typescript|"
    r"docker|kubernetes|"
    r"API|SDK|GPU|CPU|AI|ML|DL|"
    r"JSON|HTML|CSS|REST|HTTP|"
    r"pytorch|tensorflow|langchain|"
    r"machine|learning|deep|neural|"
    r"edge[- ]?tts"
    r")\b",
    re.IGNORECASE,
)

# Formal Bahasa Indonesia markers — triggers naturalization rewrite.
_FORMAL_ID = re.compile(
    r"\b("
    r"tidak|saya|anda|akan|dapat|adalah|merupakan|"
    r"sehingga|namun|tetapi|apabila|jika|maupun|"
    r"sangat|sekali|sangatlah|sebagai|secara|"
    r"sudah|belum|akan|telah|sedang|"
    r"bagaimana|mengapa|seperti|sebagaimana|"
    r"silakan|silahkan|terima\s+kasih|mohon"
    r")\b",
    re.IGNORECASE,
)

_SYSTEM_PROMPT = (
    "Kamu adalah editor teks untuk text-to-speech Bahasa Indonesia. "
    "Tugasmu: rewrite teks input agar terdengar NATURAL diucapkan, dengan DUA langkah:\n\n"
    "1. NATURALISASI: ubah kata/struktur tulis-formal menjadi gaya LISAN sehari-hari.\n"
    "   - 'tidak' -> 'nggak' atau 'gak'\n"
    "   - 'saya' -> 'aku' (kecuali konteks sangat formal)\n"
    "   - 'apabila' / 'jika' -> 'kalau'\n"
    "   - 'bagaimana' -> 'gimana'\n"
    "   - 'sudah' -> 'udah'\n"
    "   - 'akan' -> 'bakal' atau dihilangkan\n"
    "   - 'merupakan' -> 'adalah' atau dihilangkan ('X merupakan Y' -> 'X itu Y')\n"
    "   - 'sehingga' -> 'jadi'\n"
    "   - 'namun' -> 'tapi'\n"
    "   - 'tetapi' -> 'tapi'\n"
    "   - 'sangat' -> 'banget' (di akhir frasa) atau dihilangkan\n"
    "   - Pecah kalimat panjang jadi 2-3 kalimat pendek kalau perlu, biar mengalir.\n"
    "   - JANGAN menambah informasi baru. JANGAN menghilangkan fakta penting.\n"
    "   - Pertahankan nama orang, tempat, istilah teknis.\n\n"
    "2. RESPELL INGGRIS: kata Inggris diubah ke ejaan fonetik Indonesia "
    "supaya pembaca Indonesia mengucapkannya seperti bunyi Inggris aslinya.\n"
    "   - 'machine learning' -> 'masyin lerning'\n"
    "   - 'You Only Look Once' -> 'Yu Onli Luk Wans'\n"
    "   - 'deep learning' -> 'diip lerning'\n"
    "   - 'object detection' -> 'obyek diteksyen'\n"
    "   - 'computer vision' -> 'kompyuter visyen'\n"
    "   - 'AI' -> 'ei ai', 'API' -> 'ei pi ai'\n"
    "   - Nama produk/orang (Pointer, YOLO, Polinela) -> biarkan apa adanya.\n\n"
    "OUTPUT: HANYA teks hasil rewrite, tanpa penjelasan, tanpa label, tanpa tanda kutip.\n\n"
    "Contoh lengkap:\n"
    "INPUT: 'Saya akan menjelaskan apa itu machine learning. Machine learning merupakan teknik yang sangat populer.'\n"
    "OUTPUT: 'Aku jelasin ya apa itu masyin lerning. Masyin lerning itu teknik yang lagi populer banget.'\n\n"
    "INPUT: 'Apabila kamu tidak mengerti, silakan bertanya kembali.'\n"
    "OUTPUT: 'Kalau kamu nggak ngerti, tanya aja lagi.'\n\n"
    "INPUT: 'aku suka kopi'\n"
    "OUTPUT: 'aku suka kopi'  (sudah natural, biarkan)"
)

_BATCH_SYSTEM_PROMPT = (
    "Kamu adalah editor teks untuk text-to-speech Bahasa Indonesia. "
    "Input adalah daftar bernomor. Rewrite SETIAP baris menggunakan dua aturan:\n\n"
    "1. NATURALISASI: tulis-formal → gaya lisan (tidak→nggak, saya→aku, bagaimana→gimana, dll)\n"
    "2. RESPELL INGGRIS: kata Inggris → ejaan fonetik (machine learning→masyin lerning, AI→ei ai, dll)\n\n"
    "ATURAN KETAT:\n"
    "- Output HARUS berupa daftar bernomor dengan jumlah baris SAMA PERSIS dengan input.\n"
    "- Format setiap baris: 'N. teks hasil rewrite'\n"
    "- JANGAN gabungkan atau pisahkan baris.\n"
    "- JANGAN tambah penjelasan atau tanda kutip.\n\n"
    "Contoh input:\n"
    "1. Machine learning merupakan teknik yang sangat populer.\n"
    "2. Saya akan menjelaskan cara kerjanya.\n"
    "3. Tapi kalau kamu nggak paham, tanya aja.\n\n"
    "Contoh output:\n"
    "1. Masyin lerning itu teknik yang lagi populer banget.\n"
    "2. Aku jelasin cara kerjanya.\n"
    "3. Tapi kalau kamu nggak paham, tanya aja."
)

# Matches "N. content" lines in batch LLM output.
_NUMBERED_LINE = re.compile(r"^\d+\.\s*(.+)$")

def _needs_prep(text: str) -> bool:
    return bool(_LIKELY_ENGLISH.search(text)) or bool(_FORMAL_ID.search(text))


class SpeechPrepAgent:
    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def prepare(self, text: str) -> str:
        """Prepare a single sentence. Returns original if no prep needed."""
        if not text:
            return text
        text = _strip_for_speech(text)
        text = _expand_numbers_id(text)
        if not text or not _needs_prep(text):
            return text
        messages: list[ChatMessage] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ]
        try:
            raw = await self._llm.generate(messages, temperature=0.2, max_tokens=400)
        except Exception as exc:
            logger.warning(f"[SpeechPrepAgent] prepare failed, using original: {exc}")
            return text
        out = raw.strip().strip("\"'")
        return out if out else text

    async def prepare_batch(self, sentences: list[str]) -> list[str]:
        """Prepare all sentences that need it in a single LLM call.

        Sentences that pass the heuristic gate are left as-is and not sent to
        the LLM, keeping max_tokens low. Returns a list of the same length and
        order as the input.
        """
        if not sentences:
            return []

        # Strip markdown/code formatting, then expand numbers to spoken form
        sentences = [_strip_for_speech(s) for s in sentences]
        sentences = [_expand_numbers_id(s) for s in sentences]

        needs = [i for i, s in enumerate(sentences) if _needs_prep(s)]
        if not needs:
            return list(sentences)

        numbered = "\n".join(f"{n + 1}. {sentences[i]}" for n, i in enumerate(needs))
        messages: list[ChatMessage] = [
            {"role": "system", "content": _BATCH_SYSTEM_PROMPT},
            {"role": "user", "content": numbered},
        ]
        try:
            raw = await self._llm.generate(
                messages, temperature=0.2, max_tokens=150 * len(needs)
            )
        except Exception as exc:
            logger.warning(
                f"[SpeechPrepAgent] prepare_batch failed, using originals: {exc}"
            )
            return list(sentences)

        parsed: list[str] = []
        for line in raw.strip().splitlines():
            m = _NUMBERED_LINE.match(line.strip())
            if m:
                parsed.append(m.group(1).strip().strip("\"'"))

        if len(parsed) != len(needs):
            logger.warning(
                f"[SpeechPrepAgent] batch parse mismatch: "
                f"expected {len(needs)}, got {len(parsed)} — using originals"
            )
            return list(sentences)

        result = list(sentences)
        for n, orig_idx in enumerate(needs):
            if parsed[n]:
                result[orig_idx] = parsed[n]
        return result


_singleton: SpeechPrepAgent | None = None


def get_speech_prep_agent() -> SpeechPrepAgent:
    global _singleton
    if _singleton is None:
        from app.services.llm_groq import get_llm
        _singleton = SpeechPrepAgent(get_llm())
    return _singleton
