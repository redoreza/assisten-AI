"""SearchAgent — classify → search → evaluate → optional retry.

Extracted from Orchestrator so the search pipeline is independently testable
and can later run in parallel with other agents.

Interface:
    agent = SearchAgent(llm)
    result = await agent.run(user_message, history=[...])
    # result is None  → no search needed
    # result is dict  → inject result["context"] into LLM messages,
    #                    yield each event in result["ws_events"]
"""

from __future__ import annotations

from typing import Any, TypedDict

from loguru import logger

from app.services.llm_groq import ChatMessage
from app.services.search_tavily import get_search


class SearchResult(TypedDict):
    context: str
    sources: list[dict]
    queries: list[str]
    ws_events: list[dict]


class SearchAgent:
    def __init__(self, llm: Any) -> None:
        self._llm = llm

    async def run(
        self,
        user_message: str,
        history: list[ChatMessage] | None = None,
    ) -> SearchResult | None:
        """Full pipeline: classify → search → evaluate. Used when no hold-on
        phrase is needed (e.g. non-streaming callers)."""
        query = await self.classify(user_message, history)
        if not query:
            return None
        return await self.run_with_query(query, user_message)

    async def run_with_query(
        self,
        query: str,
        user_message: str,
    ) -> SearchResult | None:
        """Search + evaluate pipeline starting from an already-classified query.

        Called by the orchestrator after it has already obtained the query via
        classify() so it can emit a hold-on phrase in parallel.
        """
        logger.info(f"[SearchAgent] classifier triggered search: {query!r}")
        ws_events: list[dict] = [
            {"type": "tool_use", "tools": ["search_web"], "query": query}
        ]

        try:
            result = await get_search().search_and_extract(query)
        except Exception as exc:
            logger.warning(f"[SearchAgent] search failed: {exc}")
            return None

        context = result["llm_context"]
        sources = result["sources"]
        queries = [query]

        logger.info(
            f"[SearchAgent] search 1 ({len(context)} chars, {len(sources)} sources)"
        )

        retry_query = await self._evaluate_sufficiency(user_message, context)
        if retry_query and retry_query != query:
            logger.info(f"[SearchAgent] retry search with: {retry_query!r}")
            ws_events.append(
                {"type": "tool_use", "tools": ["search_web"], "query": retry_query}
            )
            queries.append(retry_query)
            try:
                retry_result = await get_search().search_and_extract(retry_query)
                if retry_result["sources"]:
                    context = (
                        retry_result["llm_context"]
                        + "\n\n--- additional context from previous search ---\n"
                        + context
                    )
                    seen_urls = {s["url"] for s in retry_result["sources"]}
                    for s in sources:
                        if s["url"] not in seen_urls:
                            retry_result["sources"].append(s)
                            seen_urls.add(s["url"])
                    sources = retry_result["sources"]
                    logger.info(
                        f"[SearchAgent] search 2 merged "
                        f"({len(context)} chars, {len(sources)} sources)"
                    )
            except Exception as exc:
                logger.warning(f"[SearchAgent] retry search failed: {exc}")

        if sources:
            ws_events.append(
                {
                    "type": "search_sources",
                    "query": queries[-1],
                    "sources": sources,
                }
            )

        return {
            "context": context,
            "sources": sources,
            "queries": queries,
            "ws_events": ws_events,
        }

    async def classify(
        self, user_text: str, history: list[ChatMessage] | None = None
    ) -> str | None:
        if not user_text.strip():
            return None

        context_lines: list[str] = []
        if history:
            for turn in history[-4:]:
                role = turn.get("role", "?")
                content = (turn.get("content") or "").strip()
                if not content:
                    continue
                if len(content) > 200:
                    content = content[:197] + "…"
                speaker = "User" if role == "user" else "Pointer"
                context_lines.append(f"{speaker}: {content}")
        context_block = ""
        if context_lines:
            context_block = (
                "\nKONTEKS PERCAKAPAN (jangan jawab ini, hanya untuk rewrite query):\n"
                + "\n".join(context_lines)
                + "\n"
            )

        messages: list[ChatMessage] = [
            {
                "role": "system",
                "content": (
                    "Tugasmu: tentukan apakah pesan user terakhir butuh pencarian web "
                    "untuk dijawab dengan benar.\n"
                    "Jawab PERSIS satu baris, salah satu format ini, tanpa apapun lain:\n"
                    "  SEARCH: <query>\n"
                    "  NO\n\n"
                    "PENTING — query harus SELF-CONTAINED. Pakai konteks percakapan untuk:\n"
                    "- Mengganti kata ganti ('itu', 'nya', 'tersebut') dengan subjek aslinya\n"
                    "- Tambah nama institusi/lokasi yang sedang dibahas (mis. nama kampus)\n"
                    "- Lengkapi follow-up: 'sejarahnya?' -> 'sejarah X' (X dari konteks)\n\n"
                    "PRINSIP UTAMA: Kalau ragu, pilih SEARCH. Lebih baik cari dan ternyata "
                    "sudah tahu, daripada jawab tanpa data dan ternyata salah.\n\n"
                    "Pakai SEARCH untuk:\n"
                    "- Semua pertanyaan faktual: 'siapa', 'apa', 'kapan', 'di mana', 'berapa'\n"
                    "- Fakta terkini: berita, harga, cuaca, kurs, jabatan, posisi, status\n"
                    "- Tokoh, organisasi, produk, tempat, nama apapun\n"
                    "- 'apa itu X' — SELALU cari kecuali X adalah kata sehari-hari\n"
                    "- Penjelasan konsep, istilah teknis, singkatan\n"
                    "- Event, jadwal, pengumuman, peraturan\n"
                    "- Detail institusi/program: prodi, tahun berdiri, akreditasi, biaya, dll\n"
                    "- Follow-up topik yang sudah dibahas (sinonim, detail lanjut, contoh)\n"
                    "- Perbandingan, rekomendasi, 'yang terbaik', 'paling populer'\n\n"
                    "Pakai NO HANYA untuk:\n"
                    "- Sapaan murni: 'halo', 'hai', 'selamat pagi', 'apa kabar'\n"
                    "- Pertanyaan tentang dirimu/sifat/persona Pointer\n"
                    "- Perintah perilaku: 'panggil aku X', 'jangan begitu', 'berbicara lebih pelan'\n"
                    "- Percakapan sosial tanpa fakta: 'makasih', 'oke', 'paham'\n"
                    "- Klarifikasi ulang: 'apa maksudmu?', 'bisa diulang?'\n\n"
                    "Contoh:\n"
                    "- 'halo apa kabar' -> NO\n"
                    "- 'makasih ya' -> NO\n"
                    "- 'siapa presiden indonesia sekarang' -> SEARCH: presiden indonesia 2026\n"
                    "- 'apa itu machine learning' -> SEARCH: machine learning pengertian\n"
                    "- 'ceritakan tentang photosynthesis' -> SEARCH: proses fotosintesis\n"
                    "- (Konteks: kampus Polinela) + 'jumlah prodi?' -> SEARCH: jumlah program studi Polinela Politeknik Negeri Lampung\n"
                    "- (Konteks: prodi Sains Data Terapan Polinela) + 'sejarahnya?' -> SEARCH: sejarah program studi Sains Data Terapan Polinela\n"
                    "- (Konteks: deteksi objek) + 'YOLO?' -> SEARCH: YOLO object detection model\n"
                    "- 'berapa penduduk indonesia' -> SEARCH: jumlah penduduk indonesia 2025\n"
                    "- 'jam berapa sekarang' -> NO\n"
                    "- 'ganti namaku jadi reza' -> NO"
                    + context_block
                ),
            },
            {"role": "user", "content": user_text},
        ]
        try:
            raw = await self._llm.generate(messages, temperature=0.0, max_tokens=48)
        except Exception as exc:
            logger.warning(f"[SearchAgent] _classify LLM failed: {exc}")
            return None
        line = raw.strip().splitlines()[0].strip() if raw.strip() else ""
        if line.upper() == "NO" or line.upper().startswith("NO"):
            return None
        if not line.upper().startswith("SEARCH"):
            return None
        after = line.split(":", 1)[1] if ":" in line else line.split(" ", 1)[1] if " " in line else ""
        query = after.strip().strip('"').strip("'")
        if not query or len(query) > 200:
            return None
        return query

    async def _evaluate_sufficiency(
        self, user_message: str, search_context: str
    ) -> str | None:
        if len(search_context) > 8000:
            return None
        messages: list[ChatMessage] = [
            {
                "role": "system",
                "content": (
                    "Kamu adalah evaluator hasil pencarian web. Tugasmu:\n"
                    "Lihat (a) pertanyaan user dan (b) hasil pencarian yang sudah ada.\n"
                    "Tentukan: apakah hasil ini SUDAH CUKUP untuk menjawab user secara faktual?\n\n"
                    "Jawab PERSIS satu baris, salah satu format:\n"
                    "  OK\n"
                    "  RETRY: <query baru>\n\n"
                    "Pakai RETRY hanya kalau hasil:\n"
                    "- Tidak ada/error/kosong\n"
                    "- Topiknya berbeda dari pertanyaan user (off-target)\n"
                    "- Cuma menyentuh permukaan, padahal user minta detail spesifik\n\n"
                    "Pakai OK kalau hasil sudah memuat fakta yang user butuhkan, walau pendek.\n"
                    "JANGAN RETRY hanya karena ingin info lebih banyak — hanya kalau hasil benar-benar tidak menjawab.\n"
                    "Query RETRY harus SELF-CONTAINED dan spesifik."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"PERTANYAAN USER:\n{user_message}\n\n"
                    f"HASIL PENCARIAN:\n{search_context[:6000]}"
                ),
            },
        ]
        try:
            raw = await self._llm.generate(messages, temperature=0.0, max_tokens=64)
        except Exception as exc:
            logger.warning(f"[SearchAgent] _evaluate_sufficiency failed: {exc}")
            return None
        line = raw.strip().splitlines()[0].strip() if raw.strip() else ""
        if line.upper().startswith("OK"):
            return None
        if not line.upper().startswith("RETRY"):
            return None
        after = line.split(":", 1)[1] if ":" in line else line.split(" ", 1)[1] if " " in line else ""
        query = after.strip().strip('"').strip("'")
        if not query or len(query) > 200:
            return None
        return query
