"""Tavily web search wrapper — used as a tool by the LLM.

Tuned for an Indonesian campus assistant:
- `search_depth="advanced"` for better content extraction (slower but worth it).
- `include_domains` favoring `.ac.id` and Indonesian gov domains when the
  classifier flagged the query as campus/local context (`prefer_indonesian`).
- Returns structured dict: a compact text the LLM ingests as system context,
  PLUS the raw `sources` list so the WebSocket layer can render a "Sumber:"
  panel in the UI.

API: https://docs.tavily.com/docs/python-sdk/tavily-search/api-reference
"""

from __future__ import annotations

import asyncio
from typing import TypedDict

from loguru import logger
from tavily import TavilyClient

from app.config import settings

# Indonesian academic + government domains we bias toward for campus queries.
# Tavily's include_domains accepts both full URLs and root domains.
_ID_ACADEMIC_DOMAINS = [
    "ac.id",          # all Indonesian academic institutions
    "go.id",          # Indonesian government
    "kemdikbud.go.id",
    "kemenristekdikti.go.id",
    "kemdiktisaintek.go.id",
    "ristekdikti.go.id",
]


class SearchSource(TypedDict):
    title: str
    url: str
    snippet: str


class SearchResult(TypedDict):
    answer: str | None
    sources: list[SearchSource]
    # Pre-formatted block we splice into the LLM system prompt
    llm_context: str


class ExtractedDoc(TypedDict):
    title: str
    url: str
    full_text: str


class TavilySearch:
    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or settings.tavily_api_key
        if not key:
            raise ValueError("TAVILY_API_KEY is not set in .env")
        self._client = TavilyClient(api_key=key)

    async def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        search_depth: str = "advanced",
        prefer_indonesian: bool = True,
    ) -> SearchResult:
        """Run a search and return structured results.

        If `prefer_indonesian` we first try with a domain bias toward Indonesian
        academic/government sites; if that returns < 2 sources we fall back to
        a global search so we don't strand the user when the campus-domain hit
        rate is low.
        """
        logger.info(
            f"[tavily] query={query!r} depth={search_depth} prefer_id={prefer_indonesian}"
        )

        def _run(domains: list[str] | None) -> dict:
            kwargs: dict = {
                "query": query,
                "search_depth": search_depth,
                "max_results": max_results,
                "include_answer": True,
            }
            if domains:
                kwargs["include_domains"] = domains
            return self._client.search(**kwargs)

        biased_results: dict | None = None
        try:
            if prefer_indonesian:
                biased_results = await asyncio.to_thread(_run, _ID_ACADEMIC_DOMAINS)
                num_biased = len(biased_results.get("results", []))
                logger.debug(f"[tavily] biased search got {num_biased} results")
            if biased_results is None or len(biased_results.get("results", [])) < 2:
                # Insufficient → re-run without bias
                global_results = await asyncio.to_thread(_run, None)
                # Merge: keep biased on top, then add global ones not already present
                merged = self._merge_results(biased_results, global_results)
                final = merged if merged else global_results
            else:
                final = biased_results
        except Exception as exc:
            logger.warning(f"[tavily] search failed: {exc}")
            return {
                "answer": None,
                "sources": [],
                "llm_context": f"SEARCH_ERROR: {exc}",
            }

        return self._format(final, max_results)

    @staticmethod
    def _merge_results(
        a: dict | None, b: dict | None
    ) -> dict | None:
        """Merge two Tavily results, deduplicating by URL. Preserves order
        of `a` first, then appends novel entries from `b`."""
        if not a:
            return b
        if not b:
            return a
        seen = set()
        merged_results = []
        for r in a.get("results", []):
            url = r.get("url")
            if url and url not in seen:
                seen.add(url)
                merged_results.append(r)
        for r in b.get("results", []):
            url = r.get("url")
            if url and url not in seen:
                seen.add(url)
                merged_results.append(r)
        return {
            "answer": a.get("answer") or b.get("answer"),
            "results": merged_results,
        }

    @staticmethod
    def _format(res: dict, max_results: int) -> SearchResult:
        sources: list[SearchSource] = []
        lines: list[str] = []
        answer = (res.get("answer") or "").strip()
        if answer:
            lines.append(f"ANSWER: {answer}")
        results = res.get("results", [])
        if results:
            lines.append("SOURCES:")
            for r in results[:max_results]:
                title = (r.get("title") or "").strip()
                url = (r.get("url") or "").strip()
                content = (r.get("content") or "").strip()
                snippet = content[:277] + "…" if len(content) > 280 else content
                sources.append({"title": title, "url": url, "snippet": snippet})
                lines.append(f"- {title} ({url}): {snippet}")
        if not lines:
            return {
                "answer": None,
                "sources": [],
                "llm_context": "NO_RESULTS: search returned nothing",
            }
        return {
            "answer": answer or None,
            "sources": sources,
            "llm_context": "\n".join(lines),
        }


    async def extract(self, urls: list[str], *, max_chars: int = 2000) -> list[ExtractedDoc]:
        """Pull full content from one or more URLs via Tavily Extract.
        Returns at most `len(urls)` documents, each truncated to `max_chars`."""
        if not urls:
            return []
        logger.info(f"[tavily.extract] {len(urls)} url(s)")

        def _run() -> dict:
            return self._client.extract(urls=urls)

        try:
            res = await asyncio.to_thread(_run)
        except Exception as exc:
            logger.warning(f"[tavily.extract] failed: {exc}")
            return []
        docs: list[ExtractedDoc] = []
        for r in res.get("results", []):
            url = (r.get("url") or "").strip()
            raw = (r.get("raw_content") or "").strip()
            if not raw:
                continue
            if len(raw) > max_chars:
                raw = raw[: max_chars - 1] + "…"
            docs.append({"title": url.split("/")[-1] or url, "url": url, "full_text": raw})
        return docs

    async def search_and_extract(
        self,
        query: str,
        *,
        max_results: int = 5,
        extract_top_n: int = 2,
        prefer_indonesian: bool = True,
    ) -> SearchResult:
        """Convenience: search, then deep-extract the top N URLs and weave
        their full content into llm_context. Bigger context but more accurate
        answers for fact-specific questions."""
        result = await self.search(
            query,
            max_results=max_results,
            search_depth="advanced",
            prefer_indonesian=prefer_indonesian,
        )
        if not result["sources"]:
            return result
        top_urls = [s["url"] for s in result["sources"][:extract_top_n] if s["url"]]
        if not top_urls:
            return result
        docs = await self.extract(top_urls)
        if not docs:
            return result
        # Append the deep content to the LLM context
        extra_lines = ["", "DEEP CONTENT (top sources, full text):"]
        for d in docs:
            extra_lines.append(f"--- {d['url']} ---")
            extra_lines.append(d["full_text"])
        result["llm_context"] = result["llm_context"] + "\n" + "\n".join(extra_lines)
        return result


_singleton: TavilySearch | None = None


def get_search() -> TavilySearch:
    global _singleton
    if _singleton is None:
        _singleton = TavilySearch()
    return _singleton
