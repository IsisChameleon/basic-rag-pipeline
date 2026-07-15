from __future__ import annotations

import httpx
from loguru import logger

from rag import discover, extract, fetch
from rag.models import Document


class DocumentSource:
    """Web source: discovers pages under a section URL via sitemaps, fetches
    them concurrently, and extracts each to a markdown Document. A future
    PDF/GitHub/upload source only has to produce Documents the same way."""

    async def load(self, source_url: str) -> list[Document | None]:
        """One entry per discovered page, in discovery order. None marks a
        page whose fetch or extraction failed (already logged), kept so the
        caller can report discovered vs ingested counts."""
        urls = await discover.discover_section_urls(source_url)
        if not urls:
            return []

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            htmls = await fetch.fetch_many(client, urls)
        logger.info(
            "Fetched {} page(s); extracting and chunking", sum(h is not None for h in htmls)
        )

        documents: list[Document | None] = []
        for url, html in zip(urls, htmls, strict=True):
            if html is None:
                documents.append(None)  # fetch failed; already logged in rag.fetch
                continue
            page = extract.extract_page(html, url)
            if page is None:
                logger.debug("Skipping {} -- no extractable content", url)
                documents.append(None)
                continue
            documents.append(Document(uri=url, title=page.title, markdown=page.markdown))
        return documents
