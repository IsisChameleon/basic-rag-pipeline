from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import httpx
from loguru import logger

from rag import chunk, discover, embeddings, extract, fetch, store, vectorstore
from rag.store import ChunkMetadata


@dataclass
class IngestSummary:
    pages_discovered: int
    pages_ingested: int
    chunks_stored: int


async def ingest_page_with_url(section_url: str) -> IngestSummary:
    logger.info("Starting ingest for {}", section_url)
    urls = await discover.discover_section_urls(section_url)
    if not urls:
        logger.warning("Nothing to ingest for {} -- no pages discovered", section_url)
        return IngestSummary(pages_discovered=0, pages_ingested=0, chunks_stored=0)

    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        htmls = await fetch.fetch_many(client, urls)
    logger.info("Fetched {} page(s); extracting and chunking", sum(h is not None for h in htmls))

    conn = store.get_connection()
    pages_ingested = 0
    chunks_stored = 0

    for url, html in zip(urls, htmls, strict=True):
        if html is None:
            continue  # fetch failed for this page; already logged in rag.fetch

        page = extract.extract_page(html, url)
        if page is None:
            logger.debug("Skipping {} -- no extractable content", url)
            continue

        chunks = chunk.chunk_markdown(page.markdown, count_tokens=embeddings.count_tokens)
        if not chunks:
            logger.debug("Skipping {} -- produced no chunks", url)
            continue

        store.delete_chunks_for_url(conn, url)
        vectorstore.delete_by_url(url)

        # Only one page's chunks/vectors are held in memory at a time (this
        # loop stores each page before moving to the next), not the whole
        # section. At 384-dim float32 a vector is ~1.5KB, and a ~350-token
        # chunk of text is a similar order of magnitude, so even a page with
        # a few dozen chunks stays well under a megabyte in memory.
        texts = [c.text for c in chunks]
        vectors = embeddings.embed_documents(texts)
        content_hash = hashlib.sha256(page.markdown.encode("utf-8")).hexdigest()
        fetched_at = datetime.now(UTC).isoformat()

        ids: list[str] = []
        metadatas: list[dict] = []
        for index, c in enumerate(chunks):
            meta = ChunkMetadata(
                url=url, title=page.title, heading_path=c.heading_path, chunk_index=index
            )
            row_id = store.insert_chunk(
                conn,
                metadata=meta,
                text=c.text,
                content_hash=content_hash,
                fetched_at=fetched_at,
            )
            ids.append(str(row_id))
            metadatas.append(asdict(meta))

        vectorstore.add_chunks(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
        pages_ingested += 1
        chunks_stored += len(chunks)

    conn.commit()
    conn.close()
    logger.info(
        "Ingest complete for {}: {}/{} pages ingested, {} chunks stored",
        section_url,
        pages_ingested,
        len(urls),
        chunks_stored,
    )
    return IngestSummary(
        pages_discovered=len(urls),
        pages_ingested=pages_ingested,
        chunks_stored=chunks_stored,
    )
