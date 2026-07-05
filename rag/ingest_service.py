from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from rag import chunk as chunk_mod
from rag import discover, embeddings, extract, fetch, store, vectorstore


@dataclass
class IngestSummary:
    pages_discovered: int
    pages_ingested: int
    chunks_stored: int


async def ingest_section(section_url: str) -> IngestSummary:
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
        urls = await discover.discover_section_urls(client, section_url)
        htmls = await fetch.fetch_many(client, urls)

    conn = store.get_connection()
    pages_ingested = 0
    chunks_stored = 0

    for url, html in zip(urls, htmls, strict=True):
        page = extract.extract_page(html, url)
        if page is None:
            continue

        chunks = chunk_mod.chunk_markdown(page.markdown, count_tokens=embeddings.count_tokens)
        if not chunks:
            continue

        store.delete_chunks_for_url(conn, url)
        vectorstore.delete_by_url(url)

        texts = [c.text for c in chunks]
        vectors = embeddings.embed_documents(texts)
        content_hash = hashlib.sha256(page.markdown.encode("utf-8")).hexdigest()
        fetched_at = datetime.now(UTC).isoformat()

        ids: list[str] = []
        metadatas: list[dict] = []
        for index, c in enumerate(chunks):
            row_id = store.insert_chunk(
                conn,
                url=url,
                title=page.title,
                heading_path=c.heading_path,
                chunk_index=index,
                text=c.text,
                content_hash=content_hash,
                fetched_at=fetched_at,
            )
            ids.append(str(row_id))
            metadatas.append(
                {
                    "url": url,
                    "title": page.title,
                    "heading_path": c.heading_path,
                    "chunk_index": index,
                }
            )

        vectorstore.add_chunks(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
        pages_ingested += 1
        chunks_stored += len(chunks)

    conn.commit()
    conn.close()
    return IngestSummary(
        pages_discovered=len(urls),
        pages_ingested=pages_ingested,
        chunks_stored=chunks_stored,
    )
