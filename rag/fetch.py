from __future__ import annotations

import asyncio

import httpx
from loguru import logger

# Cap concurrent requests to the target site -- fetch_many can be handed
# dozens of URLs from one sitemap section, and firing them all at once would
# be an unfriendly (and blockable) way to crawl someone else's site. The
# semaphore is acquired once per URL in `_bounded_fetch`, so at most this many
# `fetch_html` calls are in flight at a time; every other queued URL blocks on
# `async with semaphore` until a slot frees up.
_MAX_CONCURRENT_FETCHES = 5


async def fetch_html(client: httpx.AsyncClient, url: str) -> str:
    response = await client.get(url)
    response.raise_for_status()
    return response.text


async def fetch_many(client: httpx.AsyncClient, urls: list[str]) -> list[str | None]:
    """Fetch every URL, capped at `_MAX_CONCURRENT_FETCHES` concurrent requests.

    A single failing URL (404, timeout, etc.) does not abort the rest of the
    batch: it is logged and returned as None so the caller can skip that page
    while still ingesting everything else in the section.
    """
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_FETCHES)

    async def _bounded_fetch(url: str) -> str | None:
        async with semaphore:
            try:
                return await fetch_html(client, url)
            except httpx.HTTPError as exc:
                logger.warning(f"Skipping {url}: fetch failed ({exc})")
                return None

    return await asyncio.gather(*(_bounded_fetch(url) for url in urls))
