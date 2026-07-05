from __future__ import annotations

import asyncio

import httpx

# Cap concurrent requests to the target site -- fetch_many can be handed
# dozens of URLs from one sitemap section, and firing them all at once would
# be an unfriendly (and blockable) way to crawl someone else's site.
_MAX_CONCURRENT_FETCHES = 5


async def fetch_html(client: httpx.AsyncClient, url: str) -> str:
    response = await client.get(url)
    response.raise_for_status()
    return response.text


async def fetch_many(client: httpx.AsyncClient, urls: list[str]) -> list[str]:
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_FETCHES)

    async def _bounded_fetch(url: str) -> str:
        async with semaphore:
            return await fetch_html(client, url)

    return await asyncio.gather(*(_bounded_fetch(url) for url in urls))
