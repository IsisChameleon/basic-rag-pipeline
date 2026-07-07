from __future__ import annotations

import asyncio
from urllib.parse import urlparse

from loguru import logger
from usp.tree import sitemap_tree_for_homepage


async def discover_section_urls(section_url: str) -> list[str]:
    """Return every URL from the site's sitemaps whose path shares the given
    section_url's path prefix (e.g. .../docs discovers every .../docs/... entry,
    plus the section page itself if listed).

    Delegates sitemap discovery/parsing to ultimate-sitemap-parser, which
    handles the awkward parts we'd otherwise reinvent: sitemap indexes, gzipped
    sitemaps, robots.txt-declared sitemaps, plain-text sitemaps, and nesting.
    """
    parsed = urlparse(section_url)
    homepage = f"{parsed.scheme}://{parsed.netloc}/"
    logger.info("Discovering pages under {} via sitemaps of {}", section_url, homepage)

    # usp does its own synchronous HTTP + sitemap discovery -- run it off the
    # event loop so it doesn't block other in-flight requests.
    tree = await asyncio.to_thread(sitemap_tree_for_homepage, homepage)
    all_urls = sorted({page.url for page in tree.all_pages()})

    prefix = parsed.path.rstrip("/")
    matched = [
        url
        for url in all_urls
        if (path := urlparse(url).path.rstrip("/")) == prefix or path.startswith(prefix + "/")
    ]
    logger.info(
        "Sitemaps listed {} URL(s); {} match prefix {!r}", len(all_urls), len(matched), prefix
    )
    if not matched:
        logger.warning(
            "No URLs matched {!r} -- nothing will be ingested. Check the section URL's path.",
            prefix,
        )
    return matched
