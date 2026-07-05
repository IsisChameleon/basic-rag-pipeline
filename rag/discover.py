from __future__ import annotations

import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import httpx

_SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


async def discover_section_urls(client: httpx.AsyncClient, section_url: str) -> list[str]:
    """Fetch the site's sitemap.xml and return every URL whose path shares the
    given section_url's path prefix (e.g. .../engineering discovers every
    .../engineering/... entry, plus the section page itself if listed)."""
    parsed = urlparse(section_url)
    sitemap_url = f"{parsed.scheme}://{parsed.netloc}/sitemap.xml"
    response = await client.get(sitemap_url)
    response.raise_for_status()

    root = ET.fromstring(response.text)
    prefix = parsed.path.rstrip("/")

    urls = []
    for url_el in root.findall("sm:url", _SITEMAP_NS):
        loc_el = url_el.find("sm:loc", _SITEMAP_NS)
        if loc_el is None or not loc_el.text:
            continue
        loc_path = urlparse(loc_el.text).path.rstrip("/")
        if loc_path == prefix or loc_path.startswith(prefix + "/"):
            urls.append(loc_el.text)
    return urls
