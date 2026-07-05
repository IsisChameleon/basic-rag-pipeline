import httpx

from rag.discover import discover_section_urls

_SITEMAP_XML = """<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
<url><loc>https://example.com/</loc></url>
<url><loc>https://example.com/docs</loc></url>
<url><loc>https://example.com/docs/page-1</loc></url>
<url><loc>https://example.com/docs/page-2</loc></url>
<url><loc>https://example.com/careers</loc></url>
</urlset>"""


async def test_discover_filters_by_path_prefix() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/sitemap.xml"
        return httpx.Response(200, text=_SITEMAP_XML)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        urls = await discover_section_urls(client, "https://example.com/docs")

    assert urls == [
        "https://example.com/docs",
        "https://example.com/docs/page-1",
        "https://example.com/docs/page-2",
    ]
