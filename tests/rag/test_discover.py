from types import SimpleNamespace

from rag import discover


def _fake_tree(urls: list[str]):
    pages = [SimpleNamespace(url=u) for u in urls]
    return SimpleNamespace(all_pages=lambda: iter(pages))


async def test_discover_filters_by_path_prefix(monkeypatch) -> None:
    # usp does its own HTTP + sitemap discovery -- mock that library boundary
    # (patched where it's used) and test our prefix filtering. usp itself is
    # what handles sitemap-index/gzip/robots.txt, verified against a live site.
    all_urls = [
        "https://example.com/",
        "https://example.com/docs",
        "https://example.com/docs/page-1",
        "https://example.com/docs/page-2",
        "https://example.com/careers",
    ]

    captured = {}

    def fake_tree_for_homepage(homepage: str):
        captured["homepage"] = homepage
        return _fake_tree(all_urls)

    monkeypatch.setattr(discover, "sitemap_tree_for_homepage", fake_tree_for_homepage)

    urls = await discover.discover_section_urls("https://example.com/docs")

    assert captured["homepage"] == "https://example.com/"
    assert urls == [
        "https://example.com/docs",
        "https://example.com/docs/page-1",
        "https://example.com/docs/page-2",
    ]


async def test_discover_returns_empty_when_no_match(monkeypatch) -> None:
    monkeypatch.setattr(
        discover,
        "sitemap_tree_for_homepage",
        lambda homepage: _fake_tree(["https://example.com/", "https://example.com/blog"]),
    )
    urls = await discover.discover_section_urls("https://example.com/docs")
    assert urls == []
