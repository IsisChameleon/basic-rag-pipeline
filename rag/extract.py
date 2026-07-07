from __future__ import annotations

import re
from dataclasses import dataclass

from markdownify import markdownify
from readability import Document

# Site title suffixes like "Some Page \ Anthropic" or "Some Page | Site" --
# strip everything from a spaced separator onward to get just the page title.
_TITLE_SUFFIX_RE = re.compile(r"\s+[\\|·–—]\s+.*$")


@dataclass
class ExtractedPage:
    title: str
    markdown: str


def extract_page(html: str, url: str) -> ExtractedPage | None:
    """Pull clean Markdown + title out of a fetched page. Returns None if the
    page has no extractable main content.

    Two stages: readability isolates the main article (dropping nav, footer,
    and CTA boilerplate) while preserving the heading structure in the DOM,
    then markdownify converts that HTML fragment to Markdown faithfully --
    headings, tables, and code blocks included. (trafilatura, used earlier,
    silently demoted nearly every real heading to a paragraph, which made the
    chunk heading_path metadata useless -- see build_log.md.)
    """
    doc = Document(html)
    article_html = doc.summary()
    markdown = markdownify(article_html, heading_style="ATX").strip()
    if not markdown:
        return None

    raw_title = doc.short_title() or url
    title = _TITLE_SUFFIX_RE.sub("", raw_title).strip() or url
    return ExtractedPage(title=title, markdown=markdown)
