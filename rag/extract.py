from __future__ import annotations

from dataclasses import dataclass

import trafilatura


@dataclass
class ExtractedPage:
    title: str
    markdown: str


def extract_page(html: str, url: str) -> ExtractedPage | None:
    """Pull clean Markdown + title out of a fetched page. Returns None if
    trafilatura can't find a main-content article (e.g. a non-article page).

    Uses trafilatura.extract() for the Markdown body -- bare_extraction()'s
    .text field is only populated for output_format="txt"/"xml", not
    "markdown" (verified empirically against trafilatura 2.1.0) -- and
    extract_metadata() separately for the title.
    """
    markdown = trafilatura.extract(
        html,
        url=url,
        output_format="markdown",
        include_tables=True,
    )
    if not markdown:
        return None

    metadata = trafilatura.extract_metadata(html, default_url=url)
    title = metadata.title if metadata and metadata.title else url
    return ExtractedPage(title=title, markdown=markdown)
