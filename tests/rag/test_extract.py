from rag import extract

# readability only treats a region as the main article once it has enough
# text, so the body paragraphs below are padded to clear that threshold.
_PARAGRAPH = (
    "Contextual Retrieval improves the retrieval step in RAG by prepending "
    "chunk-specific context before embedding. This significantly reduces the "
    "number of failed retrievals across a wide range of knowledge bases. "
)

_HTML = f"""
<html>
  <head><title>Contextual Retrieval in AI Systems \\ Anthropic</title></head>
  <body>
    <nav>Home Products Pricing Contact</nav>
    <article>
      <h1>Introducing Contextual Retrieval</h1>
      <p>{_PARAGRAPH * 4}</p>
      <h2>A primer on RAG</h2>
      <p>{_PARAGRAPH * 4}</p>
      <h3>The context conundrum</h3>
      <p>{_PARAGRAPH * 4}</p>
    </article>
    <footer>Get the developer newsletter. Delivered monthly.</footer>
  </body>
</html>
"""


def test_extract_preserves_heading_structure() -> None:
    """Regression guard: the previous extractor silently demoted every real
    heading to a paragraph, so heading_path metadata was useless. The real
    article headings must survive as Markdown ATX headings."""
    page = extract.extract_page(_HTML, "https://example.com/contextual-retrieval")
    assert page is not None
    assert "## A primer on RAG" in page.markdown
    assert "### The context conundrum" in page.markdown


def test_extract_strips_site_suffix_from_title() -> None:
    page = extract.extract_page(_HTML, "https://example.com/contextual-retrieval")
    assert page is not None
    assert page.title == "Contextual Retrieval in AI Systems"
