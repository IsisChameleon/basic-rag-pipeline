from rag.markdown_chunker import MarkdownChunker


def _chunker(target_tokens: int, overlap_tokens: int) -> MarkdownChunker:
    return MarkdownChunker(
        count_tokens=lambda text: len(text.split()),
        target_tokens=target_tokens,
        overlap_tokens=overlap_tokens,
    )


def test_splits_on_headings() -> None:
    markdown = "# Title\n\nIntro para.\n\n## Section\n\nSection para."
    chunks = _chunker(target_tokens=100, overlap_tokens=10).chunk(markdown)
    heading_paths = {c.heading_path for c in chunks}
    assert "Title" in heading_paths
    assert "Title > Section" in heading_paths


def test_table_is_never_split_even_if_oversized() -> None:
    table = "\n".join(f"| a{i} | b{i} |" for i in range(50))
    markdown = f"# Title\n\n{table}"
    chunks = _chunker(target_tokens=10, overlap_tokens=2).chunk(markdown)

    table_chunks = [c for c in chunks if c.text.startswith("| a0")]
    assert len(table_chunks) == 1
    assert table_chunks[0].text.count("\n") == 49


def test_oversized_paragraph_is_sentence_split() -> None:
    sentences = " ".join(f"Sentence number {i}." for i in range(30))
    markdown = f"# Title\n\n{sentences}"
    chunks = _chunker(target_tokens=15, overlap_tokens=3).chunk(markdown)

    assert len(chunks) > 1
    assert all(c.heading_path == "Title" for c in chunks)
