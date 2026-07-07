from rag.chunk import chunk_markdown


def _word_count(text: str) -> int:
    return len(text.split())


def test_splits_on_headings() -> None:
    markdown = "# Title\n\nIntro para.\n\n## Section\n\nSection para."
    chunks = chunk_markdown(markdown, count_tokens=_word_count, target_tokens=100, overlap_tokens=10)
    heading_paths = {c.heading_path for c in chunks}
    assert "Title" in heading_paths
    assert "Title > Section" in heading_paths


def test_table_is_never_split_even_if_oversized() -> None:
    table = "\n".join(f"| a{i} | b{i} |" for i in range(50))
    markdown = f"# Title\n\n{table}"
    chunks = chunk_markdown(markdown, count_tokens=_word_count, target_tokens=10, overlap_tokens=2)

    table_chunks = [c for c in chunks if c.text.startswith("| a0")]
    assert len(table_chunks) == 1
    assert table_chunks[0].text.count("\n") == 49


def test_oversized_paragraph_is_sentence_split() -> None:
    sentences = " ".join(f"Sentence number {i}." for i in range(30))
    markdown = f"# Title\n\n{sentences}"
    chunks = chunk_markdown(markdown, count_tokens=_word_count, target_tokens=15, overlap_tokens=3)

    assert len(chunks) > 1
    assert all(c.heading_path == "Title" for c in chunks)
