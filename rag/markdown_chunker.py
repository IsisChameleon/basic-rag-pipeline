from __future__ import annotations

from collections.abc import Callable

from rag import chunk as _chunk_module
from rag.chunk import DEFAULT_OVERLAP_TOKENS, DEFAULT_TARGET_TOKENS
from rag.models import Chunk


class MarkdownChunker:
    """Structural markdown chunking (headings, atomic tables/code blocks,
    token-budgeted paragraph packing). The token counter is injected at
    construction -- typically Embedder.count_tokens, tying chunk sizing to
    the model that will embed the chunks -- so callers never pass it per
    call.

    Currently delegates to rag.chunk.chunk_markdown; that module's logic
    moves in here when the old function modules are deleted at the end of
    the refactor."""

    def __init__(
        self,
        count_tokens: Callable[[str], int],
        *,
        target_tokens: int = DEFAULT_TARGET_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    ) -> None:
        self._count_tokens = count_tokens
        self._target_tokens = target_tokens
        self._overlap_tokens = overlap_tokens

    def chunk(self, markdown: str) -> list[Chunk]:
        chunks = _chunk_module.chunk_markdown(
            markdown,
            count_tokens=self._count_tokens,
            target_tokens=self._target_tokens,
            overlap_tokens=self._overlap_tokens,
        )
        return [Chunk(text=c.text, heading_path=c.heading_path) for c in chunks]
