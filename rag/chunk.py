from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_CODE_FENCE_RE = re.compile(r"^\s*```")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Chunk size is tied to the embedding model's token limit, not a fixed
# constant: 350 leaves comfortable headroom under bge-small-en-v1.5's
# 512-token max_seq_length (see rag/embeddings.py). Revisit this value if the
# embedding model changes.
DEFAULT_TARGET_TOKENS = 350
DEFAULT_OVERLAP_TOKENS = 50


@dataclass
class Chunk:
    text: str
    heading_path: str


@dataclass
class _Block:
    kind: str  # "para" | "table" | "code"
    text: str


def _parse_blocks(markdown: str) -> list[tuple[str, object]]:
    """Parse Markdown into an ordered list of ("heading", (level, text)) and
    ("block", _Block) items. Tables and fenced code blocks are kept as single
    atomic blocks so a later packing pass never splits them."""
    lines = markdown.splitlines()
    items: list[tuple[str, object]] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        if not line.strip():
            i += 1
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            text = heading_match.group(2).strip()
            items.append(("heading", (level, text)))
            i += 1
            continue

        if _CODE_FENCE_RE.match(line):
            fence_lines = [line]
            i += 1
            while i < n and not _CODE_FENCE_RE.match(lines[i]):
                fence_lines.append(lines[i])
                i += 1
            if i < n:
                fence_lines.append(lines[i])
                i += 1
            items.append(("block", _Block("code", "\n".join(fence_lines))))
            continue

        if _TABLE_ROW_RE.match(line):
            table_lines = [line]
            i += 1
            while i < n and _TABLE_ROW_RE.match(lines[i]):
                table_lines.append(lines[i])
                i += 1
            items.append(("block", _Block("table", "\n".join(table_lines))))
            continue

        para_lines = [line]
        i += 1
        while (
            i < n
            and lines[i].strip()
            and not _HEADING_RE.match(lines[i])
            and not _TABLE_ROW_RE.match(lines[i])
            and not _CODE_FENCE_RE.match(lines[i])
        ):
            para_lines.append(lines[i])
            i += 1
        items.append(("block", _Block("para", "\n".join(para_lines))))

    return items


def _split_oversized_para(
    text: str, count_tokens: Callable[[str], int], target_tokens: int
) -> list[str]:
    """Sentence-level fallback for a single paragraph that alone exceeds the
    token budget. A single sentence longer than the budget is kept whole --
    no word-level hard-splitting, since that's rare enough not to be worth
    the extra complexity in a basic pipeline."""
    sentences = _SENTENCE_SPLIT_RE.split(text)
    pieces = []
    current = ""
    for sentence in sentences:
        candidate = f"{current} {sentence}".strip() if current else sentence
        if current and count_tokens(candidate) > target_tokens:
            pieces.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def chunk_markdown(
    markdown: str,
    count_tokens: Callable[[str], int],
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
) -> list[Chunk]:
    """Split Markdown into chunks on structural boundaries: a new chunk starts
    at every heading change, tables/code blocks are always their own chunk
    (never split, even if oversized), and consecutive paragraphs are packed
    together up to target_tokens with overlap_tokens carried into the next
    chunk when a size-triggered (not heading-triggered) split happens."""
    items = _parse_blocks(markdown)

    heading_stack: list[tuple[int, str]] = []
    chunks: list[Chunk] = []
    current_texts: list[str] = []
    current_tokens = 0

    def heading_path() -> str:
        return " > ".join(text for _, text in heading_stack)

    def flush(carry_overlap: bool) -> None:
        nonlocal current_texts, current_tokens
        if not current_texts:
            return
        chunks.append(Chunk(text="\n\n".join(current_texts), heading_path=heading_path()))
        if carry_overlap:
            tail = current_texts[-1]
            tail_tokens = count_tokens(tail)
            current_texts = [tail] if tail_tokens <= overlap_tokens else []
            current_tokens = tail_tokens if current_texts else 0
        else:
            current_texts = []
            current_tokens = 0

    def add_piece(piece: str) -> None:
        nonlocal current_tokens
        piece_tokens = count_tokens(piece)
        if current_texts and current_tokens + piece_tokens > target_tokens:
            flush(carry_overlap=True)
        current_texts.append(piece)
        current_tokens += piece_tokens

    for kind, payload in items:
        if kind == "heading":
            flush(carry_overlap=False)
            level, text = payload
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, text))
            continue

        block: _Block = payload
        if block.kind in ("table", "code"):
            flush(carry_overlap=False)
            chunks.append(Chunk(text=block.text, heading_path=heading_path()))
            continue

        if count_tokens(block.text) > target_tokens:
            for piece in _split_oversized_para(block.text, count_tokens, target_tokens):
                add_piece(piece)
        else:
            add_piece(block.text)

    flush(carry_overlap=False)
    return [c for c in chunks if c.text.strip()]
