"""Paragraph-aware chunking with overlap and a contextual header.

Each chunk carries a one-line header ("Apple Inc. 10-K FY2025, Item 7. Management's
Discussion and Analysis") that is embedded together with the text. Without it,
a paragraph like "Revenue grew 6%" would not say whose revenue or which year.

Financial tables (rendered by edgar.html_to_text as "cell | cell" rows) get chunks of
their own, led by their caption: the sentence that introduces them ("The following
table shows net sales by category (in millions):"). Packed into prose chunks, a table
was cut wherever the word budget ran out, so its numbers lost their row labels or
their column header, and a numeric question had no single chunk holding the answer.
A table too big for one chunk is split by rows, and every piece repeats the caption
and the header row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .sections import Section


@dataclass
class Chunk:
    section_item: str
    section_title: str
    ordinal: int
    header: str
    text: str

    is_table: bool = False

    @property
    def embed_text(self) -> str:
        return f"{self.header}\n{self.text}"


# A paragraph shorter than this is a caption or a stray line, not a chunk of its own.
MIN_CHUNK_WORDS = 15
# The longest caption kept in front of a table: the lead-in sentence, not the whole paragraph.
MAX_CAPTION_WORDS = 60
_NUMBER = re.compile(r"\d")


def _words(s: str) -> int:
    return len(s.split())


def _split_long(paragraph: str, max_words: int) -> list[str]:
    """Split an oversized paragraph on sentence boundaries (or hard-wrap as a last resort)."""
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    out, cur = [], []
    for s in sentences:
        if cur and _words(" ".join(cur + [s])) > max_words:
            out.append(" ".join(cur))
            cur = []
        if _words(s) > max_words:
            w = s.split()
            out.extend(" ".join(w[i : i + max_words]) for i in range(0, len(w), max_words))
            continue
        cur.append(s)
    if cur:
        out.append(" ".join(cur))
    return out


def _is_row(line: str) -> bool:
    return " | " in line


def _blocks(text: str) -> list[str | list[str]]:
    """Split section text into prose paragraphs (str) and financial tables (list of rows).

    A table is two or more consecutive "cell | cell" lines with numbers in at least half of
    them. Layout tables (bullet lists, signature blocks) have rows too but few numbers;
    they stay prose.
    """
    lines = [ln.strip() for ln in re.split(r"\n\s*\n|\n", text)]
    lines = [ln for ln in lines if ln]
    out: list[str | list[str]] = []
    i = 0
    while i < len(lines):
        j = i
        while j < len(lines) and _is_row(lines[j]):
            j += 1
        rows = lines[i:j]
        if len(rows) >= 2 and 2 * sum(bool(_NUMBER.search(r)) for r in rows) >= len(rows):
            out.append(rows)
            i = j
        else:
            out.append(lines[i])
            i += 1
    return out


def _caption(paragraph: str | None) -> str:
    """The lead-in of a table: the paragraph before it if short, else that paragraph's last sentence."""
    if not paragraph:
        return ""
    if _words(paragraph) <= MAX_CAPTION_WORDS:
        return paragraph
    last = re.split(r"(?<=[.!?:])\s+", paragraph.strip())[-1]
    return last if _words(last) <= MAX_CAPTION_WORDS else ""


def _table_pieces(rows: list[str], caption: str, max_words: int) -> list[str]:
    """The table as one text, or split by rows with the caption and header row repeated on every piece."""
    lead = [caption] if caption else []
    whole = "\n".join(lead + rows)
    if _words(whole) <= max_words:
        return [whole]
    head, body = rows[0], rows[1:]
    budget = max(max_words - _words("\n".join(lead + [head])), 1)
    pieces, cur = [], []
    for row in body:
        if cur and _words("\n".join(cur + [row])) > budget:
            pieces.append(cur)
            cur = []
        cur.append(row)
    if cur:
        pieces.append(cur)
    return ["\n".join(lead + [head] + p) for p in pieces]


def chunk_section(section: Section, doc_header: str, target_words: int = 320, overlap_words: int = 50) -> list[Chunk]:
    header = f"{doc_header}, {section.label}"
    chunks: list[Chunk] = []

    def emit(text: str, is_table: bool = False) -> None:
        chunks.append(Chunk(section.item, section.title, len(chunks), header, text, is_table))

    cur: list[str] = []
    fresh = 0  # words in cur that are not overlap carried from the previous chunk

    def flush() -> None:
        nonlocal cur, fresh
        emit("\n".join(cur))
        # carry the tail of the previous chunk forward as overlap
        tail = " ".join("\n".join(cur).split()[-overlap_words:]) if overlap_words else ""
        cur, fresh = ([tail] if tail else []), 0

    previous: str | None = None  # the prose paragraph right before the current block
    for block in _blocks(section.text):
        if isinstance(block, list):
            # Close the prose chunk so the text before and after the table stay in order; a
            # short lead-in alone is not worth a chunk, and also travels with the table as its caption.
            if fresh >= MIN_CHUNK_WORDS:
                flush()
            for piece in _table_pieces(block, _caption(previous), target_words):
                emit(piece, is_table=True)
            previous = None
            continue
        previous = block
        for p in _split_long(block, target_words) if _words(block) > target_words else [block]:
            if cur and _words("\n".join(cur + [p])) > target_words:
                flush()
            cur.append(p)
            fresh += _words(p)
    if cur and fresh and _words("\n".join(cur)) >= MIN_CHUNK_WORDS:
        emit("\n".join(cur))
    return chunks
