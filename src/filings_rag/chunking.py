"""Paragraph-aware chunking with overlap and a contextual header.

Each chunk carries a one-line header ("Apple Inc. 10-K FY2025, Item 7. Management's
Discussion and Analysis") that is embedded together with the text. Without it,
a paragraph like "Revenue grew 6%" would not say whose revenue or which year.
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

    @property
    def embed_text(self) -> str:
        return f"{self.header}\n{self.text}"


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


def chunk_section(section: Section, doc_header: str, target_words: int = 320, overlap_words: int = 50) -> list[Chunk]:
    paragraphs: list[str] = []
    for p in re.split(r"\n\s*\n|\n", section.text):
        p = p.strip()
        if not p:
            continue
        paragraphs.extend(_split_long(p, target_words) if _words(p) > target_words else [p])

    header = f"{doc_header}, {section.label}"
    chunks: list[Chunk] = []
    cur: list[str] = []
    for p in paragraphs:
        if cur and _words("\n".join(cur + [p])) > target_words:
            chunks.append(Chunk(section.item, section.title, len(chunks), header, "\n".join(cur)))
            # carry the tail of the previous chunk forward as overlap
            tail = " ".join("\n".join(cur).split()[-overlap_words:]) if overlap_words else ""
            cur = [tail] if tail else []
        cur.append(p)
    if cur and _words("\n".join(cur)) >= 15:
        chunks.append(Chunk(section.item, section.title, len(chunks), header, "\n".join(cur)))
    return chunks
