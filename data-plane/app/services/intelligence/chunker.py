"""Text chunking strategies for the ingest pipeline.

Supports multiple chunking strategies:
- recursive: Recursive character text splitter with atomic pattern protection (default for contextual retrieval)
- fixed: Split by character count with overlap
- sentence: Split on sentence boundaries
- late_chunking: Semantic paragraph-aware splitting
"""

import re
import uuid

from app.services.intelligence.models import ChunkResult
from app.utils.logger import get_logger

log = get_logger(__name__)

# Separators ordered from most to least semantic
RECURSIVE_SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]

# Patterns that must never be split across chunks.
# Each tuple: (compiled regex, human-readable name)
ATOMIC_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"```[\s\S]*?```"),                         "codeblock"),
    (re.compile(r"https?://[^\s)\]>\"']+"),                 "url"),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "email"),
    (re.compile(r"\+?\d[\d\s.\-()]{7,}\d"),                 "phone"),
    (re.compile(r"\b[A-Za-z0-9_\-]{32,}\b"),                "api_key"),
]


def _protect_atomic(text: str) -> tuple[str, dict[str, str]]:
    """Replace atomic patterns with unique placeholders.

    Returns the modified text and a mapping of placeholder -> original value.
    """
    placeholders: dict[str, str] = {}
    for pattern, _name in ATOMIC_PATTERNS:
        for match in pattern.finditer(text):
            original = match.group()
            # Skip if already inside a placeholder
            if original.startswith("__ATOMIC_"):
                continue
            ph = f"__ATOMIC_{uuid.uuid4().hex[:12]}__"
            placeholders[ph] = original
            text = text.replace(original, ph, 1)
    return text, placeholders


def _restore_atomic(chunks: list[str], placeholders: dict[str, str]) -> list[str]:
    """Restore atomic placeholders back to their original values."""
    if not placeholders:
        return chunks
    restored = []
    for chunk in chunks:
        for ph, original in placeholders.items():
            chunk = chunk.replace(ph, original)
        restored.append(chunk)
    return restored


class Chunker:
    """Split text into chunks for embedding."""

    def chunk(
        self,
        text: str,
        strategy: str = "late_chunking",
        max_chunk_size: int = 1200,
        overlap: int = 150,
    ) -> ChunkResult:
        if not text or not text.strip():
            return ChunkResult(chunks=[], total_chunks=0, strategy=strategy, avg_chunk_size=0)

        if strategy == "fixed":
            chunks = self._fixed_chunks(text, max_chunk_size, overlap)
        elif strategy == "sentence":
            chunks = self._sentence_chunks(text, max_chunk_size, overlap)
        elif strategy == "recursive":
            chunks = self._recursive_chunks(text, max_chunk_size, overlap)
        else:
            chunks = self._late_chunking(text, max_chunk_size, overlap)

        # Filter out empty chunks
        chunks = [c.strip() for c in chunks if c.strip()]

        avg_size = sum(len(c) for c in chunks) // max(len(chunks), 1)

        log.info(
            "chunking_complete",
            strategy=strategy,
            total_chunks=len(chunks),
            avg_chunk_size=avg_size,
        )

        return ChunkResult(
            chunks=chunks,
            total_chunks=len(chunks),
            strategy=strategy,
            avg_chunk_size=avg_size,
        )

    def _fixed_chunks(self, text: str, max_size: int, overlap: int) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
            if start >= len(text):
                break
        return chunks

    def _sentence_chunks(self, text: str, max_size: int, overlap: int) -> list[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks = []
        current = ""

        for sentence in sentences:
            if len(current) + len(sentence) + 1 > max_size and current:
                chunks.append(current)
                # Overlap: keep last portion
                if overlap > 0 and len(current) > overlap:
                    current = current[-overlap:] + " " + sentence
                else:
                    current = sentence
            else:
                current = current + " " + sentence if current else sentence

        if current.strip():
            chunks.append(current)

        return chunks

    def _recursive_chunks(self, text: str, max_size: int, overlap: int) -> list[str]:
        """Recursive character text splitter with atomic pattern protection.

        Protects URLs, emails, phone numbers, code blocks, and API keys from
        being split across chunk boundaries. Tries to split on the most
        semantic separator first and recursively falls back to finer ones.
        """
        # 1. Protect atomic patterns with placeholders
        protected_text, placeholders = _protect_atomic(text)

        # 2. Recursive split on placeholders-safe text
        chunks = self._recursive_split(protected_text, RECURSIVE_SEPARATORS, max_size, overlap)

        # 3. Restore original values
        return _restore_atomic(chunks, placeholders)

    def _recursive_split(
        self, text: str, separators: list[str], max_size: int, overlap: int,
    ) -> list[str]:
        if len(text) <= max_size:
            return [text]

        # Find the best separator that actually exists in the text
        separator = ""
        remaining_separators = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = sep
                remaining_separators = []
                break
            if sep in text:
                separator = sep
                remaining_separators = separators[i + 1:]
                break

        # Split on the chosen separator
        if separator:
            pieces = text.split(separator)
        else:
            # Last resort: character-level split
            pieces = list(text)

        # Merge pieces into chunks respecting max_size
        chunks = []
        current = ""
        for piece in pieces:
            candidate = current + separator + piece if current else piece
            if len(candidate) > max_size and current:
                chunks.append(current)
                # Overlap: carry tail of previous chunk
                if overlap > 0 and len(current) > overlap:
                    current = current[-overlap:] + separator + piece
                else:
                    current = piece
            else:
                current = candidate

        if current.strip():
            chunks.append(current)

        # Recursively split any chunks that are still too large
        final = []
        for chunk in chunks:
            if len(chunk) > max_size and remaining_separators:
                final.extend(self._recursive_split(chunk, remaining_separators, max_size, overlap))
            else:
                final.append(chunk)

        return final

    def _late_chunking(self, text: str, max_size: int, overlap: int) -> list[str]:
        """Paragraph-aware chunking that respects document structure.

        Splits on double newlines (paragraphs) first, then merges small
        paragraphs and splits large ones to stay within max_size.
        """
        paragraphs = re.split(r"\n\s*\n", text)
        chunks = []
        current = ""

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If a single paragraph exceeds max_size, split it by sentences
            if len(para) > max_size:
                if current.strip():
                    chunks.append(current)
                    current = ""
                sub_chunks = self._sentence_chunks(para, max_size, overlap)
                chunks.extend(sub_chunks)
                continue

            if len(current) + len(para) + 2 > max_size and current:
                chunks.append(current)
                # Overlap: keep tail of previous chunk
                if overlap > 0 and len(current) > overlap:
                    current = current[-overlap:] + "\n\n" + para
                else:
                    current = para
            else:
                current = current + "\n\n" + para if current else para

        if current.strip():
            chunks.append(current)

        return chunks
