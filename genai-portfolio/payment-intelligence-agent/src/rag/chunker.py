"""Smart document chunking with overlap for RAG retrieval.

Implements section-aware chunking that respects document structure,
maintains semantic coherence within chunks, and provides configurable
overlap to prevent information loss at chunk boundaries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .document_loader import Document
from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class Chunk:
    """A document chunk ready for embedding.

    Attributes:
        text: The chunk text content.
        metadata: Source document metadata plus chunk-specific info.
        chunk_id: Unique identifier for this chunk.
        token_estimate: Approximate token count.
    """

    text: str
    metadata: dict[str, Any]
    chunk_id: str
    token_estimate: int


class DocumentChunker:
    """Section-aware document chunker with configurable overlap.

    Chunking strategy:
    1. First split on section boundaries (markdown headings)
    2. If a section exceeds max_chunk_size, split on paragraph boundaries
    3. If a paragraph exceeds max_chunk_size, split on sentence boundaries
    4. Apply overlap by prepending trailing content from the previous chunk

    This approach preserves semantic coherence within chunks while ensuring
    no single chunk exceeds the embedding model's context window.

    Attributes:
        max_chunk_size: Maximum characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.
        min_chunk_size: Minimum characters for a chunk (smaller chunks are merged).
    """

    def __init__(
        self,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 200,
        min_chunk_size: int = 100,
    ) -> None:
        self.max_chunk_size = max_chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a document into overlapping chunks.

        Args:
            document: Loaded Document to chunk.

        Returns:
            List of Chunk objects with metadata.
        """
        if not document.sections:
            # Fallback: chunk raw content if non-empty
            if not document.content.strip():
                return []
            return self._chunk_text(
                document.content,
                base_metadata=document.metadata,
                section_heading="Full Document",
            )

        all_chunks: list[Chunk] = []

        for section in document.sections:
            section_text = section["content"]
            if not section_text.strip():
                continue

            # Prepend heading for context
            heading = section["heading"]
            prefixed_text = f"{heading}\n\n{section_text}"

            section_chunks = self._chunk_text(
                prefixed_text,
                base_metadata=document.metadata,
                section_heading=heading,
            )
            all_chunks.extend(section_chunks)

        # Apply overlap between chunks
        all_chunks = self._apply_overlap(all_chunks)

        logger.info(
            "document_chunked",
            source=document.metadata.get("filename", "unknown"),
            chunk_count=len(all_chunks),
        )

        return all_chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Chunk multiple documents.

        Args:
            documents: List of Document objects.

        Returns:
            Combined list of chunks from all documents.
        """
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))

        logger.info("batch_chunking_complete", total_chunks=len(all_chunks))
        return all_chunks

    def _chunk_text(
        self,
        text: str,
        base_metadata: dict[str, Any],
        section_heading: str,
    ) -> list[Chunk]:
        """Split text into chunks respecting paragraph and sentence boundaries."""
        if len(text) <= self.max_chunk_size:
            return [self._create_chunk(text, base_metadata, section_heading, 0)]

        # Split on paragraph boundaries first
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: list[Chunk] = []
        current_text = ""
        chunk_idx = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If adding this paragraph exceeds limit, finalize current chunk
            if current_text and len(current_text) + len(para) + 2 > self.max_chunk_size:
                if len(current_text) >= self.min_chunk_size:
                    chunks.append(self._create_chunk(current_text, base_metadata, section_heading, chunk_idx))
                    chunk_idx += 1
                    current_text = ""

            # If single paragraph exceeds limit, split on sentences
            if len(para) > self.max_chunk_size:
                if current_text:
                    chunks.append(self._create_chunk(current_text, base_metadata, section_heading, chunk_idx))
                    chunk_idx += 1
                    current_text = ""

                sentence_chunks = self._split_on_sentences(para)
                for sent_chunk in sentence_chunks:
                    chunks.append(self._create_chunk(sent_chunk, base_metadata, section_heading, chunk_idx))
                    chunk_idx += 1
            else:
                current_text = f"{current_text}\n\n{para}".strip() if current_text else para

        # Don't drop the last accumulated text
        if current_text and len(current_text) >= self.min_chunk_size:
            chunks.append(self._create_chunk(current_text, base_metadata, section_heading, chunk_idx))
        elif current_text and chunks:
            # Merge short trailing text into the last chunk
            last = chunks[-1]
            merged_text = f"{last.text}\n\n{current_text}"
            chunks[-1] = self._create_chunk(merged_text, base_metadata, section_heading, len(chunks) - 1)

        return chunks

    def _split_on_sentences(self, text: str) -> list[str]:
        """Split text into sentence-bounded chunks."""
        # Split on sentence-ending punctuation followed by space
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current = ""

        for sentence in sentences:
            if current and len(current) + len(sentence) + 1 > self.max_chunk_size:
                chunks.append(current.strip())
                current = ""

            current = f"{current} {sentence}".strip() if current else sentence

        if current:
            chunks.append(current.strip())

        return chunks

    def _apply_overlap(self, chunks: list[Chunk]) -> list[Chunk]:
        """Apply text overlap between consecutive chunks from the same document."""
        if len(chunks) <= 1 or self.chunk_overlap <= 0:
            return chunks

        overlapped: list[Chunk] = [chunks[0]]

        for i in range(1, len(chunks)):
            prev_text = chunks[i - 1].text
            curr_chunk = chunks[i]

            # Only apply overlap within the same source document
            if (curr_chunk.metadata.get("source") != chunks[i - 1].metadata.get("source")):
                overlapped.append(curr_chunk)
                continue

            # Extract overlap from end of previous chunk
            overlap_text = prev_text[-self.chunk_overlap:] if len(prev_text) > self.chunk_overlap else prev_text

            # Find a clean break point (sentence or paragraph boundary)
            clean_break = overlap_text.find(". ")
            if clean_break > 0:
                overlap_text = overlap_text[clean_break + 2:]

            if overlap_text:
                new_text = f"[...] {overlap_text}\n\n{curr_chunk.text}"
            else:
                new_text = curr_chunk.text

            overlapped.append(Chunk(
                text=new_text,
                metadata=curr_chunk.metadata,
                chunk_id=curr_chunk.chunk_id,
                token_estimate=self._estimate_tokens(new_text),
            ))

        return overlapped

    def _create_chunk(
        self,
        text: str,
        base_metadata: dict[str, Any],
        section_heading: str,
        index: int,
    ) -> Chunk:
        """Create a Chunk with metadata."""
        source = base_metadata.get("filename", "unknown")
        chunk_id = f"{source}::{section_heading}::{index}"

        metadata = {
            **base_metadata,
            "section": section_heading,
            "chunk_index": index,
        }

        return Chunk(
            text=text.strip(),
            metadata=metadata,
            chunk_id=chunk_id,
            token_estimate=self._estimate_tokens(text),
        )

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count using the ~4 characters per token heuristic."""
        return len(text) // 4
