"""PCI compliance document ingestion and preprocessing.

Loads, normalizes, and structures PCI-DSS documents from markdown files,
preparing them for chunking and embedding. Handles section detection,
metadata extraction, and document-level deduplication.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class Document:
    """A loaded document with metadata.

    Attributes:
        content: Full text content of the document.
        metadata: Document metadata (source, title, hash, etc.).
        sections: Parsed sections with hierarchical headers.
    """

    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    sections: list[dict[str, Any]] = field(default_factory=list)

    @property
    def content_hash(self) -> str:
        """SHA-256 hash of the document content for deduplication."""
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    @property
    def word_count(self) -> int:
        """Approximate word count."""
        return len(self.content.split())


class DocumentLoader:
    """Loads and preprocesses PCI compliance documents.

    Supports markdown files with hierarchical heading structure.
    Extracts sections, cleans formatting artifacts, and produces
    structured Document objects ready for chunking.
    """

    SUPPORTED_EXTENSIONS = {".md", ".txt", ".markdown"}

    def __init__(self, base_path: str | Path | None = None) -> None:
        if base_path is None:
            base_path = Path(__file__).resolve().parents[1] / "data" / "sample_pci_docs"
        self.base_path = Path(base_path)

    def load_directory(self, path: str | Path | None = None) -> list[Document]:
        """Load all supported documents from a directory.

        Args:
            path: Directory path. Uses base_path if not specified.

        Returns:
            List of loaded Document objects.
        """
        directory = Path(path) if path else self.base_path
        if not directory.exists():
            logger.warning("document_directory_not_found", path=str(directory))
            return []

        documents: list[Document] = []
        seen_hashes: set[str] = set()

        for file_path in sorted(directory.iterdir()):
            if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue

            try:
                doc = self.load_file(file_path)
                if doc.content_hash not in seen_hashes:
                    documents.append(doc)
                    seen_hashes.add(doc.content_hash)
                    logger.info(
                        "document_loaded",
                        file=file_path.name,
                        word_count=doc.word_count,
                        sections=len(doc.sections),
                    )
                else:
                    logger.info("duplicate_document_skipped", file=file_path.name)
            except Exception as exc:
                logger.error("document_load_failed", file=file_path.name, error=str(exc))

        logger.info("directory_loaded", document_count=len(documents), path=str(directory))
        return documents

    def load_file(self, file_path: str | Path) -> Document:
        """Load a single document file.

        Args:
            file_path: Path to the document file.

        Returns:
            Parsed Document object.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file type is not supported.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        if file_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

        content = file_path.read_text(encoding="utf-8")
        cleaned = self._clean_content(content)
        sections = self._extract_sections(cleaned)

        return Document(
            content=cleaned,
            metadata={
                "source": str(file_path),
                "filename": file_path.name,
                "title": self._extract_title(cleaned, file_path.stem),
            },
            sections=sections,
        )

    def _clean_content(self, content: str) -> str:
        """Clean and normalize document content.

        Removes excessive whitespace, normalizes line endings,
        and strips common formatting artifacts.
        """
        # Normalize line endings
        text = content.replace("\r\n", "\n").replace("\r", "\n")

        # Remove excessive blank lines (keep max 2)
        text = re.sub(r"\n{4,}", "\n\n\n", text)

        # Strip trailing whitespace from lines
        text = "\n".join(line.rstrip() for line in text.split("\n"))

        return text.strip()

    def _extract_title(self, content: str, fallback: str) -> str:
        """Extract the document title from the first heading or use fallback."""
        match = re.match(r"^#\s+(.+)$", content, re.MULTILINE)
        if match:
            return match.group(1).strip()
        return fallback.replace("_", " ").replace("-", " ").title()

    def _extract_sections(self, content: str) -> list[dict[str, Any]]:
        """Parse markdown content into hierarchical sections.

        Returns:
            List of section dictionaries with heading, level, and content.
        """
        sections: list[dict[str, Any]] = []
        current_section: dict[str, Any] | None = None
        content_lines: list[str] = []

        for line in content.split("\n"):
            heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)

            if heading_match:
                # Save previous section
                if current_section is not None:
                    current_section["content"] = "\n".join(content_lines).strip()
                    if current_section["content"]:
                        sections.append(current_section)
                    content_lines = []

                level = len(heading_match.group(1))
                heading = heading_match.group(2).strip()
                current_section = {
                    "heading": heading,
                    "level": level,
                    "content": "",
                }
            else:
                content_lines.append(line)

        # Save final section
        if current_section is not None:
            current_section["content"] = "\n".join(content_lines).strip()
            if current_section["content"]:
                sections.append(current_section)

        # If no sections found, treat entire content as one section
        if not sections and content.strip():
            sections.append({
                "heading": "Document Content",
                "level": 1,
                "content": content.strip(),
            })

        return sections
