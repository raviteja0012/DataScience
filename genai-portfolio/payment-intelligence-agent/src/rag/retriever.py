"""RAG retrieval pipeline for PCI compliance question answering.

Orchestrates the full retrieval-augmented generation pipeline: loads documents,
chunks them, generates embeddings, indexes in the vector store, and retrieves
relevant context to answer compliance questions.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .document_loader import DocumentLoader
from .chunker import DocumentChunker, Chunk
from .embeddings import EmbeddingGenerator
from .vector_store import VectorStore, SearchResult
from ..utils.logger import get_logger

logger = get_logger(__name__)

# PCI DSS requirement mapping for structured answers
_REQUIREMENT_KEYWORDS: dict[str, list[str]] = {
    "1": ["firewall", "network security", "network controls", "network segment"],
    "2": ["secure configuration", "default password", "system configuration", "hardening"],
    "3": ["stored data", "cardholder data", "data retention", "pan", "primary account", "encryption at rest", "data storage"],
    "4": ["transmission", "encrypt", "tls", "ssl", "cryptography", "data in transit"],
    "5": ["malware", "antivirus", "anti-malware", "malicious software", "anti-phishing"],
    "6": ["secure development", "software development", "vulnerability", "patch", "code review", "web application"],
    "7": ["access control", "need to know", "least privilege", "role-based", "restrict access"],
    "8": ["authentication", "password", "mfa", "multi-factor", "identity", "user id"],
    "9": ["physical access", "physical security", "media", "poi", "point of interaction"],
    "10": ["logging", "monitoring", "audit log", "audit trail", "log review", "time sync"],
    "11": ["testing", "penetration test", "vulnerability scan", "ids", "intrusion detection", "file integrity"],
    "12": ["security policy", "information security", "risk assessment", "incident response", "awareness", "third party"],
}


class RAGRetriever:
    """End-to-end RAG pipeline for PCI compliance question answering.

    Manages the lifecycle from document ingestion through retrieval:
    1. Load PCI compliance documents from markdown files
    2. Chunk documents with section-aware splitting and overlap
    3. Generate and index embeddings
    4. Retrieve relevant chunks for user queries
    5. Synthesize answers with source attribution

    Usage:
        retriever = RAGRetriever()
        retriever.initialize()
        result = retriever.query("What are the requirements for storing cardholder data?")
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        top_k: int = 5,
        score_threshold: float = 0.1,
    ) -> None:
        self.top_k = top_k
        self.score_threshold = score_threshold

        self._loader = DocumentLoader()
        self._chunker = DocumentChunker(
            max_chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self._embedder = EmbeddingGenerator(use_local=True)
        self._store = VectorStore(dimension=self._embedder.dimension)
        self._chunks: list[Chunk] = []
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Whether the pipeline has been initialized with documents."""
        return self._initialized

    def initialize(self, documents_path: str | None = None) -> None:
        """Load documents, chunk, embed, and index.

        Args:
            documents_path: Path to document directory.
                           Uses default sample_pci_docs if not specified.
        """
        # Load documents
        if documents_path:
            documents = self._loader.load_directory(documents_path)
        else:
            documents = self._loader.load_directory()

        if not documents:
            logger.warning("no_documents_loaded")
            self._initialized = True
            return

        # Chunk documents
        self._chunks = self._chunker.chunk_documents(documents)

        if not self._chunks:
            logger.warning("no_chunks_generated")
            self._initialized = True
            return

        # Generate embeddings
        texts = [chunk.text for chunk in self._chunks]
        embeddings = self._embedder.embed_batch(texts)

        # Index in vector store
        self._store.add(
            embeddings=embeddings,
            chunk_ids=[chunk.chunk_id for chunk in self._chunks],
            texts=texts,
            metadata_list=[chunk.metadata for chunk in self._chunks],
        )

        self._initialized = True
        logger.info(
            "rag_pipeline_initialized",
            documents=len(documents),
            chunks=len(self._chunks),
            vector_count=self._store.size,
        )

    def query(self, question: str) -> dict[str, Any]:
        """Answer a compliance question using RAG retrieval.

        Args:
            question: Natural language compliance question.

        Returns:
            Dictionary containing:
                - answer: Synthesized answer text.
                - sources: List of source references.
                - requirement_ids: Relevant PCI DSS requirement IDs.
                - retrieved_chunks: Raw retrieved chunk texts.
        """
        if not self._initialized:
            self.initialize()

        if self._store.size == 0:
            return {
                "answer": "The compliance knowledge base has not been loaded with documents.",
                "sources": [],
                "requirement_ids": [],
                "retrieved_chunks": [],
            }

        # Embed the query
        query_embedding = self._embedder.embed_text(question)

        # Retrieve relevant chunks
        results = self._store.search(
            query_embedding=query_embedding,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
        )

        if not results:
            return {
                "answer": "No relevant compliance information found for your question.",
                "sources": [],
                "requirement_ids": [],
                "retrieved_chunks": [],
            }

        # Identify relevant PCI requirements
        requirement_ids = self._identify_requirements(question, results)

        # Synthesize answer from retrieved chunks
        answer = self._synthesize_answer(question, results, requirement_ids)

        # Build source references
        sources = self._build_sources(results)

        return {
            "answer": answer,
            "sources": sources,
            "requirement_ids": requirement_ids,
            "retrieved_chunks": [r.text for r in results],
        }

    def _identify_requirements(
        self,
        question: str,
        results: list[SearchResult],
    ) -> list[str]:
        """Identify which PCI DSS requirements are relevant.

        Combines keyword matching on the question with requirement
        references found in the retrieved chunks.
        """
        question_lower = question.lower()
        combined_text = question_lower + " " + " ".join(r.text.lower() for r in results[:3])

        relevant_reqs: list[tuple[str, int]] = []
        for req_id, keywords in _REQUIREMENT_KEYWORDS.items():
            match_count = sum(1 for kw in keywords if kw in combined_text)
            if match_count > 0:
                relevant_reqs.append((req_id, match_count))

        # Also extract explicit requirement references (e.g., "Requirement 3.4")
        explicit = re.findall(r"requirement\s+(\d+(?:\.\d+)*)", combined_text, re.IGNORECASE)
        for req_ref in explicit:
            base_req = req_ref.split(".")[0]
            if base_req not in [r[0] for r in relevant_reqs]:
                relevant_reqs.append((base_req, 5))  # High weight for explicit mentions

        # Sort by relevance and return top matches
        relevant_reqs.sort(key=lambda x: x[1], reverse=True)
        return [req_id for req_id, _ in relevant_reqs[:5]]

    def _synthesize_answer(
        self,
        question: str,
        results: list[SearchResult],
        requirement_ids: list[str],
    ) -> str:
        """Synthesize a coherent answer from retrieved chunks.

        In production, this would call Cortex COMPLETE with the retrieved
        context. In demo mode, it constructs an answer from the chunk
        content with light restructuring.
        """
        # Combine top results into a coherent response
        answer_parts: list[str] = []

        # Lead with a direct response based on the question type
        question_lower = question.lower()

        if "what" in question_lower and "requirement" in question_lower:
            if requirement_ids:
                req_str = ", ".join(f"Requirement {r}" for r in requirement_ids)
                answer_parts.append(
                    f"Based on PCI DSS v4.0, the relevant requirements are {req_str}. "
                    "Here is a summary of the key points:\n"
                )
        elif "how" in question_lower:
            answer_parts.append(
                "According to PCI DSS v4.0, the recommended approach is as follows:\n"
            )
        else:
            answer_parts.append(
                "Based on the PCI DSS v4.0 compliance documentation:\n"
            )

        # Extract and deduplicate key points from retrieved chunks
        seen_content: set[str] = set()
        for result in results[:4]:
            text = result.text.strip()
            # Skip near-duplicate content
            text_key = text[:100].lower()
            if text_key in seen_content:
                continue
            seen_content.add(text_key)

            # Clean up chunk text for presentation
            clean_text = self._clean_for_presentation(text)
            if clean_text:
                answer_parts.append(f"\n{clean_text}")

        return "\n".join(answer_parts)

    def _build_sources(self, results: list[SearchResult]) -> list[dict[str, Any]]:
        """Build source reference list from search results."""
        sources: list[dict[str, Any]] = []
        seen_docs: set[str] = set()

        for result in results:
            doc_key = f"{result.metadata.get('filename', '')}::{result.metadata.get('section', '')}"
            if doc_key in seen_docs:
                continue
            seen_docs.add(doc_key)

            sources.append({
                "document": result.metadata.get("filename", "Unknown"),
                "section": result.metadata.get("section", ""),
                "score": result.score,
            })

        return sources

    @staticmethod
    def _clean_for_presentation(text: str) -> str:
        """Clean chunk text for inclusion in an answer.

        Removes overlap markers, excessive whitespace, and truncates
        overly long chunks.
        """
        # Remove overlap markers
        text = re.sub(r"^\[\.{3}\]\s*", "", text)

        # Remove markdown heading markers for inline presentation
        text = re.sub(r"^#{1,6}\s+", "**", text, count=1)
        if text.startswith("**") and not text.endswith("**"):
            # Find end of first line for heading bolding
            first_newline = text.find("\n")
            if first_newline > 0:
                text = text[:first_newline] + "**" + text[first_newline:]

        # Truncate very long chunks
        if len(text) > 500:
            # Find a sentence boundary near the limit
            truncated = text[:500]
            last_period = truncated.rfind(".")
            if last_period > 300:
                text = truncated[:last_period + 1]
            else:
                text = truncated + "..."

        return text.strip()
