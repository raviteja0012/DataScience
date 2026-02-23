"""Tests for the RAG pipeline components.

Validates document loading, chunking, embedding, vector search,
and end-to-end retrieval for PCI compliance question answering.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import numpy as np

from src.rag.document_loader import DocumentLoader, Document
from src.rag.chunker import DocumentChunker, Chunk
from src.rag.embeddings import EmbeddingGenerator
from src.rag.vector_store import VectorStore, SearchResult
from src.rag.retriever import RAGRetriever


_SAMPLE_DOCS_PATH = Path(__file__).resolve().parents[1] / "src" / "data" / "sample_pci_docs"


class TestDocumentLoader:
    """Test PCI document loading and parsing."""

    def test_load_sample_directory(self) -> None:
        loader = DocumentLoader(base_path=_SAMPLE_DOCS_PATH)
        documents = loader.load_directory()
        assert len(documents) >= 2
        for doc in documents:
            assert isinstance(doc, Document)
            assert doc.content
            assert doc.word_count > 0

    def test_document_has_sections(self) -> None:
        loader = DocumentLoader(base_path=_SAMPLE_DOCS_PATH)
        documents = loader.load_directory()
        # PCI DSS requirements doc should have many sections
        pci_doc = next(
            (d for d in documents if "requirements" in d.metadata.get("filename", "").lower()),
            None,
        )
        assert pci_doc is not None
        assert len(pci_doc.sections) > 5

    def test_document_metadata(self) -> None:
        loader = DocumentLoader(base_path=_SAMPLE_DOCS_PATH)
        documents = loader.load_directory()
        for doc in documents:
            assert "source" in doc.metadata
            assert "filename" in doc.metadata
            assert "title" in doc.metadata

    def test_content_hash_uniqueness(self) -> None:
        loader = DocumentLoader(base_path=_SAMPLE_DOCS_PATH)
        documents = loader.load_directory()
        hashes = [doc.content_hash for doc in documents]
        assert len(hashes) == len(set(hashes))

    def test_nonexistent_directory(self) -> None:
        loader = DocumentLoader(base_path="/nonexistent/path")
        documents = loader.load_directory()
        assert documents == []


class TestDocumentChunker:
    """Test document chunking strategies."""

    def test_basic_chunking(self) -> None:
        doc = Document(
            content="Section 1\n\nThis is some content.\n\nSection 2\n\nMore content here.",
            metadata={"filename": "test.md"},
            sections=[
                {"heading": "Section 1", "level": 1, "content": "This is some content."},
                {"heading": "Section 2", "level": 1, "content": "More content here."},
            ],
        )
        chunker = DocumentChunker(max_chunk_size=500, chunk_overlap=50)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.text
            assert chunk.chunk_id
            assert chunk.metadata

    def test_large_section_splitting(self) -> None:
        long_content = " ".join(["This is a sentence about PCI compliance."] * 100)
        doc = Document(
            content=long_content,
            metadata={"filename": "test.md"},
            sections=[{"heading": "Long Section", "level": 1, "content": long_content}],
        )
        chunker = DocumentChunker(max_chunk_size=200, chunk_overlap=50)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1
        for chunk in chunks:
            # Account for overlap making chunks slightly larger
            assert len(chunk.text) < 600

    def test_overlap_applied(self) -> None:
        content_a = "First section with important PCI information about firewalls."
        content_b = "Second section about encryption and key management."
        doc = Document(
            content=f"{content_a}\n\n{content_b}",
            metadata={"filename": "test.md", "source": "test.md"},
            sections=[
                {"heading": "Section A", "level": 1, "content": content_a},
                {"heading": "Section B", "level": 1, "content": content_b},
            ],
        )
        chunker = DocumentChunker(max_chunk_size=500, chunk_overlap=20)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 2

    def test_token_estimation(self) -> None:
        doc = Document(
            content="Hello world this is a test document",
            metadata={"filename": "test.md"},
            sections=[{"heading": "Test", "level": 1, "content": "Hello world this is a test document"}],
        )
        chunker = DocumentChunker()
        chunks = chunker.chunk_document(doc)
        for chunk in chunks:
            assert chunk.token_estimate > 0

    def test_empty_document(self) -> None:
        doc = Document(content="", metadata={"filename": "empty.md"}, sections=[])
        chunker = DocumentChunker()
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 0


class TestEmbeddingGenerator:
    """Test embedding generation."""

    def test_single_text_embedding(self) -> None:
        generator = EmbeddingGenerator(use_local=True)
        embedding = generator.embed_text("Test sentence for embedding")
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == generator.dimension

    def test_batch_embedding(self) -> None:
        generator = EmbeddingGenerator(use_local=True)
        texts = ["First sentence", "Second sentence", "Third sentence"]
        embeddings = generator.embed_batch(texts)
        assert embeddings.shape == (3, generator.dimension)

    def test_embedding_normalization(self) -> None:
        generator = EmbeddingGenerator(use_local=True)
        embedding = generator.embed_text("Normalized embedding test")
        norm = np.linalg.norm(embedding)
        # Should be approximately unit length
        assert abs(norm - 1.0) < 0.1

    def test_caching(self) -> None:
        generator = EmbeddingGenerator(use_local=True)
        text = "Cache test sentence"
        e1 = generator.embed_text(text)
        e2 = generator.embed_text(text)
        np.testing.assert_array_equal(e1, e2)

    def test_empty_batch(self) -> None:
        generator = EmbeddingGenerator(use_local=True)
        embeddings = generator.embed_batch([])
        assert embeddings.shape == (0, generator.dimension)


class TestVectorStore:
    """Test vector similarity search."""

    def test_add_and_search(self) -> None:
        store = VectorStore(dimension=4)
        embeddings = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.float32)
        store.add(
            embeddings=embeddings,
            chunk_ids=["c1", "c2", "c3"],
            texts=["text1", "text2", "text3"],
            metadata_list=[{}, {}, {}],
        )
        assert store.size == 3

        query = np.array([1, 0, 0, 0], dtype=np.float32)
        results = store.search(query, top_k=2)
        assert len(results) == 2
        assert results[0].chunk_id == "c1"

    def test_search_returns_ordered_results(self) -> None:
        store = VectorStore(dimension=3)
        embeddings = np.array([
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        store.add(
            embeddings=embeddings,
            chunk_ids=["a", "b", "c"],
            texts=["ta", "tb", "tc"],
            metadata_list=[{}, {}, {}],
        )

        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        results = store.search(query, top_k=3)
        # First result should be closest to query
        assert results[0].score >= results[1].score

    def test_empty_store_search(self) -> None:
        store = VectorStore(dimension=4)
        query = np.array([1, 0, 0, 0], dtype=np.float32)
        results = store.search(query, top_k=5)
        assert results == []

    def test_clear(self) -> None:
        store = VectorStore(dimension=4)
        embeddings = np.array([[1, 0, 0, 0]], dtype=np.float32)
        store.add(embeddings, ["c1"], ["text1"], [{}])
        assert store.size == 1
        store.clear()
        assert store.size == 0


class TestRAGRetriever:
    """Test the end-to-end RAG retrieval pipeline."""

    @pytest.fixture
    def initialized_retriever(self) -> RAGRetriever:
        retriever = RAGRetriever(chunk_size=500, chunk_overlap=100, top_k=3)
        retriever.initialize(str(_SAMPLE_DOCS_PATH))
        return retriever

    def test_initialization(self, initialized_retriever: RAGRetriever) -> None:
        assert initialized_retriever.is_initialized
        assert initialized_retriever._store.size > 0

    def test_cardholder_data_query(self, initialized_retriever: RAGRetriever) -> None:
        result = initialized_retriever.query(
            "What are the requirements for storing cardholder data?"
        )
        assert result["answer"]
        assert len(result["answer"]) > 50
        assert result["sources"]
        assert "3" in result["requirement_ids"]

    def test_encryption_key_query(self, initialized_retriever: RAGRetriever) -> None:
        result = initialized_retriever.query(
            "How should encryption keys be managed?"
        )
        assert result["answer"]
        assert len(result["answer"]) > 50
        # Should reference key management or related security concepts
        answer_lower = result["answer"].lower()
        assert any(word in answer_lower for word in [
            "key", "encrypt", "cryptograph", "pci", "security", "protect",
        ])

    def test_access_control_query(self, initialized_retriever: RAGRetriever) -> None:
        result = initialized_retriever.query(
            "What does PCI DSS say about access control?"
        )
        assert result["answer"]
        assert any(r in result["requirement_ids"] for r in ["7", "8"])

    def test_sources_included(self, initialized_retriever: RAGRetriever) -> None:
        result = initialized_retriever.query("What is PCI DSS?")
        assert result["sources"]
        for source in result["sources"]:
            assert "document" in source
            assert "score" in source

    def test_no_results_query(self, initialized_retriever: RAGRetriever) -> None:
        result = initialized_retriever.query(
            "What is the weather forecast for tomorrow?"
        )
        assert result["answer"]  # Should still return something

    def test_uninitialized_query(self) -> None:
        retriever = RAGRetriever()
        # Should auto-initialize
        result = retriever.query("What is PCI DSS?")
        assert "answer" in result
