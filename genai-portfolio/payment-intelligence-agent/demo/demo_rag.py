"""Standalone RAG pipeline demo for PCI compliance Q&A.

Demonstrates document loading, chunking, embedding, vector indexing,
and retrieval-augmented question answering against PCI DSS v4.0
compliance documents.

Run: python -m demo.demo_rag
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.rag.document_loader import DocumentLoader
from src.rag.chunker import DocumentChunker
from src.rag.embeddings import EmbeddingGenerator
from src.rag.vector_store import VectorStore
from src.rag.retriever import RAGRetriever


def main() -> None:
    print("=" * 80)
    print("  PAYMENT INTELLIGENCE AGENT - RAG Pipeline Demo")
    print("  PCI DSS v4.0 Compliance Q&A")
    print("=" * 80)

    docs_path = Path(__file__).resolve().parents[1] / "src" / "data" / "sample_pci_docs"

    # Step 1: Document Loading
    print("\n[1/5] Loading PCI compliance documents...")
    loader = DocumentLoader(base_path=docs_path)
    documents = loader.load_directory()
    print(f"  Loaded {len(documents)} documents:")
    for doc in documents:
        print(f"    - {doc.metadata['filename']}: {doc.word_count:,} words, {len(doc.sections)} sections")

    # Step 2: Chunking
    print(f"\n{'─' * 80}")
    print("[2/5] Chunking documents...")
    chunker = DocumentChunker(max_chunk_size=800, chunk_overlap=150)
    all_chunks = chunker.chunk_documents(documents)
    print(f"  Generated {len(all_chunks)} chunks")
    print(f"  Average chunk size: {sum(len(c.text) for c in all_chunks) / len(all_chunks):.0f} characters")
    print(f"  Token estimate range: {min(c.token_estimate for c in all_chunks)} - {max(c.token_estimate for c in all_chunks)}")

    # Step 3: Embeddings
    print(f"\n{'─' * 80}")
    print("[3/5] Generating embeddings...")
    embedder = EmbeddingGenerator(use_local=True)
    texts = [chunk.text for chunk in all_chunks]
    embeddings = embedder.embed_batch(texts)
    print(f"  Embedding dimension: {embedder.dimension}")
    print(f"  Embedded {len(texts)} chunks")
    print(f"  Embedding matrix shape: {embeddings.shape}")

    # Step 4: Vector Store
    print(f"\n{'─' * 80}")
    print("[4/5] Building vector index...")
    store = VectorStore(dimension=embedder.dimension)
    store.add(
        embeddings=embeddings,
        chunk_ids=[c.chunk_id for c in all_chunks],
        texts=texts,
        metadata_list=[c.metadata for c in all_chunks],
    )
    print(f"  Indexed {store.size} vectors")

    # Step 5: End-to-end RAG queries
    print(f"\n{'─' * 80}")
    print("[5/5] Running compliance queries via RAG retriever...")

    retriever = RAGRetriever(chunk_size=800, chunk_overlap=150, top_k=4)
    retriever.initialize(str(docs_path))

    queries = [
        "What are the requirements for storing cardholder data?",
        "How should encryption keys be managed?",
        "What does PCI DSS say about access control and authentication?",
        "What are the requirements for penetration testing?",
        "How should audit logs be managed and monitored?",
        "What is the customized approach in PCI DSS v4.0?",
    ]

    for i, query in enumerate(queries, 1):
        print(f"\n{'─' * 80}")
        print(f"  Q{i}: {query}")
        print(f"{'─' * 80}")

        result = retriever.query(query)

        # Display answer
        print(f"\n  Answer:")
        for line in result["answer"].split("\n"):
            if line.strip():
                print(f"    {line.strip()[:100]}")

        # Display requirements
        if result["requirement_ids"]:
            req_str = ", ".join(f"Req {r}" for r in result["requirement_ids"])
            print(f"\n  Relevant Requirements: {req_str}")

        # Display sources
        if result["sources"]:
            print(f"\n  Sources:")
            for source in result["sources"][:3]:
                print(f"    - {source['document']} / {source['section']} (score: {source['score']:.3f})")

    print(f"\n{'=' * 80}")
    print("  Demo complete. RAG pipeline fully operational.")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
