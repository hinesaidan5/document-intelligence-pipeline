"""
Vector Store — document-intelligence-pipeline
Handles embedding generation and storage/retrieval using ChromaDB.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

from backend.ingestion.pdf_processor import ProcessedDocument, DocumentChunk


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SearchResult:
    """A single result returned from a similarity search."""
    chunk_id: str
    doc_id: str
    text: str
    page_number: int
    filename: str
    similarity_score: float     # 0.0 (least similar) to 1.0 (most similar)
    chunk_index: int


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Wraps ChromaDB to provide document storage and semantic search.

    Uses ChromaDB's built-in sentence-transformers embedding function
    so no API key is needed at this stage — embeddings run locally.

    Persistence:
        Data is saved to `persist_dir` on disk. Re-instantiating with
        the same directory reloads all previously stored documents.
    """

    COLLECTION_NAME = "document_intelligence"

    def __init__(self, persist_dir: str | Path = "./chroma_db"):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        # Local embedding model — runs on CPU, no API key needed
        self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )

        # Persistent ChromaDB client
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))

        # Get or create the collection
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},   # Cosine similarity
        )

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def add_document(self, doc: ProcessedDocument) -> int:
        """
        Embed and store all chunks from a ProcessedDocument.
        Returns the number of chunks added.

        If chunks from this document already exist they are replaced,
        making this operation idempotent.
        """
        if not doc.chunks:
            return 0

        ids, documents, metadatas = [], [], []

        for chunk in doc.chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append({
                "doc_id": chunk.doc_id,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "word_count": chunk.word_count,
                "filename": chunk.metadata.get("filename", doc.filename),
            })

        # Upsert — safe to call multiple times on the same document
        self._collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        return len(ids)

    def delete_document(self, doc_id: str) -> int:
        """Remove all chunks belonging to a document. Returns chunks deleted."""
        results = self._collection.get(where={"doc_id": doc_id})
        ids_to_delete = results["ids"]
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
        doc_id: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Semantic similarity search across stored chunks.

        Args:
            query:  Natural language question or search string.
            top_k:  Number of results to return.
            doc_id: Optional — restrict search to a single document.

        Returns:
            List of SearchResult sorted by relevance (most relevant first).
        """
        where_filter = {"doc_id": doc_id} if doc_id else None

        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count() or 1),
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        search_results = []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for text, meta, distance in zip(docs, metas, distances):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score 0–1
            similarity = 1 - (distance / 2)

            search_results.append(SearchResult(
                chunk_id=meta.get("chunk_id", ""),
                doc_id=meta["doc_id"],
                text=text,
                page_number=meta["page_number"],
                filename=meta["filename"],
                similarity_score=round(similarity, 4),
                chunk_index=meta["chunk_index"],
            ))

        return search_results

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_documents(self) -> list[dict]:
        """Return a summary of all documents currently stored."""
        results = self._collection.get(include=["metadatas"])
        if not results["metadatas"]:
            return []

        # Aggregate by doc_id
        docs: dict[str, dict] = {}
        for meta in results["metadatas"]:
            doc_id = meta["doc_id"]
            if doc_id not in docs:
                docs[doc_id] = {
                    "doc_id": doc_id,
                    "filename": meta["filename"],
                    "chunk_count": 0,
                }
            docs[doc_id]["chunk_count"] += 1

        return list(docs.values())

    def chunk_count(self) -> int:
        """Total number of chunks stored across all documents."""
        return self._collection.count()

    def document_exists(self, doc_id: str) -> bool:
        """Check whether a document has already been ingested."""
        results = self._collection.get(
            where={"doc_id": doc_id},
            limit=1,
        )
        return len(results["ids"]) > 0
