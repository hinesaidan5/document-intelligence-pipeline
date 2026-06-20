"""
Tests for the ChromaDB vector storage module.
Run with: pytest tests/test_vector_store.py -v
"""

import sys
import os
import shutil
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from backend.storage.vector_store import VectorStore, SearchResult
from backend.ingestion.pdf_processor import ProcessedDocument, DocumentChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fake_document(doc_id: str, filename: str, texts: list[str]) -> ProcessedDocument:
    """Build a ProcessedDocument without needing a real PDF."""
    chunks = [
        DocumentChunk(
            chunk_id=f"{doc_id}_chunk_{i}",
            doc_id=doc_id,
            text=text,
            page_number=i + 1,
            chunk_index=i,
            word_count=len(text.split()),
            metadata={"filename": filename},
        )
        for i, text in enumerate(texts)
    ]
    return ProcessedDocument(
        doc_id=doc_id,
        filename=filename,
        total_pages=len(texts),
        total_chunks=len(chunks),
        chunks=chunks,
    )


@pytest.fixture
def temp_store():
    """Fresh VectorStore backed by a temp directory, cleaned up after the test."""
    temp_dir = tempfile.mkdtemp()
    store = VectorStore(persist_dir=temp_dir)
    yield store
    shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestVectorStore:

    def test_add_document_returns_chunk_count(self, temp_store):
        doc = make_fake_document(
            "doc1", "contract.pdf",
            ["The contract value is $2.5 million.", "Signed by John Smith on March 1st."]
        )
        added = temp_store.add_document(doc)
        assert added == 2

    def test_chunk_count_reflects_storage(self, temp_store):
        doc = make_fake_document("doc1", "contract.pdf", ["Text one.", "Text two.", "Text three."])
        temp_store.add_document(doc)
        assert temp_store.chunk_count() == 3

    def test_search_returns_results(self, temp_store):
        doc = make_fake_document(
            "doc1", "contract.pdf",
            [
                "The total contract value is $2.5 million dollars.",
                "The project deadline is December 31st 2026.",
                "John Smith is the primary contractor on this agreement.",
            ]
        )
        temp_store.add_document(doc)

        results = temp_store.search("How much is the contract worth?", top_k=2)
        assert len(results) > 0
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_returns_most_relevant_first(self, temp_store):
        doc = make_fake_document(
            "doc1", "contract.pdf",
            [
                "Bananas are a good source of potassium.",
                "The contract was signed for $5 million dollars on January 1st.",
                "The weather in Ohio was sunny today.",
            ]
        )
        temp_store.add_document(doc)

        results = temp_store.search("What was the contract dollar amount?", top_k=3)
        # The contract-related chunk should be most similar
        assert "5 million" in results[0].text or "contract" in results[0].text.lower()

    def test_document_exists(self, temp_store):
        doc = make_fake_document("doc1", "contract.pdf", ["Some text here."])
        assert temp_store.document_exists("doc1") is False
        temp_store.add_document(doc)
        assert temp_store.document_exists("doc1") is True

    def test_delete_document_removes_chunks(self, temp_store):
        doc = make_fake_document("doc1", "contract.pdf", ["Text one.", "Text two."])
        temp_store.add_document(doc)
        deleted = temp_store.delete_document("doc1")
        assert deleted == 2
        assert temp_store.chunk_count() == 0

    def test_search_respects_doc_id_filter(self, temp_store):
        doc1 = make_fake_document("doc1", "contractA.pdf", ["Contract A is worth $1 million."])
        doc2 = make_fake_document("doc2", "contractB.pdf", ["Contract B is worth $9 million."])
        temp_store.add_document(doc1)
        temp_store.add_document(doc2)

        results = temp_store.search("contract value", top_k=5, doc_id="doc1")
        assert all(r.doc_id == "doc1" for r in results)

    def test_list_documents_aggregates_correctly(self, temp_store):
        doc1 = make_fake_document("doc1", "a.pdf", ["chunk1", "chunk2"])
        doc2 = make_fake_document("doc2", "b.pdf", ["chunk1", "chunk2", "chunk3"])
        temp_store.add_document(doc1)
        temp_store.add_document(doc2)

        docs = temp_store.list_documents()
        doc_map = {d["doc_id"]: d for d in docs}

        assert doc_map["doc1"]["chunk_count"] == 2
        assert doc_map["doc2"]["chunk_count"] == 3

    def test_add_document_with_no_chunks_returns_zero(self, temp_store):
        empty_doc = ProcessedDocument(
            doc_id="empty", filename="empty.pdf",
            total_pages=0, total_chunks=0, chunks=[]
        )
        added = temp_store.add_document(empty_doc)
        assert added == 0

    def test_persistence_across_instances(self, tmp_path):
        """Data written by one VectorStore instance should be readable by another."""
        store1 = VectorStore(persist_dir=tmp_path)
        doc = make_fake_document("doc1", "persist.pdf", ["Persistent data test."])
        store1.add_document(doc)

        # New instance, same directory
        store2 = VectorStore(persist_dir=tmp_path)
        assert store2.document_exists("doc1") is True
