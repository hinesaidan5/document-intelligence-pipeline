"""
Tests for the PDF ingestion and chunking pipeline.
Run with: pytest tests/test_pdf_processor.py -v
"""

import sys
import os
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from backend.ingestion.pdf_processor import PDFProcessor, DocumentChunk, ProcessedDocument


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sample_pdf(text: str, path: Path):
    """Create a minimal real PDF using reportlab (installed as a test dep)."""
    try:
        from reportlab.pdfgen import canvas
        c = canvas.Canvas(str(path))
        # Split text into lines and write to page
        y = 750
        for line in text.split(". "):
            if y < 50:
                c.showPage()
                y = 750
            c.drawString(50, y, line[:100])
            y -= 20
        c.save()
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestPDFProcessor:

    def test_rejects_non_pdf(self, tmp_path):
        processor = PDFProcessor()
        fake_txt = tmp_path / "file.txt"
        fake_txt.write_text("hello")
        with pytest.raises(ValueError, match="Expected a .pdf file"):
            processor.process(fake_txt)

    def test_rejects_missing_file(self, tmp_path):
        processor = PDFProcessor()
        with pytest.raises(FileNotFoundError):
            processor.process(tmp_path / "ghost.pdf")

    def test_chunk_size_respected(self, tmp_path):
        """Chunks should not exceed chunk_size + 30 word sentence lookahead."""
        if not make_sample_pdf(
            "The quick brown fox jumps over the lazy dog. " * 200,
            tmp_path / "test.pdf"
        ):
            pytest.skip("reportlab not installed")

        processor = PDFProcessor(chunk_size=50, chunk_overlap=5)
        doc = processor.process(tmp_path / "test.pdf")

        for chunk in doc.chunks:
            assert chunk.word_count <= 80, f"Chunk too large: {chunk.word_count} words"

    def test_overlap_creates_more_chunks(self, tmp_path):
        """Higher overlap should produce more chunks than no overlap."""
        if not make_sample_pdf(
            "Enterprise data platform. " * 300,
            tmp_path / "test.pdf"
        ):
            pytest.skip("reportlab not installed")

        proc_overlap = PDFProcessor(chunk_size=50, chunk_overlap=20)
        proc_no_overlap = PDFProcessor(chunk_size=50, chunk_overlap=0)

        doc_overlap = proc_overlap.process(tmp_path / "test.pdf")
        doc_no_overlap = proc_no_overlap.process(tmp_path / "test.pdf")

        assert doc_overlap.total_chunks >= doc_no_overlap.total_chunks

    def test_chunk_ids_are_unique(self, tmp_path):
        if not make_sample_pdf(
            "Government contract analysis. Dollar amounts. Dates. " * 150,
            tmp_path / "test.pdf"
        ):
            pytest.skip("reportlab not installed")

        processor = PDFProcessor()
        doc = processor.process(tmp_path / "test.pdf")

        ids = [c.chunk_id for c in doc.chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs found"

    def test_doc_id_stable(self, tmp_path):
        """Same file processed twice should yield the same doc_id."""
        if not make_sample_pdf("Stable document. " * 100, tmp_path / "stable.pdf"):
            pytest.skip("reportlab not installed")

        processor = PDFProcessor()
        doc1 = processor.process(tmp_path / "stable.pdf")
        doc2 = processor.process(tmp_path / "stable.pdf")
        assert doc1.doc_id == doc2.doc_id

    def test_chunk_metadata_contains_filename(self, tmp_path):
        if not make_sample_pdf("Metadata test. " * 100, tmp_path / "metadata.pdf"):
            pytest.skip("reportlab not installed")

        processor = PDFProcessor()
        doc = processor.process(tmp_path / "metadata.pdf")

        for chunk in doc.chunks:
            assert chunk.metadata.get("filename") == "metadata.pdf"

    def test_min_chunk_filter(self, tmp_path):
        """Chunks below min_chunk_words should be discarded."""
        if not make_sample_pdf("Short. " * 50, tmp_path / "short.pdf"):
            pytest.skip("reportlab not installed")

        processor = PDFProcessor(min_chunk_words=10)
        doc = processor.process(tmp_path / "short.pdf")

        for chunk in doc.chunks:
            assert chunk.word_count >= 10

    def test_page_numbers_assigned(self, tmp_path):
        """Every chunk should have a valid page number >= 1."""
        if not make_sample_pdf("Page number test. " * 200, tmp_path / "pages.pdf"):
            pytest.skip("reportlab not installed")

        processor = PDFProcessor()
        doc = processor.process(tmp_path / "pages.pdf")

        for chunk in doc.chunks:
            assert chunk.page_number >= 1


class TestProcessedDocument:

    def test_total_chunks_matches_list(self, tmp_path):
        if not make_sample_pdf("Consistency check. " * 200, tmp_path / "check.pdf"):
            pytest.skip("reportlab not installed")

        processor = PDFProcessor()
        doc = processor.process(tmp_path / "check.pdf")

        assert doc.total_chunks == len(doc.chunks)
