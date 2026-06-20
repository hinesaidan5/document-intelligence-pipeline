"""
PDF Processor — document-intelligence-pipeline
Handles ingestion, text extraction, and intelligent chunking of large PDF documents.
This is core feature
"""

import re
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import pdfplumber


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class DocumentChunk:
    """A single chunk of text extracted from a PDF, ready for embedding."""
    chunk_id: str               # Unique hash-based ID
    doc_id: str                 # Parent document ID
    text: str                   # The actual text content
    page_number: int            # Page this chunk came from
    chunk_index: int            # Position within the document
    word_count: int             # Word count for quick filtering
    metadata: dict = field(default_factory=dict)  # Filename, title, etc.


@dataclass
class ProcessedDocument:
    """Result of processing a full PDF file."""
    doc_id: str
    filename: str
    total_pages: int
    total_chunks: int
    chunks: list[DocumentChunk]
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core processor
# ---------------------------------------------------------------------------

class PDFProcessor:
    """
    Ingests PDF files, extracts text, and splits into overlapping chunks
    optimized for RAG retrieval.

    Chunking strategy:
    - Sentence-aware splits (respects sentence boundaries)
    - Configurable chunk size and overlap
    - Filters out noise (headers, footers, page numbers)
    """

    def __init__(
        self,
        chunk_size: int = 512,       # Target words per chunk
        chunk_overlap: int = 64,     # Overlap between consecutive chunks
        min_chunk_words: int = 20,   # Discard chunks smaller than this
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_words = min_chunk_words

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, pdf_path: str | Path) -> ProcessedDocument:
        """
        Main entry point. Accepts a path to a PDF and returns a
        ProcessedDocument containing all chunks.
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {path.suffix}")

        doc_id = self._generate_doc_id(path)
        pages = self._extract_pages(path)
        chunks = self._chunk_pages(pages, doc_id, path.name)

        return ProcessedDocument(
            doc_id=doc_id,
            filename=path.name,
            total_pages=len(pages),
            total_chunks=len(chunks),
            chunks=chunks,
            metadata={"source_path": str(path)},
        )

    # ------------------------------------------------------------------
    # Text extraction
    # ------------------------------------------------------------------

    def _extract_pages(self, path: Path) -> list[dict]:
        """Extract raw text from each page using pdfplumber."""
        pages = []
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                raw_text = page.extract_text() or ""
                cleaned = self._clean_text(raw_text)
                if cleaned:
                    pages.append({"page_number": page_num, "text": cleaned})
        return pages

    def _clean_text(self, text: str) -> str:
        """
        Remove common PDF noise: excessive whitespace, isolated page numbers,
        repeated header/footer lines.
        """
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove lines that are just numbers (page numbers)
        text = re.sub(r"(?m)^\s*\d{1,4}\s*$", "", text)
        # Remove very short lines (likely headers/footers)
        lines = [ln for ln in text.split("\n") if len(ln.strip()) > 10]
        return " ".join(lines).strip()

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _chunk_pages(
        self, pages: list[dict], doc_id: str, filename: str
    ) -> list[DocumentChunk]:
        """
        Combines all pages into a token stream, then produces overlapping
        chunks that respect sentence boundaries.
        """
        # Build a flat list of (word, page_number) tuples
        word_page_pairs: list[tuple[str, int]] = []
        for page in pages:
            for word in page["text"].split():
                word_page_pairs.append((word, page["page_number"]))

        chunks: list[DocumentChunk] = []
        chunk_index = 0
        start = 0
        total_words = len(word_page_pairs)

        while start < total_words:
            end = min(start + self.chunk_size, total_words)

            # Extend to the next sentence boundary (period, ?, !)
            end = self._extend_to_sentence_boundary(word_page_pairs, end, total_words)

            words_in_chunk = [w for w, _ in word_page_pairs[start:end]]
            text = " ".join(words_in_chunk)

            if len(words_in_chunk) >= self.min_chunk_words:
                # Determine the dominant page number for this chunk
                page_numbers = [p for _, p in word_page_pairs[start:end]]
                dominant_page = max(set(page_numbers), key=page_numbers.count)

                chunk = DocumentChunk(
                    chunk_id=self._generate_chunk_id(doc_id, chunk_index),
                    doc_id=doc_id,
                    text=text,
                    page_number=dominant_page,
                    chunk_index=chunk_index,
                    word_count=len(words_in_chunk),
                    metadata={"filename": filename},
                )
                chunks.append(chunk)
                chunk_index += 1

            # Advance with overlap
            start = end - self.chunk_overlap
            if start >= end:
                break  # Safety valve

        return chunks

    def _extend_to_sentence_boundary(
        self, word_page_pairs: list[tuple[str, int]], end: int, total: int
    ) -> int:
        """
        From position `end`, scan forward up to 30 words to find a sentence-
        ending punctuation mark so chunks don't cut mid-sentence.
        """
        lookahead = min(end + 30, total)
        for i in range(end, lookahead):
            word = word_page_pairs[i][0]
            if word.endswith((".", "?", "!")):
                return i + 1
        return end  # No boundary found, use original end

    # ------------------------------------------------------------------
    # ID generation
    # ------------------------------------------------------------------

    def _generate_doc_id(self, path: Path) -> str:
        """Stable ID based on filename + file size."""
        size = path.stat().st_size
        raw = f"{path.name}:{size}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """Unique ID for each chunk within a document."""
        raw = f"{doc_id}:{chunk_index}"
        return hashlib.md5(raw.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def process_pdf(pdf_path: str | Path, **kwargs) -> ProcessedDocument:
    """Shorthand for one-off processing without instantiating the class."""
    processor = PDFProcessor(**kwargs)
    return processor.process(pdf_path)
