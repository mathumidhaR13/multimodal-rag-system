from typing import List, Dict, Any
from app.models.schemas import DocumentChunk
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class TextChunker:
    """
    Splits extracted PDF text into overlapping chunks
    for embedding and retrieval.

    Strategy:
    - Split by words (not characters) for cleaner boundaries
    - Overlap between chunks to preserve context
    - Each chunk tagged with metadata (source, page, index)
    """

    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None
    ):
        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        logger.info(
            f"🔧 TextChunker initialized | "
            f"chunk_size={self.chunk_size} | "
            f"chunk_overlap={self.chunk_overlap}"
        )

    def chunk_text(
        self,
        text: str,
        source: str,
        page_number: int = None
    ) -> List[DocumentChunk]:
        """
        Split a single page/block of text into overlapping chunks.

        Args:
            text:        Raw text string to chunk
            source:      Filename or document identifier
            page_number: Page number this text came from

        Returns:
            List of DocumentChunk objects
        """
        words = text.split()
        chunks = []
        chunk_index = 0
        start = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words).strip()

            if chunk_text:
                chunks.append(DocumentChunk(
                    chunk_id=chunk_index,
                    text=chunk_text,
                    source=source,
                    page_number=page_number,
                    chunk_index=chunk_index
                ))
                chunk_index += 1

            # Move forward by (chunk_size - overlap)
            step = self.chunk_size - self.chunk_overlap
            start += step

        logger.debug(
            f"  📃 Page {page_number}: "
            f"{len(words)} words → {len(chunks)} chunks"
        )

        return chunks

    def chunk_pages(
        self,
        pages: List[Dict[str, Any]],
        source: str
    ) -> List[DocumentChunk]:
        """
        Process all pages from a PDF and return all chunks.

        Args:
            pages:  List of page dicts from PDFParser
                    Each dict: {page_number, text, word_count}
            source: Document filename/identifier

        Returns:
            Flat list of all DocumentChunks across all pages
        """
        all_chunks = []
        global_chunk_id = 0

        logger.info(
            f"✂️  Chunking document: '{source}' | "
            f"{len(pages)} pages"
        )

        for page in pages:
            page_chunks = self.chunk_text(
                text=page["text"],
                source=source,
                page_number=page["page_number"]
            )

            # Re-assign global chunk IDs across pages
            for chunk in page_chunks:
                chunk.chunk_id = global_chunk_id
                chunk.chunk_index = global_chunk_id
                global_chunk_id += 1

            all_chunks.extend(page_chunks)

        logger.info(
            f"✅ Chunking complete: {len(all_chunks)} total chunks "
            f"from {len(pages)} pages"
        )

        return all_chunks

    def get_chunk_stats(
        self,
        chunks: List[DocumentChunk]
    ) -> Dict[str, Any]:
        """
        Returns statistics about the generated chunks.
        Useful for debugging and logging.
        """
        if not chunks:
            return {"total_chunks": 0}

        word_counts = [len(c.text.split()) for c in chunks]

        return {
            "total_chunks": len(chunks),
            "avg_words_per_chunk": round(
                sum(word_counts) / len(word_counts), 1
            ),
            "min_words": min(word_counts),
            "max_words": max(word_counts),
            "total_words": sum(word_counts)
        }