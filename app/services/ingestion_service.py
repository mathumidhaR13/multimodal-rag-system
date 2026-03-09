import shutil
from pathlib import Path
from typing import Dict, Any
from app.services.pdf_parser import PDFParser
from app.core.chunker import TextChunker
from app.core.embedder import EmbeddingGenerator
from app.core.retriever import RAGRetriever
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class IngestionService:
    """
    Orchestrates the full document ingestion pipeline:

    PDF Upload
        ↓
    Save to Disk
        ↓
    Parse PDF (extract text by page)
        ↓
    Chunk Text
        ↓
    Generate Embeddings
        ↓
    Store in FAISS
        ↓
    Persist Index to Disk
    """

    def __init__(self):
        self.upload_dir = Path(settings.UPLOAD_DIR)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

        # Initialize pipeline components
        self.chunker = TextChunker(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )
        self.embedder = EmbeddingGenerator()
        self.retriever = RAGRetriever()

        logger.info("✅ IngestionService initialized")

    def ingest_pdf(
        self,
        file_path: str,
        filename: str
    ) -> Dict[str, Any]:
        """
        Run the full ingestion pipeline for a PDF file.

        Args:
            file_path: Path to the uploaded PDF file
            filename:  Original filename (used as source identifier)

        Returns:
            Dict with ingestion results and stats
        """
        logger.info(f"📥 Starting ingestion for: {filename}")

        try:
            # Step 1 — Parse PDF
            logger.info("  [1/4] Parsing PDF...")
            parser = PDFParser(file_path)
            metadata = parser.get_metadata()
            pages = parser.extract_text_by_page()

            if not pages:
                raise ValueError(
                    f"No text could be extracted from '{filename}'. "
                    "The PDF may be scanned or image-based."
                )

            logger.info(
                f"  ✅ Parsed {len(pages)} pages | "
                f"{metadata['page_count']} total pages in PDF"
            )

            # Step 2 — Chunk text
            logger.info("  [2/4] Chunking text...")
            chunks = self.chunker.chunk_pages(pages, source=filename)
            chunk_stats = self.chunker.get_chunk_stats(chunks)

            logger.info(
                f"  ✅ Created {len(chunks)} chunks | "
                f"Avg words: {chunk_stats['avg_words_per_chunk']}"
            )

            # Step 3 — Generate embeddings
            logger.info("  [3/4] Generating embeddings...")
            embeddings = self.embedder.embed_chunks(chunks)

            logger.info(
                f"  ✅ Embeddings shape: {embeddings.shape}"
            )

            # Step 4 — Store in FAISS
            logger.info("  [4/4] Storing in FAISS vector store...")
            total_vectors = self.retriever.add_documents(
                embeddings, chunks
            )

            logger.info(
                f"  ✅ Stored | Total vectors in index: {total_vectors}"
            )

            result = {
                "message": "Document ingested successfully",
                "filename": filename,
                "total_chunks": len(chunks),
                "total_pages": len(pages),
                "total_vectors_in_index": total_vectors,
                "chunk_stats": chunk_stats,
                "pdf_metadata": metadata,
                "status": "success"
            }

            logger.info(
                f"🎉 Ingestion complete for '{filename}' | "
                f"Chunks: {len(chunks)}"
            )

            return result

        except Exception as e:
            logger.error(
                f"❌ Ingestion failed for '{filename}': {e}"
            )
            raise

    def save_upload(
        self,
        file_content: bytes,
        filename: str
    ) -> str:
        """
        Save uploaded file bytes to the uploads directory.

        Args:
            file_content: Raw bytes of the uploaded file
            filename:     Original filename

        Returns:
            Full path to the saved file
        """
        save_path = self.upload_dir / filename

        with open(save_path, "wb") as f:
            f.write(file_content)

        logger.info(
            f"💾 File saved | "
            f"Path: {save_path} | "
            f"Size: {round(len(file_content)/1024, 2)} KB"
        )

        return str(save_path)