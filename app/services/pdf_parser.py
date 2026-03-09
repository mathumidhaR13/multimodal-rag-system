import fitz  # PyMuPDF
import pdfplumber
from pathlib import Path
from typing import List, Dict, Any
from app.utils.logger import get_logger

logger = get_logger(__name__)


class PDFParser:
    """
    Handles PDF text and metadata extraction.
    Uses PyMuPDF for speed + pdfplumber for table-heavy PDFs.
    """

    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

    def extract_text_by_page(self) -> List[Dict[str, Any]]:
        """
        Extract text from each page using PyMuPDF.
        Returns a list of dicts: {page_number, text, word_count}
        """
        pages = []

        try:
            doc = fitz.open(str(self.file_path))
            logger.info(f"📄 Opened PDF: {self.file_path.name} | Pages: {len(doc)}")

            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text("text").strip()

                if text:  # Skip empty pages
                    pages.append({
                        "page_number": page_num + 1,
                        "text": text,
                        "word_count": len(text.split())
                    })
                    logger.debug(
                        f"  Page {page_num + 1}: {len(text.split())} words extracted"
                    )

            doc.close()
            logger.info(
                f"✅ Extraction complete: {len(pages)} pages with content"
            )

        except Exception as e:
            logger.error(f"❌ Failed to parse PDF: {e}")
            raise

        return pages

    def extract_full_text(self) -> str:
        """
        Extract all text from the PDF as a single string.
        """
        pages = self.extract_text_by_page()
        full_text = "\n\n".join([p["text"] for p in pages])
        logger.info(
            f"📝 Full text extracted: {len(full_text)} characters"
        )
        return full_text

    def get_metadata(self) -> Dict[str, Any]:
        """
        Extract PDF metadata (title, author, page count, etc.)
        """
        try:
            doc = fitz.open(str(self.file_path))
            metadata = doc.metadata
            page_count = len(doc)
            doc.close()

            return {
                "filename": self.file_path.name,
                "title": metadata.get("title", "Unknown"),
                "author": metadata.get("author", "Unknown"),
                "page_count": page_count,
                "file_size_kb": round(
                    self.file_path.stat().st_size / 1024, 2
                )
            }
        except Exception as e:
            logger.error(f"❌ Failed to get metadata: {e}")
            raise