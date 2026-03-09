import faiss
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from app.models.schemas import DocumentChunk
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class FAISSVectorStore:
    """
    Manages FAISS index for storing and retrieving
    document chunk embeddings.

    Features:
    - Add embeddings with metadata
    - Similarity search by query vector
    - Persist index to disk
    - Load index from disk
    - Reset/clear index
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        self.index_dir = Path(settings.FAISS_INDEX_DIR)
        self.index_path = self.index_dir / "faiss.index"
        self.metadata_path = self.index_dir / "metadata.json"

        # FAISS index — IndexFlatIP = Inner Product (cosine sim for normalized vectors)
        self.index = faiss.IndexFlatIP(self.embedding_dim)

        # Metadata store — maps FAISS integer ID → chunk metadata
        self.metadata: List[Dict[str, Any]] = []

        logger.info(
            f"🗄️  FAISSVectorStore initialized | "
            f"dim={self.embedding_dim}"
        )

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        chunks: List[DocumentChunk]
    ) -> int:
        """
        Add embeddings and their corresponding chunk metadata
        to the FAISS index.

        Args:
            embeddings: numpy array of shape (N, embedding_dim)
            chunks:     List of DocumentChunk objects (same order)

        Returns:
            Total number of vectors now in the index
        """
        if len(embeddings) != len(chunks):
            raise ValueError(
                f"Mismatch: {len(embeddings)} embeddings "
                f"vs {len(chunks)} chunks"
            )

        # Ensure float32 — FAISS requirement
        embeddings = np.array(embeddings, dtype=np.float32)

        # Add to FAISS index
        self.index.add(embeddings)

        # Store metadata for each chunk
        for chunk in chunks:
            self.metadata.append({
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "source": chunk.source,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index
            })

        total = self.index.ntotal
        logger.info(
            f"✅ Added {len(chunks)} vectors | "
            f"Total in index: {total}"
        )

        return total

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Search the FAISS index for most similar chunks.

        Args:
            query_vector: numpy array of shape (embedding_dim,)
            top_k:        Number of top results to return

        Returns:
            List of dicts with chunk metadata + similarity score
        """
        top_k = top_k or settings.TOP_K_RESULTS

        if self.index.ntotal == 0:
            logger.warning("⚠️  FAISS index is empty — no results")
            return []

        # Reshape query to (1, dim) for FAISS
        query = np.array(query_vector, dtype=np.float32).reshape(1, -1)

        # Perform search
        scores, indices = self.index.search(query, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue

            result = self.metadata[idx].copy()
            result["score"] = float(score)
            results.append(result)

        logger.info(
            f"🔍 Search complete | "
            f"top_k={top_k} | "
            f"results found: {len(results)}"
        )

        return results

    def save(self) -> None:
        """
        Persist the FAISS index and metadata to disk.
        """
        try:
            self.index_dir.mkdir(parents=True, exist_ok=True)

            # Save FAISS binary index
            faiss.write_index(self.index, str(self.index_path))

            # Save metadata as JSON
            with open(self.metadata_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata, f, ensure_ascii=False, indent=2)

            logger.info(
                f"💾 Index saved | "
                f"Vectors: {self.index.ntotal} | "
                f"Path: {self.index_path}"
            )

        except Exception as e:
            logger.error(f"❌ Failed to save index: {e}")
            raise

    def load(self) -> bool:
        """
        Load the FAISS index and metadata from disk.

        Returns:
            True if loaded successfully, False if no saved index found
        """
        if not self.index_path.exists() or not self.metadata_path.exists():
            logger.warning("⚠️  No saved index found — starting fresh")
            return False

        try:
            # Load FAISS index
            self.index = faiss.read_index(str(self.index_path))

            # Load metadata
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

            logger.info(
                f"📂 Index loaded | "
                f"Vectors: {self.index.ntotal} | "
                f"Metadata entries: {len(self.metadata)}"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Failed to load index: {e}")
            raise

    def reset(self) -> None:
        """
        Clear the FAISS index and metadata completely.
        Also deletes saved files on disk.
        """
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.metadata = []

        # Remove saved files
        if self.index_path.exists():
            os.remove(self.index_path)
        if self.metadata_path.exists():
            os.remove(self.metadata_path)

        logger.info("🗑️  FAISS index reset and cleared")

    @property
    def total_vectors(self) -> int:
        """Return total number of vectors in the index."""
        return self.index.ntotal

    def get_stats(self) -> Dict[str, Any]:
        """Return stats about the current index."""
        sources = list(set(m["source"] for m in self.metadata))
        return {
            "total_vectors": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "total_documents": len(sources),
            "documents": sources
        }