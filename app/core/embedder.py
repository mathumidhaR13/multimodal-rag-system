import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingGenerator:
    """
    Generates dense vector embeddings for text chunks
    using Sentence Transformers.

    Model: all-MiniLM-L6-v2 (default)
    - Fast, lightweight, 384-dimensional embeddings
    - Great balance of speed vs quality for RAG systems
    - Downloads automatically on first use (~90MB)
    """

    _instance = None  # Singleton — load model only once

    def __new__(cls):
        """
        Singleton pattern to avoid reloading the model
        on every request. Model loads once and stays in memory.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        logger.info(
            f"🤖 Loading embedding model: {settings.EMBEDDING_MODEL}"
        )
        logger.info(
            "⏳ This may take a moment on first run (downloading model)..."
        )

        try:
            self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self._initialized = True

            logger.info(
                f"✅ Embedding model loaded | "
                f"Dimension: {self.embedding_dim}"
            )

        except Exception as e:
            logger.error(f"❌ Failed to load embedding model: {e}")
            raise

    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text string.

        Args:
            text: Input text string

        Returns:
            numpy array of shape (embedding_dim,)
        """
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True  # L2 normalize for cosine similarity
        )

        logger.debug(
            f"  🔢 Embedded text | "
            f"Length: {len(text)} chars | "
            f"Vector dim: {embedding.shape}"
        )

        return embedding

    def embed_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for a list of texts efficiently.
        Uses batching to handle large document sets.

        Args:
            texts:         List of text strings
            batch_size:    Number of texts per batch
            show_progress: Show progress bar during encoding

        Returns:
            numpy array of shape (num_texts, embedding_dim)
        """
        if not texts:
            raise ValueError("Cannot embed empty list of texts")

        logger.info(
            f"🔢 Embedding {len(texts)} chunks | "
            f"batch_size={batch_size}"
        )

        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=show_progress
            )

            logger.info(
                f"✅ Embeddings generated | "
                f"Shape: {embeddings.shape} | "
                f"dtype: {embeddings.dtype}"
            )

            return embeddings

        except Exception as e:
            logger.error(f"❌ Batch embedding failed: {e}")
            raise

    def embed_chunks(self, chunks) -> np.ndarray:
        """
        Convenience method — embed a list of DocumentChunk objects.

        Args:
            chunks: List of DocumentChunk objects

        Returns:
            numpy array of shape (num_chunks, embedding_dim)
        """
        texts = [chunk.text for chunk in chunks]
        return self.embed_batch(texts)

    @property
    def dimension(self) -> int:
        """Return embedding dimension size."""
        return self.embedding_dim