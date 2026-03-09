from typing import List, Dict, Any
from app.core.embedder import EmbeddingGenerator
from app.core.vector_store import FAISSVectorStore
from app.models.schemas import RetrievedChunk
from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RAGRetriever:
    """
    Orchestrates the full RAG retrieval pipeline:

    Query Text
        ↓
    Embed Query (Sentence Transformer)
        ↓
    Search FAISS Index
        ↓
    Return Ranked Chunks with Scores

    This class is the bridge between the user's question
    and the vector store — it handles everything needed
    to find the most relevant context chunks.
    """

    def __init__(self):
        logger.info("🔗 Initializing RAG Retriever Pipeline...")

        # Load embedding model (singleton — won't reload)
        self.embedder = EmbeddingGenerator()

        # Initialize vector store and try loading saved index
        self.vector_store = FAISSVectorStore(
            embedding_dim=self.embedder.dimension
        )
        loaded = self.vector_store.load()

        if loaded:
            logger.info(
                f"✅ RAG Retriever ready | "
                f"Index loaded with {self.vector_store.total_vectors} vectors"
            )
        else:
            logger.warning(
                "⚠️  RAG Retriever ready but index is EMPTY — "
                "please ingest documents first"
            )

    def retrieve(
        self,
        query: str,
        top_k: int = None
    ) -> List[RetrievedChunk]:
        """
        Main retrieval method.
        Takes a query string and returns the top-k
        most relevant document chunks.

        Args:
            query:  User's question or search query
            top_k:  Number of chunks to retrieve

        Returns:
            List of RetrievedChunk objects ranked by similarity
        """
        top_k = top_k or settings.TOP_K_RESULTS

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        if self.vector_store.total_vectors == 0:
            logger.warning(
                "⚠️  Cannot retrieve — vector store is empty"
            )
            return []

        logger.info(f"🔍 Retrieving for query: '{query[:80]}...' | top_k={top_k}")

        # Step 1 — Embed the query
        query_vector = self.embedder.embed_text(query)
        logger.debug(f"  Query embedded | dim={query_vector.shape}")

        # Step 2 — Search FAISS index
        raw_results = self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k
        )

        # Step 3 — Convert to RetrievedChunk schema
        retrieved_chunks = []
        for result in raw_results:
            chunk = RetrievedChunk(
                chunk_id=result["chunk_id"],
                text=result["text"],
                source=result["source"],
                score=round(result["score"], 4)
            )
            retrieved_chunks.append(chunk)

        logger.info(
            f"✅ Retrieved {len(retrieved_chunks)} chunks | "
            f"Top score: {retrieved_chunks[0].score if retrieved_chunks else 'N/A'}"
        )

        return retrieved_chunks

    def build_context(
        self,
        chunks: List[RetrievedChunk],
        max_context_length: int = 2000
    ) -> str:
        """
        Combine retrieved chunks into a single context string
        to pass to the LLM.

        Respects a max character limit to avoid exceeding
        LLM context window.

        Args:
            chunks:             List of RetrievedChunk objects
            max_context_length: Max characters for combined context

        Returns:
            Formatted context string
        """
        if not chunks:
            return ""

        context_parts = []
        total_length = 0

        for i, chunk in enumerate(chunks):
            # Format each chunk with source info
            chunk_text = (
                f"[Source: {chunk.source} | Score: {chunk.score}]\n"
                f"{chunk.text}"
            )

            # Stop if adding this chunk exceeds the limit
            if total_length + len(chunk_text) > max_context_length:
                logger.debug(
                    f"  Context limit reached at chunk {i+1} — "
                    f"stopping at {total_length} chars"
                )
                break

            context_parts.append(chunk_text)
            total_length += len(chunk_text)

        context = "\n\n---\n\n".join(context_parts)

        logger.debug(
            f"  📝 Context built | "
            f"Chunks used: {len(context_parts)}/{len(chunks)} | "
            f"Total chars: {len(context)}"
        )

        return context

    def add_documents(
        self,
        embeddings,
        chunks
    ) -> int:
        """
        Add new document embeddings to the vector store
        and persist to disk.

        Args:
            embeddings: numpy array of embeddings
            chunks:     List of DocumentChunk objects

        Returns:
            Total vectors in index after adding
        """
        total = self.vector_store.add_embeddings(embeddings, chunks)
        self.vector_store.save()

        logger.info(
            f"📥 Documents added to retriever | "
            f"Total vectors: {total}"
        )

        return total

    def get_index_stats(self) -> Dict[str, Any]:
        """Return current index statistics."""
        return self.vector_store.get_stats()

    def reset_index(self) -> None:
        """Clear the entire vector store."""
        self.vector_store.reset()
        logger.info("🗑️  Retriever index has been reset")