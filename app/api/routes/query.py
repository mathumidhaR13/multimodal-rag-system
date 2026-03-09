from fastapi import APIRouter, HTTPException
from app.core.retriever import RAGRetriever
from app.core.llm import LLMGenerator
from app.models.schemas import QueryRequest, QueryResponse
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/query", tags=["Query"])

# Shared instances
retriever = RAGRetriever()
llm = LLMGenerator()


@router.post(
    "/",
    response_model=QueryResponse,
    summary="Query the RAG system",
    description="Ask a question and get an answer from your documents."
)
async def query_documents(request: QueryRequest):

    if retriever.vector_store.total_vectors == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents ingested yet. "
                   "Please upload a PDF first via /ingest."
        )

    try:
        logger.info(
            f"❓ Query received: '{request.question[:80]}'"
        )

        # Step 1 — Retrieve relevant chunks
        retrieved_chunks = retriever.retrieve(
            query=request.question,
            top_k=request.top_k
        )

        if not retrieved_chunks:
            return QueryResponse(
                question=request.question,
                answer="No relevant content found in the documents.",
                retrieved_chunks=[],
                total_chunks_retrieved=0
            )

        # Step 2 — Build context from retrieved chunks
        context = retriever.build_context(
            chunks=retrieved_chunks,
            max_context_length=2000
        )

        # Step 3 — Generate answer using LLM
        answer = llm.answer(
            question=request.question,
            context=context,
            max_new_tokens=256
        )

        logger.info(
            f"✅ Query answered | "
            f"Chunks used: {len(retrieved_chunks)} | "
            f"Answer length: {len(answer)} chars"
        )

        return QueryResponse(
            question=request.question,
            answer=answer,
            retrieved_chunks=retrieved_chunks,
            total_chunks_retrieved=len(retrieved_chunks)
        )

    except Exception as e:
        logger.error(f"❌ Query endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Query failed: {str(e)}"
        )


@router.get(
    "/health",
    summary="Check RAG pipeline health",
    description="Verify embedder, retriever and LLM are all loaded."
)
async def query_health():
    return {
        "embedder": "loaded",
        "retriever_vectors": retriever.vector_store.total_vectors,
        "llm_model": llm.model_name,
        "status": "healthy"
    }