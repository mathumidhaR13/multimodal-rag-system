from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


# ---------- Ingestion Schemas ----------

class IngestResponse(BaseModel):
    """Response returned after a document is ingested."""
    message: str
    filename: str
    total_chunks: int
    status: str


# ---------- Query Schemas ----------

class QueryRequest(BaseModel):
    """Request body for querying the RAG system."""
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        example="What is the main topic of the document?"
    )
    top_k: Optional[int] = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of relevant chunks to retrieve"
    )


class RetrievedChunk(BaseModel):
    """A single retrieved chunk with its metadata."""
    chunk_id: int
    text: str
    source: str
    score: float


class QueryResponse(BaseModel):
    """Full response from the RAG pipeline."""
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    total_chunks_retrieved: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ---------- Document Schemas ----------

class DocumentChunk(BaseModel):
    """Internal model representing a text chunk from a document."""
    chunk_id: int
    text: str
    source: str
    page_number: Optional[int] = None
    chunk_index: int