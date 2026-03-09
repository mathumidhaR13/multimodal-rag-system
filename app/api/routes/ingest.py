from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from app.services.ingestion_service import IngestionService
from app.models.schemas import IngestResponse
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ingest", tags=["Ingestion"])

# Single shared instance
ingestion_service = IngestionService()


@router.post(
    "/",
    response_model=IngestResponse,
    summary="Upload and ingest a PDF document",
    description="Upload a PDF file to parse, chunk, embed and store in FAISS."
)
async def ingest_document(
    file: UploadFile = File(
        ...,
        description="PDF file to ingest"
    )
):
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported."
        )

    # Validate file size (max 50MB)
    MAX_SIZE = 50 * 1024 * 1024
    content = await file.read()

    if len(content) > MAX_SIZE:
        raise HTTPException(
            status_code=400,
            detail="File too large. Maximum size is 50MB."
        )

    if len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty."
        )

    try:
        logger.info(
            f"📤 Received upload: {file.filename} | "
            f"Size: {round(len(content)/1024, 2)} KB"
        )

        # Save file to disk
        file_path = ingestion_service.save_upload(
            file_content=content,
            filename=file.filename
        )

        # Run ingestion pipeline
        result = ingestion_service.ingest_pdf(
            file_path=file_path,
            filename=file.filename
        )

        return IngestResponse(
            message=result["message"],
            filename=result["filename"],
            total_chunks=result["total_chunks"],
            status=result["status"]
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    except Exception as e:
        logger.error(f"❌ Ingestion endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )


@router.get(
    "/stats",
    summary="Get vector store statistics",
    description="Returns stats about the current FAISS index."
)
async def get_index_stats():
    try:
        stats = ingestion_service.retriever.get_index_stats()
        return {
            "status": "success",
            "index_stats": stats
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get stats: {str(e)}"
        )


@router.delete(
    "/reset",
    summary="Reset the vector store",
    description="Clears all vectors and metadata from FAISS index."
)
async def reset_index():
    try:
        ingestion_service.retriever.reset_index()
        return {
            "status": "success",
            "message": "Vector store has been reset successfully."
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to reset index: {str(e)}"
        )