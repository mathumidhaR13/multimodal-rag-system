from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.utils.logger import get_logger
from app.api.routes import ingest, query

logger = get_logger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Multimodal RAG System — FastAPI + FAISS + Sentence Transformers + LLM"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(ingest.router)
app.include_router(query.router)


@app.on_event("startup")
async def startup_event():
    logger.info(
        f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} starting..."
    )


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Application shutting down...")


@app.get("/", tags=["Health"])
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy"}