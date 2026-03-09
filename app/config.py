from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    # App Info
    APP_NAME: str = "MultimodalRAG"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Directories
    UPLOAD_DIR: str = "data/uploads"
    FAISS_INDEX_DIR: str = "data/faiss_index"

    # Embedding Model
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # LLM Model
    LLM_MODEL: str = "google/flan-t5-base"

    # Chunking
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50

    # Retrieval
    TOP_K_RESULTS: int = 5

    class Config:
        env_file = ".env"
        extra = "allow"

# Single global instance — import this everywhere
settings = Settings()

# Ensure required directories exist on startup
Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
Path(settings.FAISS_INDEX_DIR).mkdir(parents=True, exist_ok=True)