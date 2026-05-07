from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    MODEL_NAME: str = "google/gemma-2b-it"
    MODEL_PATH: str = "./models/fine_tuned_gemma"
    VLLM_GPU_MEMORY_UTILIZATION: float = 0.9
    VLLM_MAX_MODEL_LEN: int = 2048
    
    WHISPER_MODEL: str = "base"
    VIDEO_DIR: str = "./data/videos"
    TRANSCRIPT_DIR: str = "./data/transcripts"
    TRAINING_DATA_DIR: str = "./data/training"
    
    VECTOR_DB_PATH: str = "./data/vectordb"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    SIMILARITY_THRESHOLD: float = 0.7
    MAX_CONTEXT_LENGTH: int = 1500
    
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    CORS_ORIGINS: str = "*"
    
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "production"
    HUGGING_FACE_TOKEN: str = ""
    
    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
