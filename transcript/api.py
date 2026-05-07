from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from typing import Optional, List
import logging
import os
from datetime import datetime
import asyncio
from contextlib import asynccontextmanager

from config import settings
from vllm_server import VLLMServer
from context_manager import ContextManager

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

vllm_server = None
context_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global vllm_server, context_manager
    
    logger.info("Starting up application...")
    
    try:
        vllm_server = VLLMServer()
        vllm_server.load_model()
        logger.info("VLLM server initialized")
    except Exception as e:
        logger.error(f"Failed to initialize VLLM server: {e}")
        vllm_server = None
    
    try:
        context_manager = ContextManager()
        logger.info("Context manager initialized")
    except Exception as e:
        logger.error(f"Failed to initialize context manager: {e}")
        context_manager = None
    
    yield
    
    logger.info("Shutting down application...")

app = FastAPI(
    title="Video-Trained Chatbot API",
    description="AI chatbot trained on video transcripts with context-aware responses",
    version="1.0.0",
    lifespan=lifespan
)

origins = settings.CORS_ORIGINS.split(",") if settings.CORS_ORIGINS != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str
    temperature: Optional[float] = 0.7
    use_context: Optional[bool] = True

class ChatResponse(BaseModel):
    response: str
    context_used: bool
    sources: Optional[List[dict]] = None
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    context_manager_ready: bool

@app.get("/", response_class=FileResponse)
async def read_root():
    static_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if os.path.exists(static_path):
        return FileResponse(static_path)
    return {"message": "Video-Trained Chatbot API", "docs": "/docs"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=vllm_server is not None,
        context_manager_ready=context_manager is not None
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if vllm_server is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        context = ""
        sources = []
        context_used = False
        
        if request.use_context and context_manager is not None:
            is_relevant = context_manager.is_query_relevant(request.message)
            
            if not is_relevant:
                return ChatResponse(
                    response="I apologize, but I can only answer questions related to the topics covered in my training videos. Your question appears to be outside my knowledge domain. Please ask something related to the video content I was trained on.",
                    context_used=False,
                    sources=None,
                    timestamp=datetime.now().isoformat()
                )
            
            context_result = context_manager.search_context(request.message)
            
            if context_result["found"]:
                context = context_result["context"]
                sources = context_result["sources"]
                context_used = True
        
        response = vllm_server.chat(
            user_message=request.message,
            context=context,
            temperature=request.temperature
        )
        
        return ChatResponse(
            response=response,
            context_used=context_used,
            sources=sources if context_used else None,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    try:
        transcript_dir = settings.TRANSCRIPT_DIR
        training_dir = settings.TRAINING_DATA_DIR
        
        transcript_count = 0
        if os.path.exists(transcript_dir):
            transcript_count = len([f for f in os.listdir(transcript_dir) if f.endswith('.json') and f != 'transcription_summary.json'])
        
        training_examples = 0
        if os.path.exists(os.path.join(training_dir, "training_stats.json")):
            import json
            with open(os.path.join(training_dir, "training_stats.json"), 'r') as f:
                stats = json.load(f)
                training_examples = stats.get("total_examples", 0)
        
        return {
            "transcripts": transcript_count,
            "training_examples": training_examples,
            "model_status": "loaded" if vllm_server else "not loaded",
            "context_db_status": "ready" if context_manager else "not ready"
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return {"error": str(e)}

app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False,
        log_level=settings.LOG_LEVEL.lower()
    )
