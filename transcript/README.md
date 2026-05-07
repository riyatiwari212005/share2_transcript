# Video-Trained AI Chatbot

A production-ready chatbot system that transcribes video content, fine-tunes Google's Gemma model to match the tone and style of the videos, and serves responses using VLLM for high-performance inference.

## 🌟 Features

- **Video Transcription**: Automatically transcribe video files using OpenAI Whisper with tone preservation
- **Fine-tuned Gemma Model**: Train Gemma to respond in the same tone and style as your video content
- **Context-Aware Responses**: Uses ChromaDB vector database to ensure responses stay within video content scope
- **Off-Topic Detection**: Automatically rejects queries unrelated to training material
- **VLLM Inference**: High-performance model serving with GPU optimization
- **Modern UI**: Beautiful, responsive web interface built with vanilla HTML/CSS/JS
- **Production Ready**: Docker support, health checks, and comprehensive logging

## 📋 Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended: 16GB+ VRAM)
- NVIDIA Docker runtime (for containerized deployment)
- FFmpeg

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project directory
cd chatbot/transcript

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Hugging Face Authentication (Required for Gemma)

Gemma models require authentication on Hugging Face. Follow these steps:

**Step A: Create a Hugging Face Account**
1. Go to [huggingface.co](https://huggingface.co)
2. Sign up for a free account

**Step B: Accept the Gemma Model License**
1. Visit [google/gemma-2b-it](https://huggingface.co/google/gemma-2b-it)
2. Click "Access Repository" button
3. Read and accept the terms of use
4. Repeat for any other Gemma model you plan to use

**Step C: Get Your Access Token**
1. Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Name it (e.g., "video-chatbot")
4. Select "Read" role
5. Click "Generate token"
6. Copy the token

**Step D: Configure Your Token**
```bash
# Edit the .env file
# Add your token:
HUGGING_FACE_TOKEN=your_token_here
```

### 3. Download the Model

```bash
# Option 1: Use the download script (recommended)
python download_model.py

# Option 2: Download specific model
python download_model.py --model google/gemma-7b-it --output ./models/gemma7b

# Option 3: List available Gemma models
python download_model.py --list

# Option 4: Check if authentication works
python download_model.py --check
```

This downloads the model to your local machine so you don't need to re-download it.

### 4. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your settings
# Key settings:
# - MODEL_NAME: Base Gemma model to use
# - MODEL_PATH: Where the downloaded model is saved
# - WHISPER_MODEL: Whisper model size (tiny/base/small/medium/large)
# - SIMILARITY_THRESHOLD: Context relevance threshold (0.0-1.0)
# - HUGGING_FACE_TOKEN: Your Hugging Face access token
```

### 5. Prepare Your Data

```bash
# Create data directories
mkdir -p data/videos data/transcripts data/training data/vectordb models

# Add your video files to data/videos/
# Supported formats: .mp4, .avi, .mkv, .mov, .flv, .wmv
```

### 6. Transcribe Videos

```bash
# Run transcription
python transcribe.py

# This will:
# - Extract audio from videos
# - Transcribe using Whisper
# - Save transcripts to data/transcripts/
# - Preserve timing and tone information
```

### 5. Prepare Training Data

```bash
# Generate training dataset from transcripts
python prepare_training_data.py

# This creates:
# - Conversation pairs from video segments
# - Gemma-formatted training data
# - Training statistics
```

### 8. Fine-tune the Model

```bash
# Train Gemma on your video content
python train_model.py

# This will:
# - Load base Gemma model
# - Apply LoRA fine-tuning
# - Save fine-tuned model to models/fine_tuned_gemma/
# 
# Note: Requires GPU with sufficient VRAM
# Training time depends on dataset size
```

### 9. Index Content for Context Search

```bash
# Build vector database
python context_manager.py

# This creates:
# - Embeddings for all transcript segments
# - ChromaDB index for semantic search
# - Enables context-aware responses
```

### 10. Start the API Server

```bash
# Run the FastAPI server
python api.py

# Server will be available at:
# - Web UI: http://localhost:8000
# - API docs: http://localhost:8000/docs
# - Health check: http://localhost:8000/health
```

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Build and start
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

### Manual Docker Build

```bash
# Build image
docker build -t video-chatbot .

# Run container
docker run -d \
  --name video-chatbot \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  --env-file .env \
  video-chatbot
```

## 📁 Project Structure

```
transcript/
├── api.py                      # FastAPI server
├── vllm_server.py             # VLLM model server
├── context_manager.py         # Vector DB and context search
├── transcribe.py              # Video transcription
├── prepare_training_data.py   # Training data preparation
├── train_model.py             # Model fine-tuning
├── download_model.py          # Download Gemma from Hugging Face
├── config.py                  # Configuration management
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose setup
├── .env.example              # Environment template
├── README.md                 # This file
├── static/                   # Frontend files
│   ├── index.html           # Web UI
│   ├── styles.css           # Styling
│   └── script.js            # Client-side logic
└── data/                    # Data directories
    ├── videos/              # Input videos
    ├── transcripts/         # Generated transcripts
    ├── training/            # Training datasets
    ├── vectordb/            # ChromaDB storage
    └── models/              # Fine-tuned models
```

## 🔧 API Endpoints

### POST /chat
Send a message to the chatbot.

**Request:**
```json
{
  "message": "What topics are covered in the videos?",
  "temperature": 0.7,
  "use_context": true
}
```

**Response:**
```json
{
  "response": "Based on the videos, the main topics include...",
  "context_used": true,
  "sources": [
    {
      "video": "video_name",
      "timestamp": "10.50s - 15.30s",
      "similarity": 0.85
    }
  ],
  "timestamp": "2024-01-01T12:00:00"
}
```

### GET /health
Check system health.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "context_manager_ready": true
}
```

### GET /stats
Get system statistics.

**Response:**
```json
{
  "transcripts": 5,
  "training_examples": 1250,
  "model_status": "loaded",
  "context_db_status": "ready"
}
```

## 🎯 How It Works

### 1. Transcription Pipeline
- Videos are processed using OpenAI Whisper
- Word-level timestamps are preserved
- Tone and speaking style information is captured
- Transcripts are saved in JSON format with metadata

### 2. Training Data Generation
- Transcripts are converted into conversation pairs
- Sequential segments create natural dialogue flow
- Context windows are created for topic-based Q&A
- Data is formatted for Gemma's instruction format

### 3. Model Fine-tuning
- Base Gemma model is loaded with 4-bit quantization
- LoRA (Low-Rank Adaptation) is applied for efficient training
- Model learns the tone, style, and content from videos
- Fine-tuned weights are saved for inference

### 4. Context-Aware Inference
- User queries are embedded using sentence transformers
- ChromaDB performs semantic similarity search
- Relevant transcript segments are retrieved
- If similarity is below threshold, query is rejected
- Context is injected into the prompt for accurate responses

### 5. VLLM Serving
- Fine-tuned model is loaded with VLLM
- GPU memory is optimized for maximum throughput
- Batched inference for multiple concurrent requests
- Streaming support for real-time responses

## ⚙️ Configuration Options

### Model Settings
- `MODEL_NAME`: Base Gemma model (default: google/gemma-2b-it)
- `VLLM_GPU_MEMORY_UTILIZATION`: GPU memory usage (0.0-1.0)
- `VLLM_MAX_MODEL_LEN`: Maximum sequence length

### Transcription Settings
- `WHISPER_MODEL`: Whisper model size (tiny/base/small/medium/large)
- `VIDEO_DIR`: Directory containing video files
- `TRANSCRIPT_DIR`: Output directory for transcripts

### Context Settings
- `SIMILARITY_THRESHOLD`: Minimum similarity for relevant context (0.0-1.0)
- `MAX_CONTEXT_LENGTH`: Maximum context characters to include

### API Settings
- `API_HOST`: Server host (default: 0.0.0.0)
- `API_PORT`: Server port (default: 8000)
- `CORS_ORIGINS`: Allowed CORS origins

## 🔍 Troubleshooting

### Out of Memory Errors
- Reduce `VLLM_GPU_MEMORY_UTILIZATION`
- Use smaller Whisper model
- Reduce `VLLM_MAX_MODEL_LEN`
- Use gradient checkpointing during training

### Model Not Responding Correctly
- Check if training data was generated correctly
- Verify context manager has indexed transcripts
- Adjust `SIMILARITY_THRESHOLD` (lower = more permissive)
- Review training logs for convergence

### Transcription Failures
- Ensure FFmpeg is installed
- Check video file formats are supported
- Verify sufficient disk space
- Try smaller Whisper model first

### Context Not Found
- Run `python context_manager.py` to rebuild index
- Check if transcripts exist in `data/transcripts/`
- Verify ChromaDB path is writable

## 📊 Performance Tips

1. **GPU Optimization**
   - Use VLLM for 2-3x faster inference vs standard transformers
   - Enable tensor parallelism for multi-GPU setups
   - Adjust batch size based on available VRAM

2. **Training Efficiency**
   - Use 4-bit quantization (already configured)
   - LoRA reduces trainable parameters by 99%
   - Gradient accumulation for larger effective batch sizes

3. **Context Search**
   - Pre-compute embeddings during indexing
   - Use smaller embedding models for faster search
   - Cache frequent queries

## 🛡️ Security Considerations

- Never commit `.env` file with API keys
- Use environment variables for sensitive data
- Implement rate limiting for production
- Add authentication for public deployments
- Validate and sanitize user inputs

## 📝 License

This project is provided as-is for educational and commercial use.

## 🤝 Contributing

Contributions are welcome! Please ensure:
- Code follows existing style
- Tests pass
- Documentation is updated
- Commits are descriptive

## 📧 Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs in `data/` directories
3. Consult API documentation at `/docs`

## 🎓 Advanced Usage

### Custom System Prompts
Edit the system message in `vllm_server.py` to customize bot behavior.

### Multi-GPU Training
Set `CUDA_VISIBLE_DEVICES` to use specific GPUs:
```bash
CUDA_VISIBLE_DEVICES=0,1 python train_model.py
```

### Batch Transcription
Process multiple video directories:
```python
from transcribe import VideoTranscriber
transcriber = VideoTranscriber()
transcriber.transcribe_all_videos("path/to/videos")
```

### Custom Embedding Models
Change the embedding model in `context_manager.py`:
```python
self.embedding_model = SentenceTransformer('your-model-name')
```

## 🚀 Roadmap

- [ ] Streaming responses
- [ ] Multi-language support
- [ ] Voice input/output
- [ ] Conversation history
- [ ] User authentication
- [ ] Model versioning
- [ ] A/B testing framework
- [ ] Analytics dashboard

---

Built with ❤️ using Gemma, VLLM, FastAPI, and Whisper
