import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager
import os

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.inference import predict_sentiment, load_resources

# Lifespan context manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Starting up application...")
    success = load_resources()
    if not success:
        print("WARNING: Failed to load model resources!")
    yield
    # Shutdown (if needed)
    print("Shutting down application...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Sentiment Analysis API",
    description="AI-powered sentiment analysis using deep learning",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://sentiment-analyser-czaw.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

# Request/Response Models
class SentimentRequest(BaseModel):
    sentiment: str

# Routes
@app.get("/")
async def read_root():
    """Serve the frontend or return API info"""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "message": "Sentiment Analysis API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "docs": "/docs"
        }
    }

@app.get("/styles.css")
async def get_styles():
    """Serve the CSS file"""
    css_path = FRONTEND_DIR / "styles.css"
    if css_path.exists():
        return FileResponse(str(css_path), media_type="text/css")
    raise HTTPException(status_code=404, detail="CSS file not found")

@app.get("/script.js")
async def get_script():
    """Serve the JavaScript file"""
    js_path = FRONTEND_DIR / "script.js"
    if js_path.exists():
        return FileResponse(str(js_path), media_type="application/javascript")
    raise HTTPException(status_code=404, detail="JavaScript file not found")

@app.post("/predict")
async def predict(request: SentimentRequest):
    """
    Predict sentiment of the provided text
    
    - **sentiment**: Text to analyze (1-5000 characters)
    
    Returns the predicted sentiment (Positive/Negative) with probability scores
    """
    try:
        if not request.sentiment or not request.sentiment.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        result = predict_sentiment(request.sentiment)
        return result
    except Exception as e:
        print(f"Error in /predict endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    from .services.inference import model, tokenizer
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
