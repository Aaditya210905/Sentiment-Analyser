import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.services.inference import inference_service

# Initialize FastAPI app
app = FastAPI(
    title="Sentiment Analysis API",
    description="AI-powered sentiment analysis using deep learning",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "https://*.onrender.com"  # Allow all Render subdomains
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
    sentiment: str = Field(..., min_length=1, max_length=5000, description="Text to analyze")

class SentimentResponse(BaseModel):
    Predicted_Sentiment: str = Field(..., alias="Predicted Sentiment")
    probability: float = Field(None, description="Probability score")
    confidence: float = Field(None, description="Confidence percentage")

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

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": True}

@app.post("/predict", response_model=SentimentResponse)
async def predict_sentiment(request: SentimentRequest):
    """
    Predict sentiment of the provided text
    
    - **sentiment**: Text to analyze (1-5000 characters)
    
    Returns the predicted sentiment (Positive/Negative) with probability scores
    """
    try:
        result = inference_service.predict(request.sentiment)
        
        return {
            "Predicted Sentiment": result["sentiment"],
            "probability": result["probability"],
            "confidence": result["confidence"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    import os
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
