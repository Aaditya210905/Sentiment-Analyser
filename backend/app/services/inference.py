import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from ..utils.preprocessing import preprocess_text

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
MODEL_PATH = os.path.join(PROJECT_ROOT, 'models', 'sentiment_model.keras')
TOKENIZER_PATH = os.path.join(PROJECT_ROOT, 'models', 'tokenizer.json')

# Global variables
model = None
tokenizer = None

def load_resources():
    """Load model and tokenizer on startup"""
    global model, tokenizer
    
    try:
        # Check if files exist
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        if not os.path.exists(TOKENIZER_PATH):
            raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")
        
        # Load model
        print(f"Loading model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        print("Model loaded successfully")
        
        # Load tokenizer
        print(f"Loading tokenizer from {TOKENIZER_PATH}...")
        with open(TOKENIZER_PATH, 'r') as f:
            tokenizer_json = json.load(f)
        tokenizer = tokenizer_from_json(tokenizer_json)
        print("Tokenizer loaded successfully")
        
        return True
    except Exception as e:
        print(f"Error loading resources: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def predict_sentiment(text: str) -> dict:
    """Predict sentiment for input text"""
    global model, tokenizer
    
    try:
        # Ensure resources are loaded
        if model is None or tokenizer is None:
            if not load_resources():
                raise RuntimeError("Failed to load model resources")
        
        # Preprocess text
        cleaned_text = preprocess_text(text)
        
        # Tokenize and pad
        sequence = tokenizer.texts_to_sequences([cleaned_text])
        padded_sequence = pad_sequences(sequence, maxlen=300, padding='post', truncating='post')
        
        # Predict
        prediction = model.predict(padded_sequence, verbose=0)[0][0]
        
        # Format response
        sentiment = "Positive" if prediction >= 0.5 else "Negative"
        confidence = float(prediction * 100 if prediction >= 0.5 else (1 - prediction) * 100)
        
        return {
            "Predicted Sentiment": sentiment,
            "probability": float(prediction),
            "confidence": round(confidence, 2)
        }
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise
