import pickle
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model
from pathlib import Path

# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

class SentimentInference:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.label_encoder = None
        self.load_models()
    
    def load_models(self):
        """Load the trained model, tokenizer, and label encoder"""
        models_dir = PROJECT_ROOT / "models"
        
        # Load tokenizer
        with open(models_dir / "tokenizer.json", "r", encoding="utf-8") as f:
            self.tokenizer = tokenizer_from_json(f.read())
        
        # Load model
        self.model = load_model(models_dir / "sentiment_model.keras")
        
        # Load label encoder
        with open(models_dir / "label_encoder.pkl", "rb") as f:
            self.label_encoder = pickle.load(f)
    
    def predict(self, text: str) -> dict:
        """
        Predict sentiment for the given text
        
        Args:
            text: Input text to analyze
            
        Returns:
            Dictionary with sentiment and probability
        """
        import pandas as pd
        import sys
        from pathlib import Path
        
        # Add backend directory to path
        backend_dir = Path(__file__).parent.parent.parent
        if str(backend_dir) not in sys.path:
            sys.path.insert(0, str(backend_dir))
        
        from app.utils.preprocessing import preprocess_sample
        
        # Convert to pandas Series
        text_series = pd.Series([text])
        
        # Preprocess the text
        processed_text = preprocess_sample(
            text_series,
            tokenizer=self.tokenizer
        )
        
        # Make prediction
        predicted_prob = self.model.predict(processed_text, verbose=0)
        predicted_class = (predicted_prob > 0.5).astype(int)
        
        # Get sentiment label
        sentiment = self.label_encoder.inverse_transform(predicted_class)[0]
        probability = float(predicted_prob[0][0])
        
        return {
            "sentiment": sentiment,
            "probability": probability,
            "confidence": probability if sentiment == "positive" else 1 - probability
        }

# Global instance
inference_service = SentimentInference()
