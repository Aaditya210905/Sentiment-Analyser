import pandas as pd
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import pickle
from src.preprocessing import preprocess_sample 
from tensorflow.keras.models import load_model

# Load the tokenizer, model, and label encoder
with open("models/tokenizer.json", "r", encoding="utf-8") as f:
    tokenizer = tokenizer_from_json(f.read())
model = load_model("models/sentiment_model.keras")
label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))

# Sample text for prediction
sample_texts = pd.Series([
    "Amazing movie."
])

# Preprocess the sample text
sample_text = preprocess_sample(
    sample_texts,
    tokenizer=tokenizer
)

# Make predictions
predicted_prob = model.predict(sample_text)
predicted_class = (predicted_prob > 0.5).astype(int)
sentiment = label_encoder.inverse_transform(predicted_class)
print(f"Predicted Sentiment: {sentiment[0]}\n Probability of Positive Sentiment: {predicted_prob[0][0]:.4f}")