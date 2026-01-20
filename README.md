# AI Sentiment Analyzer 🎭

A production-ready sentiment analysis application that leverages deep learning to classify text as positive or negative. Built with a Bidirectional LSTM neural network trained on 50,000 IMDB movie reviews, this tool delivers accurate real-time sentiment predictions through both a modern web interface and REST API.

## ✨ Features

- **Advanced Deep Learning**: Bidirectional LSTM architecture with word embeddings for contextual understanding
- **Real-Time Analysis**: Instant sentiment predictions with confidence scores
- **Dual Interface**: Interactive web UI and comprehensive REST API
- **Robust Preprocessing**: NLTK-powered text normalization and cleaning
- **Production Ready**: FastAPI backend with automatic documentation and health checks
- **Responsive Design**: Mobile-friendly interface with live character counting

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation & Setup

**1. Clone and navigate to the project:**
```bash
git clone <repository-url>
cd Sentiment-Analyser
```

**2. Install dependencies:**
```bash
cd backend
pip install -r requirements.txt
```

*Alternatively, if you're not in the backend directory:*
```bash
pip install -r backend/requirements.txt
```

**3. Download required NLTK data:**
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

### Running the Application

**Using startup scripts:**

Windows:
```bash
start.bat
```

Linux/Mac:
```bash
chmod +x start.sh
./start.sh
```

**Manual start:**
```bash
cd backend
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

**Access the application:**
- Web Interface: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## 📁 Project Structure

```
Sentiment-Analyser/
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI application & endpoints
│   │   ├── services/
│   │   │   └── inference.py        # Model inference logic
│   │   └── utils/
│   │       └── preprocessing.py    # Text preprocessing utilities
│   └── requirements.txt
├── frontend/
│   ├── index.html                  # Web interface
│   ├── script.js                   # Frontend JavaScript
│   └── styles.css                  # Styling
├── models/
│   ├── sentiment_model.keras       # Trained model weights
│   └── tokenizer.json              # Vocabulary tokenizer
├── src/
│   ├── train.py                    # Model training pipeline
│   ├── preprocessing.py            # Training data preprocessing
│   └── sample_predict.py           # Test predictions
├── data/
│   └── IMDB Dataset.csv           # Training dataset
├── notebook/
│   └── code.ipynb                 # Exploratory analysis
├── start.bat                       # Windows launcher
└── start.sh                        # Unix/Mac launcher
```

## 🧠 Model Architecture

The sentiment classifier employs a sophisticated neural network designed for sequential text processing:

```
Input Text (max 300 tokens)
    ↓
Embedding Layer (vocab: 16,000 | dim: 128)
    ↓
Bidirectional LSTM (64 units, dropout: 0.3, L2 reg)
    ↓
Dropout (0.3)
    ↓
Bidirectional LSTM (32 units, dropout: 0.3, L2 reg)
    ↓
Dense Layer (sigmoid activation)
    ↓
Binary Classification (Positive/Negative)
```

**Training Configuration:**
- **Dataset**: IMDB Movie Reviews (50,000 samples)
- **Split**: 80% training, 20% validation
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy
- **Regularization**: L2 + Dropout (prevents overfitting)
- **Early Stopping**: Monitors validation performance

## 🔌 API Reference

### Sentiment Prediction

**Endpoint**: `POST /predict`

**Request:**
```json
{
  "sentiment": "This movie was absolutely fantastic! The acting was superb."
}
```

**Response:**
```json
{
  "Predicted Sentiment": "Positive",
  "probability": 0.9654,
  "confidence": 96.54
}
```

### Health Check

**Endpoint**: `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

**Interactive API Documentation**: Available at `/docs` with Swagger UI for testing endpoints directly in your browser.

## 💻 Development Guide

### Training a Custom Model

Retrain the model with your own dataset or adjust hyperparameters:

```bash
cd src
python train.py
```

**Output:**
- `models/sentiment_model.keras` - Trained model
- `models/tokenizer.json` - Fitted tokenizer
- Training metrics and performance graphs

### Testing Predictions

Validate model performance on sample texts:

```bash
cd src
python sample_predict.py
```

### Modifying the Model

Key files to edit:
- `src/train.py` - Architecture, hyperparameters, training config
- `backend/app/services/inference.py` - Prediction pipeline
- `backend/app/utils/preprocessing.py` - Text preprocessing logic

## 🛠️ Technology Stack

**Backend:**
- FastAPI - High-performance async web framework
- TensorFlow/Keras - Deep learning model training and inference
- NLTK - Natural language text preprocessing
- Pandas - Data manipulation and analysis
- Scikit-learn - Train/test splitting and metrics
- Uvicorn - ASGI server for production deployment

**Frontend:**
- HTML5/CSS3 - Modern responsive design
- Vanilla JavaScript - No framework dependencies
- Fetch API - Async communication with backend

**Machine Learning:**
- Bidirectional LSTM - Captures context from both directions
- Word Embeddings - Dense semantic representations
- Tokenization - Converts text to numerical sequences
- Regularization Techniques - L2 weight decay and dropout

## 📊 Use Cases

The model performs best on opinion-based text similar to its training data:

**Ideal Applications:**
- Movie and book reviews
- Product feedback analysis
- Social media sentiment monitoring
- Customer service ticket classification
- Brand perception tracking
- Survey response analysis

**Example Inputs:**

Positive:
> "Outstanding service! The team went above and beyond to help me. Highly recommend!"

Negative:
> "Terrible experience. The product arrived damaged and customer support was unresponsive."

## 🎯 Model Performance

Trained on 50,000 balanced IMDB reviews, the model achieves strong binary classification performance. Evaluation metrics generated during training include:

- Accuracy on validation set
- Precision and recall scores
- Confusion matrix visualization
- ROC curve and AUC score
- Training/validation loss curves

View detailed performance metrics in the training logs or Jupyter notebook.

## 🔒 Security & Production Notes

**CORS Configuration:**
The API currently allows all origins for development. For production deployment, update CORS settings in `backend/app/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Restrict to your domain
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
```

**Recommendations:**
- Use environment variables for sensitive configuration
- Implement rate limiting for public APIs
- Add authentication for production deployments
- Monitor API usage and model performance
- Set up logging and error tracking

## 🐛 Troubleshooting

**Model files not found:**
```bash
# Ensure models directory contains required files
cd src
python train.py  # Retrain if necessary
```

**NLTK data missing:**
```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```

**Port already in use:**
```bash
# Change port number
uvicorn app.main:app --reload --port 8001
```

**Import errors:**
```bash
# Reinstall dependencies
pip install -r requirements.txt --upgrade
```

## 📦 Dependencies

Core packages (see `requirements.txt` for complete list):

```
fastapi==0.104.1
tensorflow==2.15.0
uvicorn==0.24.0
pandas==2.1.3
nltk==3.8.1
scikit-learn==1.3.2
```

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

**Areas for contribution:**
- Additional preprocessing techniques
- Model architecture improvements
- Multi-class sentiment analysis (positive/neutral/negative)
- Support for other languages
- Performance optimizations
- Enhanced frontend features

## 📄 License

This project is open source and available under the MIT License. Free for educational and commercial use.

## 📧 Support

For questions, bug reports, or feature requests:
- Open an issue on GitHub
- Check existing issues for solutions
- Review the API documentation at `/docs`

---

**Built with ❤️ using TensorFlow, FastAPI, and modern web technologies**

*Star ⭐ this repository if you find it helpful!*