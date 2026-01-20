#!/bin/bash

# Sentiment Analysis App - Start Script

echo "🚀 Starting Sentiment Analysis Application..."
echo ""

# Navigate to project root
cd "$(dirname "$0")"


# Kill any process using port 8000
echo "🔍 Checking for processes on port 8000..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || fuser -k 8000/tcp 2>/dev/null || pkill -f "uvicorn.*8000" 2>/dev/null || true
sleep 1

echo ""
echo "✨ Starting FastAPI server..."
echo "📍 Server: http://localhost:8000"
echo "📚 API Docs: http://localhost:8000/docs"
echo ""
echo "Press CTRL+C to stop the server"
echo ""

# Start the server
uvicorn backend.app.main:app --reload
