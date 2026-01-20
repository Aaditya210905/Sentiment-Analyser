@echo off
REM Sentiment Analysis App - Start Script (Windows)

echo Starting Sentiment Analysis Application...
echo.

cd /d "%~dp0"

echo.
echo Starting FastAPI server...
echo Server: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
echo Press CTRL+C to stop the server
echo.

REM Start the server
uvicorn backend.app.main:app --reload
