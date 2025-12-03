#!/bin/bash
# Start server script with error handling

echo "Starting Review Helpfulness Predictor API Server..."
echo ""

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Virtual environment activated"
fi

# Check if dependencies are installed
echo "Checking dependencies..."
python3 -c "import fastapi, uvicorn" 2>/dev/null && echo "FastAPI and Uvicorn installed" || echo "Missing FastAPI/Uvicorn - run: pip install fastapi uvicorn"
python3 -c "from bs4 import BeautifulSoup" 2>/dev/null && echo "BeautifulSoup4 installed" || echo "Missing BeautifulSoup4 - run: pip install beautifulsoup4 lxml"

echo ""
echo "Starting server..."
echo "Server will be available at: http://localhost:8000"
echo "API docs will be available at: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python3 main.py

