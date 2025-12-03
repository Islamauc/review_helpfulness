#!/bin/bash
# Script to run the application locally with SQLite

cd "$(dirname "$0")"

# Unset DATABASE_URL to ensure SQLite is used
unset DATABASE_URL

# Check if port 8000 is in use and kill it
PORT=8000
PID=$(lsof -ti:$PORT 2>/dev/null)
if [ ! -z "$PID" ]; then
    echo "⚠️  Port $PORT is already in use (PID: $PID)"
    echo "🛑 Stopping existing process..."
    kill $PID 2>/dev/null
    sleep 1
    # Check if it's still running
    if lsof -ti:$PORT >/dev/null 2>&1; then
        echo "⚠️  Process still running, force killing..."
        kill -9 $PID 2>/dev/null
        sleep 1
    fi
    echo "✅ Port $PORT is now free"
    echo ""
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "🚀 Starting application with local SQLite database..."
echo ""

# Verify database configuration
python3 << 'PYTHON_SCRIPT'
import os
os.environ.pop('DATABASE_URL', None)

from database.database import DATABASE_URL
print(f"📊 Database: {DATABASE_URL}")
if not DATABASE_URL.startswith("sqlite"):
    print("⚠️  WARNING: Not using SQLite! This might cause errors.")
    print("   Make sure DATABASE_URL is not set in your environment.")
else:
    print("✅ Using local SQLite database")
print("")
PYTHON_SCRIPT

# Run the application
echo "Starting FastAPI server on http://localhost:8000"
echo ""
python3 main.py

