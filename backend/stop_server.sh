#!/bin/bash
# Script to stop the server running on port 8000

PORT=8000
PID=$(lsof -ti:$PORT 2>/dev/null)

if [ -z "$PID" ]; then
    echo "✅ No process running on port $PORT"
else
    echo "Stopping server on port $PORT (PID: $PID)..."
    kill $PID 2>/dev/null
    sleep 1
    
    # Check if still running
    if lsof -ti:$PORT >/dev/null 2>&1; then
        echo "Process still running, force killing..."
        kill -9 $PID 2>/dev/null
        sleep 1
    fi
    
    if lsof -ti:$PORT >/dev/null 2>&1; then
        echo "Failed to stop server"
    else
        echo "Server stopped successfully"
    fi
fi
