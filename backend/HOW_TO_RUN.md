# How to Run the Backend

## Quick Start

### Option 1: Using the Script (Recommended)

```bash
cd backend
chmod +x run_local.sh
./run_local.sh
```

This script will:
- ✅ Kill any existing process on port 8000
- ✅ Activate the virtual environment
- ✅ Verify database configuration
- ✅ Start the FastAPI server

### Option 2: Manual Start

```bash
# 1. Navigate to backend directory
cd backend

# 2. Activate virtual environment
source venv/bin/activate

# 3. Start the server
python3 main.py
```

### Option 3: Using uvicorn directly

```bash
cd backend
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

## Verify It's Running

Once started, you should see:
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

Test it:
```bash
curl http://localhost:8000/api/v1/health
```

Or visit in browser:
- API: http://localhost:8000
- Interactive Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/api/v1/health

## Troubleshooting

### Port 8000 already in use
```bash
# Kill the process
lsof -ti:8000 | xargs kill -9

# Or use a different port
uvicorn main:app --port 8001
```

### Virtual environment not found
```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Model not loaded
Make sure model files exist in `backend/models/`:
- `model_v1.0.0.pkl`
- `scaler_v1.0.0.pkl`
- `imputer_v1.0.0.pkl`
- `metadata_v1.0.0.json`

If missing, you may need to train the model first (see training instructions).

## Stop the Server

Press `Ctrl+C` in the terminal where it's running.

Or kill by process:
```bash
lsof -ti:8000 | xargs kill -9
```


