#!/bin/bash
# Script to ensure local SQLite database is used (no MySQL/remote connections)

cd "$(dirname "$0")"

echo "🔧 Ensuring local SQLite database configuration..."
echo ""

# Unset any DATABASE_URL that might be set in the environment
unset DATABASE_URL

# Update .env file to explicitly use SQLite (or not set DATABASE_URL)
cat > .env << 'EOF'
# Local Development Configuration
# Using SQLite for local development (no remote database needed)
# DATABASE_URL is intentionally NOT set, so it defaults to sqlite:///./reviewpro.db

# Path to dataset CSV file
DATASET_PATH=../Software_Cleaned_norm.csv
EOF

echo "✅ .env file updated for local SQLite"
echo ""
echo "📋 Current configuration:"
echo "   Database: SQLite (local)"
echo "   Database file: ./reviewpro.db"
echo "   No remote database needed"
echo ""

# Test the connection
echo "🧪 Testing local SQLite connection..."
source venv/bin/activate

python3 << 'PYTHON_SCRIPT'
import os
# Explicitly unset DATABASE_URL
os.environ.pop('DATABASE_URL', None)

from database.database import DATABASE_URL, engine, SessionLocal

print(f"   DATABASE_URL: {DATABASE_URL}")
print(f"   Engine URL: {engine.url}")

if DATABASE_URL.startswith("sqlite"):
    print("✅ Using local SQLite database!")
    try:
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        print("✅ Connection successful!")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
else:
    print(f"❌ ERROR: Still trying to use {DATABASE_URL}")
    print("   This should be sqlite:///./reviewpro.db")
PYTHON_SCRIPT

echo ""
echo "💡 To use this configuration:"
echo "   1. Make sure to run: source venv/bin/activate"
echo "   2. Unset DATABASE_URL if it's in your shell: unset DATABASE_URL"
echo "   3. Then run your application: python3 main.py"

