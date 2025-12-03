#!/bin/bash
# Fix virtual environment issue
# The venv is pointing to a wrong Python path

cd "$(dirname "$0")"

echo "🔧 Fixing virtual environment..."
echo ""

# Check if venv exists
if [ -d "venv" ]; then
    echo "⚠️  Existing venv found but pointing to wrong location"
    echo "🗑️  Removing old venv..."
    rm -rf venv
fi

echo "📦 Creating new virtual environment..."
python3 -m venv venv

echo ""
echo "✅ Virtual environment created!"
echo ""
echo "📝 Next steps:"
echo "   1. source venv/bin/activate"
echo "   2. pip install -r requirements.txt"
echo "   3. pip install pymysql"
echo ""

