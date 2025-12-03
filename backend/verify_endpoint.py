#!/usr/bin/env python3
"""
Verify that the scraping endpoint is properly defined in main.py
"""

import ast
import sys
from pathlib import Path

def check_endpoint_registration():
    """Check if the scrape-product endpoint is defined."""
    main_py = Path(__file__).parent / "main.py"
    
    if not main_py.exists():
        print("❌ main.py not found")
        return False
    
    print("Checking main.py for endpoint registration...")
    
    with open(main_py, 'r') as f:
        content = f.read()
    
    # Check for endpoint definition
    if '@app.post("/api/v1/scrape-product"' in content:
        print("✅ Endpoint definition found: @app.post('/api/v1/scrape-product')")
    else:
        print("❌ Endpoint definition NOT found")
        return False
    
    # Check for ScrapeProductInput
    if 'class ScrapeProductInput' in content:
        print("✅ ScrapeProductInput model found")
    else:
        print("❌ ScrapeProductInput model NOT found")
        return False
    
    # Check for ScrapeProductOutput
    if 'class ScrapeProductOutput' in content:
        print("✅ ScrapeProductOutput model found")
    else:
        print("❌ ScrapeProductOutput model NOT found")
        return False
    
    # Check for ScrapingService import
    if 'from services.scraping_service import ScrapingService' in content:
        print("✅ ScrapingService import found")
    else:
        print("❌ ScrapingService import NOT found")
        return False
    
    # Check for scraping_service initialization
    if 'scraping_service = ScrapingService()' in content:
        print("✅ ScrapingService initialization found")
    else:
        print("❌ ScrapingService initialization NOT found")
        return False
    
    # Try to parse the file for syntax errors
    try:
        with open(main_py, 'r') as f:
            ast.parse(f.read())
        print("✅ No syntax errors in main.py")
    except SyntaxError as e:
        print(f"❌ Syntax error in main.py: {e}")
        return False
    
    print("\n" + "="*60)
    print("✅ All checks passed! The endpoint is properly defined.")
    print("="*60)
    print("\n⚠️  IMPORTANT: You must RESTART the server for the endpoint to be registered!")
    print("\nTo restart:")
    print("  1. Stop the current server (Ctrl+C)")
    print("  2. Run: python main.py")
    print("  3. Check http://localhost:8000/docs for the endpoint")
    
    return True

if __name__ == "__main__":
    check_endpoint_registration()

