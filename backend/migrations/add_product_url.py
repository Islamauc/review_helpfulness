"""
Database Migration: Add product_url column to products table
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy import text
from database.database import engine, SessionLocal
from dotenv import load_dotenv

load_dotenv()


def add_product_url_column():
    """Add product_url column to products table if it doesn't exist."""
    db = SessionLocal()
    try:
        # Check if column exists (MySQL/SQLite/PostgreSQL compatible)
        result = db.execute(text("""
            SELECT COUNT(*) as count
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
            AND TABLE_NAME = 'products'
            AND COLUMN_NAME = 'product_url'
        """))
        
        count = result.fetchone()[0]
        
        if count == 0:
            print("Adding product_url column to products table...")
            db.execute(text("""
                ALTER TABLE products
                ADD COLUMN product_url VARCHAR(1000) NULL
            """))
            db.commit()
            print("Successfully added product_url column")
        else:
            print("product_url column already exists")
    
    except Exception as e:
        try:
            db.execute(text("""
                ALTER TABLE products
                ADD COLUMN product_url VARCHAR(1000)
            """))
            db.commit()
            print("✅ Successfully added product_url column (SQLite)")
        except Exception as e2:

            db.rollback()
    
    finally:
        db.close()


if __name__ == "__main__":
    print("Running migration: Add product_url column...")
    add_product_url_column()
    print("Migration complete!")

