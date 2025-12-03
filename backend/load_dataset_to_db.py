"""
Load CSV dataset into database table.
This allows storing the training dataset in database instead of CSV file.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
import os
from dotenv import load_dotenv
from tqdm import tqdm

# Import dataset model
from database.database import SessionLocal, init_db, engine
from database.dataset_table import DatasetRow
from database.models import Base  # Import Base to ensure tables are registered

load_dotenv()

def create_dataset_table():
    """Create dataset table if it doesn't exist."""
    DatasetRow.__table__.create(bind=engine, checkfirst=True)
    print("✅ Dataset table created/verified")


def load_csv_to_database(csv_path: str, batch_size: int = 10000, limit: int = None):
    """
    Load CSV file into database table.
    
    Args:
        csv_path: Path to CSV file
        batch_size: Number of rows to insert at once
        limit: Maximum number of rows to load (None for all)
    """
    print(f"📥 Loading dataset from CSV to database...")
    print(f"   CSV file: {csv_path}")
    print(f"   Batch size: {batch_size}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Create table
    create_dataset_table()
    
    db = SessionLocal()
    
    try:
        # Check if data already exists
        existing_count = db.query(DatasetRow).count()
        if existing_count > 0:
            response = input(f"⚠️  Database already contains {existing_count} rows. Overwrite? (y/n): ")
            if response.lower() != 'y':
                print("Loading cancelled.")
                return
            else:
                print("🗑️  Clearing existing data...")
                db.query(DatasetRow).delete()
                db.commit()
        
        # Read CSV in chunks
        print("\n📖 Reading CSV file...")
        chunk_iter = pd.read_csv(csv_path, chunksize=batch_size, low_memory=False)
        
        total_rows = 0
        total_loaded = 0
        
        for chunk_num, chunk in enumerate(chunk_iter):
            if limit and total_loaded >= limit:
                break
            
            # Process chunk
            rows_to_insert = []
            
            for idx, row in chunk.iterrows():
                if limit and total_loaded >= limit:
                    break
                
                # Helper function to safely get values with fallback
                def safe_get(col_names, default, dtype=str):
                    """Try multiple column names, return first found or default."""
                    if isinstance(col_names, str):
                        col_names = [col_names]
                    
                    for col_name in col_names:
                        if col_name in row and pd.notna(row[col_name]):
                            val = row[col_name]
                            try:
                                if dtype == int:
                                    return int(val)
                                elif dtype == float:
                                    return float(val)
                                elif dtype == bool:
                                    return bool(val)
                                else:
                                    return str(val)
                            except:
                                continue
                    return default
                
                # Calculate review_len_chars if not in CSV
                review_text = safe_get(['text', 'review_text'], '')
                review_len_chars = safe_get(['review_len_chars'], len(review_text) if review_text else 0, int)
                
                # Map CSV columns to database columns (flexible mapping)
                dataset_row = DatasetRow(
                    # Review features
                    review_title=safe_get(['title', 'review_title'], '')[:500],
                    review_text=review_text,
                    rating=safe_get(['rating'], 3, int),
                    verified=safe_get(['verified', 'verified_purchase'], False, bool),
                    review_image_count=safe_get(['review_image_count', 'images'], 0, int),
                    review_len_chars=review_len_chars,
                    review_is_long=safe_get(['review_is_long'], False, bool),
                    punct_emph_count=safe_get(['punct_emph_count'], 0, int),
                    uppercase_ratio=safe_get(['uppercase_ratio'], 0.0, float),
                    sentiment_polarity=safe_get(['sentiment_polarity'], 0.0, float),
                    sentiment_subjectivity=safe_get(['sentiment_subjectivity'], 0.0, float),
                    
                    # User features
                    user_id=safe_get(['user_id'], '')[:50],
                    user_review_count=safe_get(['user_review_count'], 0, int),
                    user_avg_helpful_votes=safe_get(['user_avg_helpful_votes'], 0.0, float),
                    
                    # Product features
                    product_id=safe_get(['product_id', 'asin', 'parent_asin'], '')[:50],
                    product_reviews_number=safe_get(['product_reviews_number', 'reviews_number'], 0, int),
                    product_price=safe_get(['product_price', 'price'], 0.0, float),
                    product_specs_chars=safe_get(['product_specs_chars', 'specs_chars'], 0, int),
                    product_average_rating=safe_get(['product_average_rating', 'average_rating'], 0.0, float),
                    product_store_name=safe_get(['store_name', 'product_store_name'], 'Other')[:200],
                    product_title=safe_get(['product_title', 'product_name'], 'Other')[:500],
                    product_category=safe_get(['category', 'product_category'], 'Other')[:100],
                    
                    # Target
                    helpful_votes=safe_get(['helpful_votes'], 0, int),
                    helpful=bool(safe_get(['helpful_votes'], 0, int) > 0) or safe_get(['helpful'], False, bool),
                )
                
                rows_to_insert.append(dataset_row)
                total_rows += 1
            
            # Bulk insert
            if rows_to_insert:
                db.bulk_save_objects(rows_to_insert)
                db.commit()
                total_loaded += len(rows_to_insert)
                
                print(f"   Loaded chunk {chunk_num + 1}: {total_loaded:,} rows", end='\r')
        
        print(f"\n✅ Dataset loaded successfully!")
        print(f"   Total rows: {total_loaded:,}")
        
        # Create indexes for better performance
        print("\n📊 Creating indexes...")
        try:
            db.execute(text("CREATE INDEX IF NOT EXISTS idx_helpful ON dataset_rows(helpful)"))
            db.execute(text("CREATE INDEX IF NOT EXISTS idx_user_id ON dataset_rows(user_id)"))
            db.execute(text("CREATE INDEX IF NOT EXISTS idx_product_id ON dataset_rows(product_id)"))
            db.commit()
            print("✅ Indexes created")
        except Exception as e:
            print(f"⚠️  Could not create indexes (may already exist): {e}")
        
    except Exception as e:
        db.rollback()
        print(f"\n❌ Error loading dataset: {e}")
        raise
    finally:
        db.close()


def get_dataset_count():
    """Get count of rows in dataset table."""
    db = SessionLocal()
    try:
        count = db.query(DatasetRow).count()
        return count
    finally:
        db.close()


if __name__ == "__main__":
    import sys
    
    # Get CSV path from command line or environment
    csv_path = sys.argv[1] if len(sys.argv) > 1 else os.getenv("DATASET_PATH", "../Software_Cleaned_norm.csv")
    
    # Optional limit for testing
    limit = None
    if len(sys.argv) > 2:
        limit = int(sys.argv[2])
        print(f"⚠️  Loading limited to {limit:,} rows (for testing)")
    
    print("=" * 60)
    print("Dataset Database Loader")
    print("=" * 60)
    print()
    
    # Initialize database
    init_db()
    
    # Load dataset
    load_csv_to_database(csv_path, batch_size=10000, limit=limit)
    
    # Show statistics
    count = get_dataset_count()
    print(f"\n📊 Database now contains {count:,} dataset rows")
    print("\n✅ Ready for training from database!")

