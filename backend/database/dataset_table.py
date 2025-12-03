"""
Dataset Table Model
Stores the training dataset in database instead of CSV file.
"""

from sqlalchemy import Column, Integer, Float, String, Boolean, Text, DateTime, Index
from datetime import datetime

from .models import Base


class DatasetRow(Base):
    """Model for storing dataset rows in database."""
    __tablename__ = "dataset_rows"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    review_title = Column(String(500))
    review_text = Column(Text)
    rating = Column(Integer)
    verified = Column(Boolean, default=False)
    review_image_count = Column(Integer, default=0)
    review_len_chars = Column(Integer)
    review_is_long = Column(Boolean, default=False)
    punct_emph_count = Column(Integer, default=0)
    uppercase_ratio = Column(Float, default=0.0)
    sentiment_polarity = Column(Float, default=0.0)
    sentiment_subjectivity = Column(Float, default=0.0)
    
    user_id = Column(String(50))
    user_review_count = Column(Integer, default=0)
    user_avg_helpful_votes = Column(Float, default=0.0)
    
    product_id = Column(String(50))
    product_reviews_number = Column(Integer, default=0)
    product_price = Column(Float, default=0.0)
    product_specs_chars = Column(Integer, default=0)
    product_average_rating = Column(Float, default=0.0)
    product_store_name = Column(String(200))
    product_title = Column(String(500))
    product_category = Column(String(100))
    
    helpful_votes = Column(Integer, default=0)
    helpful = Column(Boolean, default=False)
    
    review_date = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)

