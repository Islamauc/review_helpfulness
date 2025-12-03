"""
Database Models
SQLAlchemy models for the application.
"""

from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime

Base = declarative_base()


class User(Base):
    """User model."""
    __tablename__ = "users"
    
    user_id = Column(String(50), primary_key=True)
    username = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    review_count = Column(Integer, default=0)
    avg_helpful_votes = Column(Float, default=0.0)
    
    reviews = relationship("Review", back_populates="user")


class Product(Base):
    """Product model."""
    __tablename__ = "products"
    
    product_id = Column(String(50), primary_key=True)
    title = Column(String(500), nullable=False)
    store_name = Column(String(200))
    price = Column(Float)
    reviews_number = Column(Integer, default=0)
    average_rating = Column(Float, default=0.0)
    specs_chars = Column(Integer, default=0)
    category = Column(String(100))
    created_at = Column(DateTime, default=datetime.utcnow)
    
    reviews = relationship("Review", back_populates="product")


class Review(Base):
    """Review model."""
    __tablename__ = "reviews"
    
    review_id = Column(String(50), primary_key=True)
    user_id = Column(String(50), ForeignKey("users.user_id"))
    product_id = Column(String(50), ForeignKey("products.product_id"))
    title = Column(String(200))
    text = Column(Text)
    rating = Column(Integer)
    verified = Column(Boolean, default=False)
    helpful_votes = Column(Integer, default=0)
    prediction_score = Column(Float)
    prediction_label = Column(String(20))
    review_date = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    user = relationship("User", back_populates="reviews")
    product = relationship("Product", back_populates="reviews")
    feedbacks = relationship("Feedback", back_populates="review")


class Feedback(Base):
    """User feedback on predictions for online learning."""
    __tablename__ = "feedbacks"
    
    feedback_id = Column(String(50), primary_key=True)
    review_id = Column(String(50), ForeignKey("reviews.review_id"))
    user_id = Column(String(50), ForeignKey("users.user_id"))
    predicted_label = Column(String(20))
    actual_label = Column(String(20))  # helpful or unhelpful
    is_correct = Column(Boolean)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    review = relationship("Review", back_populates="feedbacks")


class TrainingJob(Base):
    """Model training job tracking."""
    __tablename__ = "training_jobs"
    
    job_id = Column(String(50), primary_key=True)
    status = Column(String(20))  # pending, running, completed, failed
    model_version = Column(String(20))
    metrics = Column(Text)  # JSON string
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)

