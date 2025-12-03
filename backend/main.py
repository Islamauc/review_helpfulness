"""
FastAPI Application
Main API server for review helpfulness prediction.
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import time
import os
from pathlib import Path
import uuid
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from services.feature_service import FeatureService
from services.model_service import ModelService
from services.online_learning import OnlineLearningService
from database.database import get_db, init_db
from database.models import Review, User, Product, Feedback
from sqlalchemy.orm import Session
from sqlalchemy import func

try:
    from services.scraping_service import ScrapingService
    scraping_service = ScrapingService()
    SCRAPING_AVAILABLE = scraping_service.available
except ImportError:
    SCRAPING_AVAILABLE = False
    scraping_service = None
    print("Warning: Scraping service not available.")

try:
    from services.retraining_scheduler import start_scheduler_in_background
    SCHEDULER_AVAILABLE = True
except ImportError:
    SCHEDULER_AVAILABLE = False
    print("Warning: Scheduler not available. Install 'schedule' package for periodic retraining.")

app = FastAPI(
    title="Review Helpfulness Predictor API",
    description="API for predicting software review helpfulness",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For local development - restrict in production if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = Path("models")
MODEL_VERSION = "v1.0.0"

try:
    feature_service = FeatureService(
        scaler_path=str(MODELS_DIR / f"scaler_{MODEL_VERSION}.pkl"),
        imputer_path=str(MODELS_DIR / f"imputer_{MODEL_VERSION}.pkl"),
        metadata_path=str(MODELS_DIR / f"metadata_{MODEL_VERSION}.json")
    )
    model_service = ModelService(
        model_path=str(MODELS_DIR / f"model_{MODEL_VERSION}.pkl"),
        metadata_path=str(MODELS_DIR / f"metadata_{MODEL_VERSION}.json")
    )
except Exception as e:
    print(f"Warning: Could not load model artifacts: {e}")
    print("Please train the model first using train_model.py")
    feature_service = None
    model_service = None

init_db()

scheduler = None
if SCHEDULER_AVAILABLE:
    try:
        RETRAIN_INTERVAL_HOURS = int(os.getenv("RETRAIN_INTERVAL_HOURS", "24"))
        scheduler = start_scheduler_in_background(
            models_dir=str(MODELS_DIR),
            interval_hours=RETRAIN_INTERVAL_HOURS
        )
        print(f"Periodic retraining scheduler started (interval: {RETRAIN_INTERVAL_HOURS} hours)")
    except Exception as e:
        print(f"Warning: Could not start retraining scheduler: {e}")
        scheduler = None


class ReviewInput(BaseModel):
    review_title: str = Field(..., max_length=200, description="Review title")
    review_text: str = Field(..., max_length=5000, description="Review text content")
    rating: int = Field(..., ge=1, le=5, description="Star rating (1-5)")
    verified_purchase: bool = Field(default=False, description="Verified purchase flag")
    product_id: str = Field(..., description="Product identifier")
    user_id: str = Field(..., description="User identifier")
    review_image_count: int = Field(default=0, ge=0, description="Number of images in review")


class PredictionOutput(BaseModel):
    prediction: str
    probability: float
    confidence: str
    processing_time_ms: float
    model_version: str
    suggestions: Optional[List[str]] = None
    review_id: Optional[str] = None


class BatchReviewInput(BaseModel):
    reviews: List[ReviewInput] = Field(..., max_length=1000)


class BatchPredictionOutput(BaseModel):
    predictions: List[dict]
    total_processed: int
    processing_time_ms: float


class FeedbackInput(BaseModel):
    review_id: str
    actual_label: str = Field(..., description="Actual helpfulness: 'helpful' or 'unhelpful'")
    user_id: str


class HealthOutput(BaseModel):
    status: str
    model_loaded: bool
    model_version: Optional[str] = None
    uptime_seconds: Optional[float] = None


class ScrapeProductInput(BaseModel):
    url: str = Field(..., description="Amazon product URL")
    product_id: Optional[str] = Field(None, description="Optional product ID (uses ASIN if not provided)")


class ScrapeProductOutput(BaseModel):
    success: bool
    product_id: Optional[str] = None
    product_data: Optional[dict] = None
    message: str


class ScrapeReviewsInput(BaseModel):
    url: str = Field(..., description="Amazon product URL")
    max_reviews: int = Field(10, ge=1, le=50, description="Maximum number of reviews to scrape")


class ScrapeReviewsOutput(BaseModel):
    success: bool
    reviews: List[dict] = []
    total_scraped: int
    product_asin: Optional[str] = None
    message: str


def get_user_data(db: Session, user_id: str) -> dict:
    """Get user data from database."""
    user = db.query(User).filter(User.user_id == user_id).first()
    if user:
        return {
            'review_count': user.review_count,
            'avg_helpful_votes': user.avg_helpful_votes
        }
    return {'review_count': 0, 'avg_helpful_votes': 0.0}


def get_product_data(db: Session, product_id: str) -> dict:
    """Get product data from database."""
    product = db.query(Product).filter(Product.product_id == product_id).first()
    if product:
        return {
            'reviews_number': product.reviews_number,
            'price': product.price or 0.0,
            'specs_chars': product.specs_chars,
            'average_rating': product.average_rating or 0.0,
            'store_name': product.store_name or 'Other',
            'title': product.title or 'Other',
            'category': product.category or 'Other'
        }
    return {
        'reviews_number': 0,
        'price': 0.0,
        'specs_chars': 0,
        'average_rating': 0.0,
        'store_name': 'Other',
        'title': 'Other',
        'category': 'Other'
    }


def apply_content_quality_adjustments(
    result: dict, 
    content_quality: dict, 
    rating: int, 
    review_length: int
) -> dict:
    """
    Apply post-processing adjustments based on content quality.
    This helps catch edge cases the model might miss.
    """
    prediction = result['prediction']
    probability = result['probability']
    
    if review_length < 100 and rating <= 2:
        if prediction == 'helpful':
            probability = min(probability, 0.3)  # Cap at 30%
            if probability < 0.5:
                prediction = 'unhelpful'
    
    has_error = content_quality.get('has_error_admission', 0)
    has_non_usage = content_quality.get('has_non_usage_admission', 0)
    has_future_update = content_quality.get('has_future_update_promise', 0)
    has_packaging_focus = content_quality.get('has_packaging_focus', 0)
    has_delivery_focus = content_quality.get('has_delivery_focus', 0)
    has_self_blame_listing = content_quality.get('has_self_blame_listing', 0)
    has_question_only = content_quality.get('has_question_only_focus', 0)
    has_repetition_issue = content_quality.get('has_repetition_issue', 0)
    has_caps_shouting = content_quality.get('has_caps_shouting', 0)
    uppercase_ratio_content = content_quality.get('uppercase_ratio_content', 0.0)
    
    if has_error:
        if prediction == 'helpful':
            probability = max(0.0, probability - 0.5)  # Reduce by 50%
            if probability < 0.5:
                prediction = 'unhelpful'
        else:
            probability = min(1.0, probability + 0.2)
    
    if has_non_usage:
        prediction = 'unhelpful'
        probability = max(probability, 0.8)
    
    if has_future_update and not has_non_usage:
        if prediction == 'helpful':
            probability = max(0.0, probability - 0.3)
            if probability < 0.5:
                prediction = 'unhelpful'
        else:
            probability = min(1.0, probability + 0.1)
        if review_length < 200:
            prediction = 'unhelpful'
            probability = max(probability, 0.7)
    
    if has_self_blame_listing and not content_quality['has_specific_features']:
        prediction = 'unhelpful'
        probability = max(probability, 0.75)
    
    if has_question_only:
        prediction = 'unhelpful'
        probability = max(probability, 0.85)
    
    if content_quality['vague_phrase_count'] >= 2:
        if prediction == 'helpful':
            probability = max(0.0, probability - 0.4)  # Reduce by 40%
            if probability < 0.5:
                prediction = 'unhelpful'
    
    if content_quality['vague_phrase_count'] >= 2 and rating <= 2:
        prediction = 'unhelpful'
        probability = max(probability, 0.75)
    
    if review_length < 50:
        prediction = 'unhelpful'
        probability = max(probability, 0.7)
    
    if rating <= 2 and content_quality['has_error_admission'] and not content_quality['has_specific_features']:
        prediction = 'unhelpful'
        probability = max(probability, 0.8)
    
    if rating <= 2 and content_quality['vague_phrase_count'] >= 2 and review_length < 200:
        prediction = 'unhelpful'
        probability = max(probability, 0.8)
    
    if (has_packaging_focus or has_delivery_focus) and not content_quality['has_specific_features']:
        prediction = 'unhelpful'
        probability = max(probability, 0.75)
    
    if content_quality['vague_phrase_count'] >= 3 and not content_quality['has_specific_features']:
        prediction = 'unhelpful'
        probability = max(probability, 0.75)
    
    # Rule 6d: Repetitive filler content is unhelpful
    if has_repetition_issue and not content_quality['has_specific_features']:
        prediction = 'unhelpful'
        probability = max(probability, 0.8)
    
    if has_caps_shouting or uppercase_ratio_content > 0.65:
        probability = max(0.0, probability - 0.4)
            if prediction == 'helpful' and probability < 0.5:
                prediction = 'unhelpful'
    
    if rating >= 4 and has_error and content_quality['vague_phrase_count'] >= 2:
        prediction = 'unhelpful'
        probability = max(probability, 0.85)
    elif rating >= 4 and has_error:
        probability = max(0.0, probability - 0.3)
        if probability < 0.5:
            prediction = 'unhelpful'
    
    if probability > 0.8 or probability < 0.2:
        confidence = "high"
    elif probability > 0.6 or probability < 0.4:
        confidence = "medium"
    else:
        confidence = "low"
    
    return {
        'prediction': prediction,
        'probability': probability,
        'confidence': confidence,
        'model_version': result['model_version'],
        'adjusted': True  # Flag to indicate post-processing was applied
    }


def generate_suggestions(prediction: str, probability: float, review_data: dict) -> List[str]:
    """Generate suggestions to improve review helpfulness."""
    suggestions = []
    
    if prediction == "unhelpful" or probability < 0.5:
        if review_data.get('review_text', ''):
            text_len = len(review_data['review_text'])
            if text_len < 100:
                suggestions.append("Consider writing a longer, more detailed review")
            elif text_len < 300:
                suggestions.append("Add more specific examples and details")
        
        if review_data.get('rating') in [1, 5]:
            suggestions.append("Explain your rating with specific pros and cons")
        
        if not review_data.get('verified_purchase'):
            suggestions.append("Verified purchases tend to be more helpful")
    
    if prediction == "helpful" and probability > 0.7:
        suggestions.append("Your review looks great! It's likely to be helpful to others.")
    
    return suggestions if suggestions else ["Your review is well-written!"]


# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint."""
    return {
        "message": "Review Helpfulness Predictor API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.post("/api/v1/predict", response_model=PredictionOutput)
async def predict_helpfulness(
    review: ReviewInput,
    db: Session = Depends(get_db)
):
    """Predict review helpfulness."""
    if not feature_service or not model_service:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    start_time = time.time()
    
    try:
        user_data = get_user_data(db, review.user_id)
        product_data = get_product_data(db, review.product_id)
        
        review_data = {
            'review_title': review.review_title,
            'review_text': review.review_text,
            'rating': review.rating,
            'verified_purchase': review.verified_purchase,
            'review_image_count': review.review_image_count,
            'review_date': datetime.now()
        }
        
        features = feature_service.extract_and_preprocess(
            review_data, user_data, product_data
        )
        
        result = model_service.predict(features)
        
        content_quality = feature_service.extract_content_quality_features(
            review.review_text, review.review_title
        )
        
        adjusted_result = apply_content_quality_adjustments(
            result, content_quality, review.rating, len(review.review_text)
        )
        
        suggestions = generate_suggestions(
            adjusted_result['prediction'],
            adjusted_result['probability'],
            review_data
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        review_id = str(uuid.uuid4())
        try:
            review_record = Review(
                review_id=review_id,
                user_id=review.user_id,
                product_id=review.product_id,
                title=review.review_title,
                text=review.review_text,
                rating=review.rating,
                verified=review.verified_purchase,
                prediction_score=adjusted_result['probability'],
                prediction_label=adjusted_result['prediction']
            )
            db.add(review_record)
            db.commit()
        except Exception as e:
            print(f"Warning: Could not save review to database: {e}")
        
        return PredictionOutput(
            prediction=adjusted_result['prediction'],
            probability=adjusted_result['probability'],
            confidence=adjusted_result['confidence'],
            processing_time_ms=processing_time,
            model_version=adjusted_result['model_version'],
            suggestions=suggestions,
            review_id=review_id
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/predict/batch", response_model=BatchPredictionOutput)
async def predict_batch(
    batch_input: BatchReviewInput,
    db: Session = Depends(get_db)
):
    """Batch prediction endpoint."""
    if not feature_service or not model_service:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    start_time = time.time()
    predictions = []
    
    try:
        for review_input in batch_input.reviews:
            try:
                user_data = get_user_data(db, review_input.user_id)
                product_data = get_product_data(db, review_input.product_id)
                
                review_data = {
                    'review_title': review_input.review_title,
                    'review_text': review_input.review_text,
                    'rating': review_input.rating,
                    'verified_purchase': review_input.verified_purchase,
                    'review_image_count': review_input.review_image_count,
                    'review_date': datetime.now()
                }
                
                features = feature_service.extract_and_preprocess(
                    review_data, user_data, product_data
                )
                
                result = model_service.predict(features)
                predictions.append({
                    'review_id': str(uuid.uuid4()),
                    'prediction': result['prediction'],
                    'probability': result['probability'],
                    'confidence': result['confidence']
                })
            except Exception as e:
                predictions.append({
                    'error': str(e)
                })
        
        processing_time = (time.time() - start_time) * 1000
        
        return BatchPredictionOutput(
            predictions=predictions,
            total_processed=len(predictions),
            processing_time_ms=processing_time
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/feedback")
async def submit_feedback(
    feedback: FeedbackInput,
    db: Session = Depends(get_db)
):
    """Submit user feedback on prediction for online learning."""
    try:
        review = db.query(Review).filter(Review.review_id == feedback.review_id).first()
        if not review:
            raise HTTPException(status_code=404, detail="Review not found")
        
        if feedback.actual_label not in ['helpful', 'unhelpful']:
            raise HTTPException(
                status_code=400,
                detail="actual_label must be 'helpful' or 'unhelpful'"
            )
        
        predicted_label = review.prediction_label or 'unhelpful'
        is_correct = (predicted_label == feedback.actual_label)
        
        feedback_record = Feedback(
            feedback_id=str(uuid.uuid4()),
            review_id=feedback.review_id,
            user_id=feedback.user_id,
            predicted_label=predicted_label,
            actual_label=feedback.actual_label,
            is_correct=is_correct
        )
        
        db.add(feedback_record)
        db.commit()
        
        return {
            "message": "Feedback submitted successfully",
            "feedback_id": feedback_record.feedback_id,
            "is_correct": is_correct
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/health", response_model=HealthOutput)
async def health_check():
    """Health check endpoint."""
    import time
    start_time = getattr(health_check, 'start_time', time.time())
    uptime = time.time() - start_time
    
    return HealthOutput(
        status="healthy" if model_service and model_service.is_loaded() else "degraded",
        model_loaded=model_service is not None and model_service.is_loaded(),
        model_version=MODEL_VERSION if model_service else None,
        uptime_seconds=uptime
    )


@app.get("/api/v1/metrics")
async def get_metrics(db: Session = Depends(get_db)):
    """Get model performance metrics (admin endpoint)."""
    try:
        total_predictions = db.query(Review).count()
        total_feedbacks = db.query(Feedback).count()
        correct_predictions = db.query(Feedback).filter(Feedback.is_correct == True).count()
        
        accuracy = (correct_predictions / total_feedbacks * 100) if total_feedbacks > 0 else 0
        
        return {
            "total_predictions": total_predictions,
            "total_feedbacks": total_feedbacks,
            "correct_predictions": correct_predictions,
            "accuracy_percentage": round(accuracy, 2),
            "model_version": MODEL_VERSION
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/retrain")
async def retrain_model(
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Trigger model retraining with user feedback (admin endpoint)."""
    try:
        online_learning = OnlineLearningService(models_dir=str(MODELS_DIR))
        
        def run_retraining():
            result = online_learning.retrain_model(db)
            print(f"Retraining completed: {result}")
        
        background_tasks.add_task(run_retraining)
        
        return {
            "message": "Model retraining started in background",
            "status": "pending"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/scrape-product", response_model=ScrapeProductOutput)
async def scrape_product(
    input_data: ScrapeProductInput,
    db: Session = Depends(get_db)
):
    """Scrape product information from Amazon URL."""
    if not SCRAPING_AVAILABLE or not scraping_service:
        raise HTTPException(
            status_code=503,
            detail="Scraping service not available. Make sure scrape_amazon.py exists and dependencies are installed."
        )
    
    try:
        product_data = scraping_service.scrape_product(input_data.url)
        
        if product_data is None:
            raise HTTPException(
                status_code=400,
                detail="Failed to scrape product. Check if URL is valid and accessible."
            )
        
        product_id = input_data.product_id or product_data.get("product_asin")
        if not product_id:
            raise HTTPException(
                status_code=400,
                detail="Could not determine product ID from URL. Please provide product_id."
            )
        
        saved_product = scraping_service.create_or_update_product(
            db, product_data, product_id
        )
        
        return ScrapeProductOutput(
            success=True,
            product_id=saved_product.get("product_id"),
            product_data=saved_product,
            message="Product scraped and saved successfully"
        )
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraping error: {str(e)}")


@app.post("/api/v1/scrape-reviews", response_model=ScrapeReviewsOutput)
async def scrape_reviews(
    input_data: ScrapeReviewsInput
):
    """Scrape reviews from an Amazon product URL."""
    if not SCRAPING_AVAILABLE or not scraping_service:
        raise HTTPException(
            status_code=503,
            detail="Scraping service not available. Make sure scrape_amazon.py exists and dependencies are installed."
        )
    
    try:
        reviews = scraping_service.scrape_reviews(input_data.url, input_data.max_reviews)
        
        if reviews is None or len(reviews) == 0:
            return ScrapeReviewsOutput(
                success=False,
                reviews=[],
                total_scraped=0,
                product_asin=None,
                message="No reviews found or failed to scrape reviews. Reviews may be dynamically loaded."
            )
        
        import re
        asin_match = re.search(r"/dp/([A-Z0-9]{10})", input_data.url)
        if not asin_match:
            asin_match = re.search(r"/gp/product/([A-Z0-9]{10})", input_data.url)
        asin = asin_match.group(1) if asin_match else None
        
        return ScrapeReviewsOutput(
            success=True,
            reviews=reviews,
            total_scraped=len(reviews),
            product_asin=asin,
            message=f"Successfully scraped {len(reviews)} reviews"
        )
    
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scraping error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

