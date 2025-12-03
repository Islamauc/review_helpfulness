"""
Online Learning Service
Handles retraining the model with user feedback data.
"""

import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
from sqlalchemy.orm import Session
from sqlalchemy import func

from database.models import Review, Feedback, User, Product
from database.dataset_table import DatasetRow
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import json


class OnlineLearningService:
    """Service for online learning with user feedback."""
    
    def __init__(self, models_dir: str = "models", min_feedback_threshold: int = 100):
        """
        Initialize online learning service.
        
        Args:
            models_dir: Directory containing model artifacts
            min_feedback_threshold: Minimum number of feedback samples before retraining
        """
        self.models_dir = Path(models_dir)
        self.min_feedback_threshold = min_feedback_threshold
        self.current_version = "v1.0.0"
    
    def collect_feedback_data(self, db: Session, days_back: int = 30) -> pd.DataFrame:
        """
        Collect feedback data from database.
        
        Args:
            db: Database session
            days_back: Number of days to look back for feedback
        
        Returns:
            DataFrame with feedback data
        """
        cutoff_date = datetime.now() - timedelta(days=days_back)
        
        feedbacks = db.query(Feedback, Review).join(
            Review, Feedback.review_id == Review.review_id
        ).filter(
            Feedback.created_at >= cutoff_date
        ).all()
        
        if len(feedbacks) < self.min_feedback_threshold:
            return pd.DataFrame()
        
        data = []
        for feedback, review in feedbacks:
            user = db.query(User).filter(User.user_id == review.user_id).first()
            product = db.query(Product).filter(Product.product_id == review.product_id).first()
            
            if not user or not product:
                continue
            
            row = {
                'review_title': review.title or '',
                'review_text': review.text or '',
                'rating': review.rating or 3,
                'verified_purchase': review.verified or False,
                'review_image_count': 0,  # TODO: Store image count in review model
                'user_id': review.user_id,
                'product_id': review.product_id,
                'user_review_count': user.review_count or 0,
                'user_avg_helpful_votes': user.avg_helpful_votes or 0.0,
                'product_reviews_number': product.reviews_number or 0,
                'product_price': product.price or 0.0,
                'product_specs_chars': product.specs_chars or 0,
                'product_average_rating': product.average_rating or 0.0,
                'product_store_name': product.store_name or 'Other',
                'product_title': product.title or 'Other',
                'product_category': product.category or 'Other',
                'helpful_votes': 1 if feedback.actual_label == 'helpful' else 0,
                'review_date': review.review_date or datetime.now()
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def load_original_data_from_db(self, db: Session, sample_ratio: float = 0.1) -> pd.DataFrame:
        """
        Load original training data from database table.
        
        Args:
            db: Database session
            sample_ratio: Fraction of data to sample (0.1 = 10%)
        
        Returns:
            DataFrame with original training data
        """
        print(f"Loading original data from database (sampling {sample_ratio*100:.1f}%)...")
        
        try:
            total_count = db.query(DatasetRow).count()
            if total_count == 0:
                return pd.DataFrame()
            
            sample_size = int(total_count * sample_ratio)
            query = db.query(DatasetRow).limit(sample_size)
            df = pd.read_sql(query.statement, db.bind)
            
            print(f"   Loaded {len(df):,} rows from database")
            return df
        except Exception as e:
            print(f"   Error loading from database: {e}")
            return pd.DataFrame()
    
    def prepare_training_data(self, feedback_df: pd.DataFrame, original_data_path: str = None, db: Session = None) -> tuple:
        """
        Prepare training data by combining original data with feedback data.
        
        Args:
            feedback_df: DataFrame with feedback data
            original_data_path: Path to original training data CSV (optional if using database)
            db: Database session (optional, for loading from database)
        
        Returns:
            Tuple of (X, y, feature_names)
        """
        original_df = pd.DataFrame()
        if db is not None:
            original_df = self.load_original_data_from_db(db, sample_ratio=0.1)
        
        if len(original_df) == 0 and original_data_path and os.path.exists(original_data_path):
            print(f"Loading original data from CSV: {original_data_path}...")
            original_df = pd.read_csv(original_data_path, low_memory=False, nrows=100000)
        elif len(original_df) == 0:
            print("Warning: No original data found in database or CSV. Using only feedback data.")
        
        # Note: This is a simplified version. In production, you'd need the full feature extraction pipeline
        feedback_processed = feedback_df.copy()
        
        if len(original_df) > 0:
            combined_df = pd.concat([original_df, feedback_processed], ignore_index=True)
        else:
            combined_df = feedback_processed
        
        # Extract features (simplified version - full feature extraction pipeline needed for production)
        exclude_cols = [
            'helpful_votes', 'text', 'title', 'asin', 'parent_asin', 'user_id',
            'clean_text', 'clean_tokens', 'review_date', 'timestamp',
            'images', 'category', 'helpfulness_category', 'days_bin'
        ]
        
        feature_cols = [col for col in combined_df.columns if col not in exclude_cols]
        numeric_features = [col for col in feature_cols 
                          if col in combined_df.select_dtypes(include=[np.number]).columns]
        categorical_features = [col for col in feature_cols if col not in numeric_features]
        
        if categorical_features:
            MAX_CATEGORIES_PER_COL = 20
            for cat_col in categorical_features:
                if cat_col in combined_df.columns:
                    unique_count = combined_df[cat_col].nunique()
                    if unique_count > MAX_CATEGORIES_PER_COL:
                        top_categories = combined_df[cat_col].value_counts().head(MAX_CATEGORIES_PER_COL).index.tolist()
                        mask = ~combined_df[cat_col].isin(top_categories)
                        combined_df.loc[mask, cat_col] = 'Other'
            
            combined_df = pd.get_dummies(combined_df, columns=categorical_features, 
                                       drop_first=True, dtype=np.int8)
            feature_cols = [col for col in combined_df.columns if col not in exclude_cols]
        
        y = (combined_df['helpful_votes'] > 0).astype(int) if 'helpful_votes' in combined_df.columns else None
        X = combined_df[feature_cols].values if feature_cols else None
        
        return X, y, feature_cols
    
    def retrain_model(self, db: Session, original_data_path: str = None) -> Dict:
        """
        Retrain model with new feedback data.
        
        Args:
            db: Database session
            original_data_path: Path to original training data (defaults to ../Software_Cleaned_norm.csv)
        
        Returns:
            Dictionary with retraining results
        """
        if original_data_path is None:
            original_data_path = os.getenv("DATASET_PATH", "../Software_Cleaned_norm.csv")
        
        print("Starting model retraining with feedback data...")
        
        feedback_df = self.collect_feedback_data(db)
        
        if len(feedback_df) < self.min_feedback_threshold:
            return {
                'status': 'insufficient_data',
                'message': f'Need at least {self.min_feedback_threshold} feedback samples, got {len(feedback_df)}',
                'feedback_count': len(feedback_df)
            }
        
        print(f"Collected {len(feedback_df)} feedback samples")
        
        try:
            X, y, feature_names = self.prepare_training_data(feedback_df, original_data_path, db=db)
        except Exception as e:
            return {
                'status': 'error',
                'message': f'Error preparing training data: {str(e)}'
            }
        
        if X is None or y is None:
            return {
                'status': 'error',
                'message': 'Failed to prepare training data'
            }
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_imputed)
        X_test_scaled = scaler.transform(X_test_imputed)
        
        # Training a fresh model from scratch rather than continuing from previous weights
        model = MLPClassifier(
            hidden_layer_sizes=(100,),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate_init=0.001,
            learning_rate='adaptive',
            max_iter=100,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42
        )
        
        print("Training model...")
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        new_version = f"v1.{int(datetime.now().timestamp())}"
        
        joblib.dump(model, self.models_dir / f"model_{new_version}.pkl")
        joblib.dump(scaler, self.models_dir / f"scaler_{new_version}.pkl")
        joblib.dump(imputer, self.models_dir / f"imputer_{new_version}.pkl")
        
        metadata = {
            'version': new_version,
            'feature_names': feature_names,
            'n_features': len(feature_names),
            'metrics': {
                'accuracy': float(accuracy),
                'f1': float(f1),
                'roc_auc': float(roc_auc)
            },
            'feedback_samples': len(feedback_df),
            'training_date': datetime.now().isoformat()
        }
        
        with open(self.models_dir / f"metadata_{new_version}.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Model retrained and saved as {new_version}")
        
        return {
            'status': 'success',
            'new_version': new_version,
            'metrics': {
                'accuracy': float(accuracy),
                'f1': float(f1),
                'roc_auc': float(roc_auc)
            },
            'feedback_samples': len(feedback_df)
        }

