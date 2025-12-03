"""
Model Service
Handles model loading and prediction.
"""

import joblib
import numpy as np
from typing import Dict, Optional
from pathlib import Path
import json


class ModelService:
    """Service for model inference."""
    
    def __init__(self, model_path: str, metadata_path: Optional[str] = None):
        """Initialize model service."""
        self.model = joblib.load(model_path)
        self.model_version = "v1.0.0"
        
        if metadata_path:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
                self.model_version = self.metadata.get('version', 'v1.0.0')
        else:
            self.metadata = {}
    
    def predict(self, features: np.ndarray) -> Dict:
        """
        Run prediction on preprocessed features.
        
        Args:
            features: Preprocessed feature array (1, 64)
        
        Returns:
            Dictionary with prediction results
        """
        if features.shape[0] != 1:
            raise ValueError(f"Expected features shape (1, n), got {features.shape}")
        
        prediction = self.model.predict(features)[0]
        probabilities = self.model.predict_proba(features)[0]
        
        pred_label = "helpful" if prediction == 1 else "unhelpful"
        probability = float(probabilities[1])
        
        if probability > 0.8 or probability < 0.2:
            confidence = "high"
        elif probability > 0.6 or probability < 0.4:
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            "prediction": pred_label,
            "probability": probability,
            "confidence": confidence,
            "model_version": self.model_version,
            "class_0_prob": float(probabilities[0]),
            "class_1_prob": float(probabilities[1])
        }
    
    def predict_batch(self, features: np.ndarray) -> list:
        """
        Run batch predictions.
        
        Args:
            features: Preprocessed feature array (n, 64)
        
        Returns:
            List of prediction dictionaries
        """
        predictions = self.model.predict(features)
        probabilities = self.model.predict_proba(features)
        
        results = []
        for i in range(len(predictions)):
            pred_label = "helpful" if predictions[i] == 1 else "unhelpful"
            prob = float(probabilities[i, 1])
            
            if prob > 0.8 or prob < 0.2:
                confidence = "high"
            elif prob > 0.6 or prob < 0.4:
                confidence = "medium"
            else:
                confidence = "low"
            
            results.append({
                "prediction": pred_label,
                "probability": prob,
                "confidence": confidence,
                "model_version": self.model_version
            })
        
        return results
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

