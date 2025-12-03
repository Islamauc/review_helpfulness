"""
Model Training Script
Trains the final Neural Network model and serializes it for deployment.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, classification_report, confusion_matrix
)
import json
import os
import sys
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

def load_and_prepare_data(csv_path: str, target_col: str = 'helpful_votes'):
    """Load and prepare data for training."""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Loaded {len(df)} samples")
    
    # Define columns to exclude
    exclude_cols = [
        target_col, 'text', 'title', 'asin', 'parent_asin', 'user_id',
        'clean_text', 'clean_tokens', 'review_date', 'timestamp',
        'images', 'category', 'helpfulness_category', 'days_bin'
    ]
    
    # Get feature columns
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    numeric_features = [col for col in feature_cols if col in df.select_dtypes(include=[np.number]).columns]
    categorical_features = [col for col in feature_cols if col not in numeric_features]
    
    print(f"Total features: {len(feature_cols)}")
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")
    
    # Handle categorical features
    if categorical_features:
        MAX_CATEGORIES_PER_COL = 20
        for cat_col in categorical_features:
            unique_count = df[cat_col].nunique()
            if unique_count > MAX_CATEGORIES_PER_COL:
                top_categories = df[cat_col].value_counts().head(MAX_CATEGORIES_PER_COL).index.tolist()
                mask = ~df[cat_col].isin(top_categories)
                df.loc[mask, cat_col] = 'Other'
        
        df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True, dtype=np.int8)
        feature_cols = [col for col in df_encoded.columns if col not in exclude_cols]
    else:
        df_encoded = df
    
    print(f"Final feature count: {len(feature_cols)}")
    
    # Prepare target variable
    if target_col in df_encoded.columns:
        y = (df_encoded[target_col] > 0).astype(int)
    else:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    
    # Prepare features
    X = df_encoded[feature_cols].values
    
    # Save feature names for later use
    feature_names = feature_cols
    
    return X, y, feature_names

def train_model(X_train, y_train, X_val, y_val):
    """Train the Neural Network model."""
    print("\nTraining Neural Network model...")
    
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
        random_state=42,
        verbose=True
    )
    
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""
    print("\nEvaluating model...")
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n" + "="*50)
    print("MODEL PERFORMANCE METRICS")
    print("="*50)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1 Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")
    print("="*50)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'roc_auc': float(roc_auc)
    }

def main():
    """Main training pipeline."""
    # Paths - can be overridden by environment variable
    data_path = os.getenv("DATASET_PATH", "../Software_Cleaned_norm.csv")
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Check if dataset exists
    if not os.path.exists(data_path):
        print(f"❌ Error: Dataset file not found at: {data_path}")
        print(f"\nPlease:")
        print(f"  1. Download the dataset from Google Drive")
        print(f"  2. Place it in the project root as 'Software_Cleaned_norm.csv'")
        print(f"  3. Or run: python3 backend/download_dataset.py")
        print(f"\nAlternatively, set DATASET_PATH environment variable:")
        print(f"  export DATASET_PATH=/path/to/your/dataset.csv")
        sys.exit(1)
    
    # Load and prepare data
    X, y, feature_names = load_and_prepare_data(data_path)
    
    # Train-test split
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Further split training data for validation
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
    )
    
    print(f"Training set: {len(X_train_final)} samples")
    print(f"Validation set: {len(X_val)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Preprocessing
    print("\nPreprocessing data...")
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train_final)
    X_val_imputed = imputer.transform(X_val)
    X_test_imputed = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_val_scaled = scaler.transform(X_val_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)
    
    # Train model
    model = train_model(X_train_scaled, y_train_final, X_val_scaled, y_val)
    
    # Evaluate on test set
    metrics = evaluate_model(model, X_test_scaled, y_test)
    
    # Save model artifacts
    print("\nSaving model artifacts...")
    version = "v1.0.0"
    
    joblib.dump(model, models_dir / f"model_{version}.pkl")
    joblib.dump(scaler, models_dir / f"scaler_{version}.pkl")
    joblib.dump(imputer, models_dir / f"imputer_{version}.pkl")
    
    # Save feature names and metadata
    metadata = {
        'version': version,
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'metrics': metrics,
        'model_config': {
            'hidden_layer_sizes': (100,),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.001,
            'learning_rate_init': 0.001,
            'max_iter': 100
        }
    }
    
    with open(models_dir / f"metadata_{version}.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModel artifacts saved to {models_dir}/")
    print(f"  - model_{version}.pkl")
    print(f"  - scaler_{version}.pkl")
    print(f"  - imputer_{version}.pkl")
    print(f"  - metadata_{version}.json")
    
    return model, scaler, imputer, metadata

if __name__ == "__main__":
    main()

