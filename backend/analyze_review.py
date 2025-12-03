"""
Script to analyze why a specific review was predicted as helpful/unhelpful.
"""

import sys
import json
import numpy as np
from pathlib import Path
from services.feature_service import FeatureService
from services.model_service import ModelService

MODELS_DIR = Path("models")
MODEL_VERSION = "v1.0.0"

feature_service = FeatureService(
    scaler_path=str(MODELS_DIR / f"scaler_{MODEL_VERSION}.pkl"),
    imputer_path=str(MODELS_DIR / f"imputer_{MODEL_VERSION}.pkl"),
    metadata_path=str(MODELS_DIR / f"metadata_{MODEL_VERSION}.json")
)

model_service = ModelService(
    model_path=str(MODELS_DIR / f"model_{MODEL_VERSION}.pkl"),
    metadata_path=str(MODELS_DIR / f"metadata_{MODEL_VERSION}.json")
)

def analyze_review(review_title, review_text, rating, verified_purchase=False, 
                   product_id="PROD001", user_id="USER001"):
    """Analyze a review and show feature values."""
    
    review_data = {
        'review_title': review_title,
        'review_text': review_text,
        'rating': rating,
        'verified_purchase': verified_purchase,
        'review_image_count': 0,
        'review_date': None
    }
    
    user_data = {'review_count': 0, 'avg_helpful_votes': 0.0}
    product_data = {
        'reviews_number': 0,
        'price': 0.0,
        'specs_chars': 0,
        'average_rating': 0.0,
        'store_name': 'Other',
        'title': 'Other',
        'category': 'Other'
    }
    
    features = feature_service.extract_features(review_data, user_data, product_data)
    features_preprocessed = feature_service.preprocess(features)
    
    result = model_service.predict(features_preprocessed)
    
    feature_names = feature_service.feature_names
    
    feature_dict = {}
    for i, name in enumerate(feature_names):
        feature_dict[name] = float(features[0, i])
    
    text_features = feature_service.extract_text_features(review_text, review_title)
    
    print("="*80)
    print("REVIEW ANALYSIS")
    print("="*80)
    print(f"\nReview Title: {review_title}")
    print(f"Review Text: {review_text}")
    print(f"Rating: {rating}")
    print(f"Verified Purchase: {verified_purchase}")
    print(f"\nPrediction: {result['prediction']}")
    print(f"Probability: {result['probability']:.4f} ({result['probability']*100:.2f}%)")
    print(f"Confidence: {result['confidence']}")
    
    print("\n" + "="*80)
    print("TEXT FEATURES")
    print("="*80)
    print(f"Review Length (chars): {text_features['review_len_chars']}")
    print(f"Is Long (>500 chars): {text_features['review_is_long']}")
    print(f"Punctuation Emphasis Count: {text_features['punct_emph_count']}")
    print(f"Uppercase Ratio: {text_features['uppercase_ratio']:.4f}")
    print(f"Sentiment Polarity: {text_features['sentiment_polarity']:.4f}")
    print(f"Sentiment Subjectivity: {text_features['sentiment_subjectivity']:.4f}")
    
    print("\n" + "="*80)
    print("KEY FEATURES (Non-Zero or Significant)")
    print("="*80)
    
    # Show important features
    important_features = [
        'review_len_chars', 'review_len_chars_z',
        'review_is_long',
        'punct_emph_count', 'punct_emph_ratio_z',
        'uppercase_ratio', 'uppercase_ratio_z',
        'sentiment_polarity',
        'sentiment_subjectivity',
        'rating', 'rating_z',
        'verified',
        'review_image_count'
    ]
    
    for feat in important_features:
        if feat in feature_dict:
            print(f"{feat:30s}: {feature_dict[feat]:.6f}")
    
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    # Analyze why it might be predicted as helpful
    issues = []
    
    if text_features['review_len_chars'] < 100:
        issues.append(f"❌ Very short review ({text_features['review_len_chars']} chars) - helpful reviews are typically longer")
    
    if text_features['review_is_long'] == 0:
        issues.append(f"⚠️  Review is not long (under 500 chars) - longer reviews tend to be more helpful")
    
    if text_features['punct_emph_count'] == 0:
        issues.append("⚠️  No punctuation emphasis (! or ?) - can indicate lack of engagement")
    
    if text_features['sentiment_polarity'] > -0.1 and text_features['sentiment_polarity'] < 0.1:
        issues.append(f"⚠️  Neutral sentiment ({text_features['sentiment_polarity']:.3f}) - helpful reviews often have stronger sentiment")
    
    if rating <= 2:
        issues.append(f"❌ Low rating ({rating}) - low ratings without detailed explanation are often unhelpful")
    
    if not verified_purchase:
        issues.append("⚠️  Not a verified purchase - verified purchases tend to be more helpful")
    
    if issues:
        print("\nPotential Issues:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ No obvious issues detected in feature extraction")
    
    print("\n" + "="*80)
    print("WHY IT MIGHT BE PREDICTED AS HELPFUL")
    print("="*80)
    
    # Check what might be pushing it toward helpful
    helpful_indicators = []
    
    if text_features['sentiment_polarity'] > 0:
        helpful_indicators.append(f"Positive sentiment ({text_features['sentiment_polarity']:.3f})")
    
    if text_features['review_len_chars'] > 50:
        helpful_indicators.append(f"Not extremely short ({text_features['review_len_chars']} chars)")
    
    if rating >= 3:
        helpful_indicators.append(f"Moderate to high rating ({rating})")
    
    if helpful_indicators:
        print("\nPossible helpful indicators:")
        for indicator in helpful_indicators:
            print(f"  • {indicator}")
    else:
        print("\nNo strong helpful indicators found")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS TO FIX THE MODEL")
    print("="*80)
    print("""
1. ADD MORE TEXT QUALITY FEATURES:
   - Word count (not just character count)
   - Specificity score (mentions specific features/use cases)
   - Readability score
   - Presence of concrete examples
   - Ratio of subjective vs objective language

2. IMPROVE SENTIMENT ANALYSIS:
   - Current sentiment might be too simplistic
   - Consider detecting vague language ("kinda", "maybe", "probably")
   - Detect user error admissions (should be negative signal)

3. ADD CONTENT QUALITY FEATURES:
   - Check for vague phrases ("didn't vibe", "kinda confusing")
   - Detect lack of specific details
   - Identify user error admissions
   - Measure information density

4. RETRAIN WITH MORE FEEDBACK:
   - Collect more user feedback on edge cases
   - The feedback you just submitted will help
   - Need at least 100 feedback samples for retraining

5. ADJUST FEATURE WEIGHTS:
   - The model might be over-weighting sentiment
   - Consider reducing weight on sentiment for short reviews
   - Increase weight on review length and specificity

6. ADD RULE-BASED FILTERS:
   - Very short reviews (< 100 chars) with low ratings should be flagged
   - Reviews admitting user error should be penalized
   - Vague language should reduce helpfulness score
    """)
    
    return result, feature_dict, text_features

if __name__ == "__main__":
    review_title = "Did not read instructions"
    review_text = "Did not really read the instructions, so maybe that is why it did not work right for me. Kinda confusing. Probably okay for someone else, but I did not vibe with it."
    rating = 2
    verified_purchase = False
    
    analyze_review(review_title, review_text, rating, verified_purchase)

