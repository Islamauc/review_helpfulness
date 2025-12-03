"""
Feature Extraction Service
Extracts and preprocesses features from review data for model prediction.
"""

import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import hashlib
from collections import Counter

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None


class FeatureService:
    """Service for extracting and preprocessing features."""
    
    def __init__(self, scaler_path: str, imputer_path: str, metadata_path: str):
        """Initialize feature service with pre-trained scaler and imputer."""
        self.scaler = joblib.load(scaler_path)
        self.imputer = joblib.load(imputer_path)
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        self.feature_names = self.metadata.get('feature_names', [])
        
        self._build_feature_mapping()
    
    def _build_feature_mapping(self):
        """Build mapping for categorical feature encoding."""
        self.categorical_prefixes = {}
        for feat_name in self.feature_names:
            if '_' in feat_name:
                parts = feat_name.split('_')
                if len(parts) >= 3:
                    prefix = '_'.join(parts[:-1])
                    value = parts[-1]
                    if prefix not in self.categorical_prefixes:
                        self.categorical_prefixes[prefix] = []
                    self.categorical_prefixes[prefix].append(feat_name)
    
    def extract_content_quality_features(self, review_text: str, review_title: str = "") -> Dict:
        """Extract content quality features to detect vague language and user errors."""
        original_text = f"{review_title} {review_text}".strip()
        full_text = original_text.lower()
        full_text = full_text.replace("'", "'").replace("'", "'")
        
        vague_phrases = [
            'kinda', 'kind of', 'sort of', 'maybe', 'probably', 'perhaps',
            "didn't vibe", "wasn't my thing", "not my thing", "not for me",
            'somewhat', 'a bit', 'a little', 'pretty much', 'more or less',
            'i guess', 'i think', 'i suppose', 'seems like', 'seems to',
            'hard to explain', 'could be better', 'could be worse', "don't ask why",
            'whatever'
        ]
        
        non_usage_phrases = [
            "haven't used", "have not used", "haven't really used", "not used yet",
            "haven't tried", "have not tried", "didn't try", "did not try",
            "not tried yet", "haven't tested", "have not tested", "not tested yet",
            "haven't checked", "have not checked", "not checked yet",
            "haven't opened", "have not opened", "only opened", "just opened",
            "still in the box", "still in box", "never used", "not sure if it works",
            "don't know if it works", "haven't even", "not unboxed yet",
            "haven't bothered", "didn't bother using", "forgot why i bought",
            "forgot why i got", "don't remember why i bought", "don't know if this is the one",
            "not sure if this is the one"
        ]
        
        future_update_phrases = [
            "might update later", "will update later", "maybe update", "update later",
            "will update once", "tbd", "probably not", "check back later"
        ]
        
        packaging_focus_phrases = [
            "box was", "packaging was", "package was", "box looked", "box is", "packaging looks"
        ]
        
        delivery_only_phrases = [
            "delivery driver", "shipping was", "arrived late", "arrived early", "delivery guy",
            "courier", "fedex", "ups", "package arrived"
        ]
        
        error_admission_keywords = [
            ("didn't", "read"), ("did", "not", "read"), ("don't", "read"), ("do", "not", "read"),
            ("didn't", "follow"), ("did", "not", "follow"), ("don't", "follow"), ("do", "not", "follow"),
            ("my", "fault"), ("user", "error"), ("my", "mistake"), ("my", "error"),
            ("should", "have"), ("could", "have"), ("would", "have"),
            ("didn't", "understand"), ("did", "not", "understand"), ("don't", "understand"),
            ("misread",), ("misunderstood",),
            ("didn't", "pay", "attention"), ("did", "not", "pay", "attention"),
            ("wasn't", "paying", "attention"), ("was", "not", "paying", "attention"),
            ("not", "paying", "attention")
        ]
        
        self_blame_listing_phrases = [
            "didn't look at the pictures", "did not look at the pictures",
            "didn't look at pictures", "did not look at pictures",
            "didn't read the description", "did not read the description",
            "didn't read the listing", "did not read the listing",
            "should have read the description", "should've read the description",
            "should have checked the pictures", "should've checked the pictures",
            "my bad for not reading", "my fault for not reading",
            "i didn't pay attention to the listing", "i did not pay attention to the listing",
            "i didn't look carefully", "i didn't even look",
            "didn't review the photos", "did not review the photos"
        ]
        
        vague_count = sum(1 for phrase in vague_phrases if phrase in full_text)
        
        non_usage_count = sum(1 for phrase in non_usage_phrases if phrase in full_text)
        future_update_count = sum(1 for phrase in future_update_phrases if phrase in full_text)
        packaging_focus_count = sum(1 for phrase in packaging_focus_phrases if phrase in full_text)
        delivery_focus_count = sum(1 for phrase in delivery_only_phrases if phrase in full_text)
        
        error_count = 0
        words = full_text.split()
        for pattern in error_admission_keywords:
            pattern_words = [w.lower() for w in pattern]
            word_lower = [w.lower() for w in words]
            
            i = 0
            found = False
            for word in word_lower:
                if i < len(pattern_words) and pattern_words[i] in word:
                    i += 1
                    if i == len(pattern_words):
                        found = True
                        break
                elif i > 0:
                    pass
            
            if found:
                error_count += 1
                break
        
        self_blame_listing = any(phrase in full_text for phrase in self_blame_listing_phrases)
        
        words = full_text.split()
        word_count = len(words)
        unique_words = set(words)
        unique_word_ratio = (len(unique_words) / word_count) if word_count else 1.0
        max_word_freq = 0
        if words:
            from collections import Counter
            counts = Counter(words)
            max_word_freq = max(counts.values())
        repetition_issue = word_count >= 6 and (unique_word_ratio < 0.5 or max_word_freq >= max(3, int(word_count * 0.4)))
        
        sentences = [s for s in full_text.replace('?', '.').replace('!', '.').split('.') if s.strip()]
        avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0
        
        question_mark_count = original_text.count('?')
        question_sentences = question_mark_count
        total_sentences = max(len(sentences), 1)
        question_ratio = question_sentences / total_sentences
        question_only = question_ratio >= 0.8 and question_mark_count >= 2 and word_count <= 80
        
        has_numbers = any(char.isdigit() for char in full_text)
        has_specific_features = any(word in full_text for word in ['feature', 'function', 'option', 'setting', 'button', 'menu'])
        
        letters = [c for c in original_text if c.isalpha()]
        uppercase_ratio = (
            sum(1 for c in letters if c.isupper()) / len(letters)
            if letters else 0.0
        )
        caps_shouting = uppercase_ratio > 0.65 and word_count >= 5
        
        return {
            'vague_phrase_count': vague_count,
            'error_admission_count': error_count,
            'has_vague_language': int(vague_count > 0),
            'has_error_admission': int(error_count > 0),
            'non_usage_phrase_count': non_usage_count,
            'has_non_usage_admission': int(non_usage_count > 0),
            'future_update_phrase_count': future_update_count,
            'has_future_update_promise': int(future_update_count > 0),
            'packaging_focus_phrase_count': packaging_focus_count,
            'has_packaging_focus': int(packaging_focus_count > 0),
            'delivery_focus_phrase_count': delivery_focus_count,
            'has_delivery_focus': int(delivery_focus_count > 0),
            'has_self_blame_listing': int(self_blame_listing),
            'question_only_ratio': question_ratio,
            'has_question_only_focus': int(question_only),
            'has_repetition_issue': int(repetition_issue),
            'uppercase_ratio_content': uppercase_ratio,
            'has_caps_shouting': int(caps_shouting),
            'word_count': word_count,
            'avg_sentence_length': avg_sentence_length,
            'has_numbers': int(has_numbers),
            'has_specific_features': int(has_specific_features),
            'content_quality_score': max(
                0,
                1.0
                - (vague_count * 0.2)
                - (error_count * 0.3)
                - (non_usage_count * 0.3)
                - (packaging_focus_count * 0.2)
                - (int(question_only) * 0.2)
            )
        }
    
    def extract_text_features(self, review_text: str, review_title: str = "") -> Dict:
        """Extract text-based features from review."""
        full_text = f"{review_title} {review_text}".strip()
        
        features = {
            'review_len_chars': len(full_text),
            'review_is_long': int(len(full_text) > 500),
            'punct_emph_count': full_text.count('!') + full_text.count('?'),
        }
        
        letters = [c for c in full_text if c.isalpha()]
        features['uppercase_ratio'] = (
            sum(1 for c in letters if c.isupper()) / len(letters) 
            if letters else 0.0
        )
        
        if TextBlob:
            try:
                blob = TextBlob(full_text)
                sentiment_polarity = blob.sentiment.polarity
                sentiment_subjectivity = blob.sentiment.subjectivity
                
                content_quality = self.extract_content_quality_features(review_text, review_title)
                
                if content_quality['has_vague_language']:
                    sentiment_polarity *= 0.5
                if content_quality['has_error_admission']:
                    sentiment_polarity -= 0.3
                    sentiment_polarity = max(-1.0, min(1.0, sentiment_polarity))
                
                features['sentiment_polarity'] = sentiment_polarity
                features['sentiment_subjectivity'] = sentiment_subjectivity
            except:
                features['sentiment_polarity'] = 0.0
                features['sentiment_subjectivity'] = 0.0
        else:
            features['sentiment_polarity'] = 0.0
            features['sentiment_subjectivity'] = 0.0
        
        return features
    
    def extract_features(
        self, 
        review_data: Dict,
        user_data: Optional[Dict] = None,
        product_data: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Extract 64 features from review, user, and product data.
        
        Args:
            review_data: Dictionary with review information
            user_data: Dictionary with user information (optional)
            product_data: Dictionary with product information (optional)
        
        Returns:
            Feature vector as numpy array (1, 64)
        """
        user_data = user_data or {}
        product_data = product_data or {}
        
        review_text = review_data.get('review_text', '')
        review_title = review_data.get('review_title', '')
        text_features = self.extract_text_features(review_text, review_title)
        
        feature_dict = {}
        
        feature_dict['review_len_chars'] = text_features['review_len_chars']
        feature_dict['review_is_long'] = text_features['review_is_long']
        feature_dict['punct_emph_count'] = text_features['punct_emph_count']
        feature_dict['uppercase_ratio'] = text_features['uppercase_ratio']
        feature_dict['sentiment_polarity'] = text_features['sentiment_polarity']
        feature_dict['sentiment_subjectivity'] = text_features['sentiment_subjectivity']
        feature_dict['rating'] = review_data.get('rating', 3)
        feature_dict['verified'] = int(review_data.get('verified_purchase', False))
        feature_dict['review_image_count'] = review_data.get('review_image_count', 0)
        
        feature_dict['user_review_count'] = user_data.get('review_count', 0)
        feature_dict['user_avg_helpful_votes'] = user_data.get('avg_helpful_votes', 0.0)
        
        feature_dict['product_reviews_number'] = product_data.get('reviews_number', 0)
        feature_dict['product_price'] = product_data.get('price', 0.0)
        feature_dict['product_specs_chars'] = product_data.get('specs_chars', 0)
        feature_dict['product_average_rating'] = product_data.get('average_rating', 0.0)
        
        review_date = review_data.get('review_date')
        if review_date:
            try:
                if isinstance(review_date, str):
                    review_dt = datetime.fromisoformat(review_date.replace('Z', '+00:00'))
                else:
                    review_dt = review_date
                days_since = (datetime.now() - review_dt.replace(tzinfo=None)).days
            except:
                days_since = 0
        else:
            days_since = 0
        feature_dict['days_since_review'] = days_since
        
        for key in ['review_len_chars', 'user_review_count', 'user_avg_helpful_votes',
                    'product_reviews_number', 'product_price', 'product_specs_chars',
                    'rating', 'days_since_review']:
            if key in feature_dict:
                feature_dict[f'{key}_z'] = feature_dict[key]
        
        product_store_name = product_data.get('store_name', 'Other')
        product_title = product_data.get('title', 'Other')
        product_category = product_data.get('category', 'Other')
        
        for feat_name in self.feature_names:
            if any(prefix in feat_name for prefix in ['product_store_name_', 'product_title_', 'product_category_']):
                feature_dict[feat_name] = 0
        
        store_feat = f'product_store_name_{product_store_name}'
        if store_feat in self.feature_names:
            feature_dict[store_feat] = 1
        
        title_feat = f'product_title_{product_title[:50]}'  # Truncate long titles
        for feat_name in self.feature_names:
            if feat_name.startswith('product_title_') and product_title in feat_name:
                feature_dict[feat_name] = 1
                break
        
        category_feat = f'product_category_{product_category}'
        if category_feat in self.feature_names:
            feature_dict[category_feat] = 1
        
        feature_vector = []
        for feat_name in self.feature_names:
            if feat_name in feature_dict:
                feature_vector.append(feature_dict[feat_name])
            else:
                feature_vector.append(0.0)
        
        return np.array(feature_vector, dtype=np.float64).reshape(1, -1)
    
    def preprocess(self, features: np.ndarray) -> np.ndarray:
        """Apply imputation and scaling to features."""
        features = self.imputer.transform(features)
        features = self.scaler.transform(features)
        return features
    
    def extract_and_preprocess(
        self,
        review_data: Dict,
        user_data: Optional[Dict] = None,
        product_data: Optional[Dict] = None
    ) -> np.ndarray:
        """Extract features and preprocess in one step."""
        features = self.extract_features(review_data, user_data, product_data)
        features = self.preprocess(features)
        return features

