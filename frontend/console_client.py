#!/usr/bin/env python3
"""
Console Client Application
A command-line interface for the Review Helpfulness Prediction System.
Takes human-readable review input and displays predictions in a user-friendly way.
"""

import requests
import json
import sys
from typing import Optional

# API Configuration
API_BASE_URL = "http://localhost:8000"


def print_header():
    """Print application header."""
    print("\n" + "="*70)
    print("  Review Helpfulness Predictor - Console Client")
    print("  Predict whether your software review will be helpful to others")
    print("="*70 + "\n")


def get_user_input(prompt: str, required: bool = True, input_type: type = str) -> Optional[str]:
    """Get user input with validation."""
    while True:
        try:
            value = input(prompt).strip()
            if not value and required:
                print("  Warning: This field is required. Please try again.")
                continue
            
            if input_type == int:
                value = int(value)
                if value < 1 or value > 5:
                    print("  Warning: Rating must be between 1 and 5.")
                    continue
            elif input_type == bool:
                value = value.lower() in ['y', 'yes', 'true', '1']
            
            return value
        except ValueError:
            print(f"  Warning: Invalid input. Please enter a valid {input_type.__name__}.")
        except KeyboardInterrupt:
            print("\n\n  Goodbye!")
            sys.exit(0)


def collect_review_input() -> dict:
    """Collect review information from user."""
    print("Please enter your review details:\n")
    
    review_title = get_user_input("  Review Title: ", required=True)
    print()
    
    print("  Review Text (press Enter twice when done):")
    review_lines = []
    empty_lines = 0
    while empty_lines < 2:
        line = input()
        if not line.strip():
            empty_lines += 1
        else:
            empty_lines = 0
            review_lines.append(line)
    
    review_text = "\n".join(review_lines)
    if not review_text.strip():
        print("  Warning: Review text cannot be empty.")
        return collect_review_input()
    
    print()
    rating = get_user_input("  Rating (1-5 stars): ", required=True, input_type=int)
    print()
    
    verified = get_user_input("  Verified Purchase? (y/n): ", required=False, input_type=bool)
    print()
    
    product_id = get_user_input("  Product ID (or press Enter for default): ", required=False) or "PROD001"
    user_id = get_user_input("  User ID (or press Enter for default): ", required=False) or "USER001"
    
    return {
        "review_title": review_title,
        "review_text": review_text,
        "rating": rating,
        "verified_purchase": verified,
        "product_id": product_id,
        "user_id": user_id,
        "review_image_count": 0
    }


def display_prediction(prediction: dict):
    """Display prediction results in a user-friendly format."""
    print("\n" + "="*70)
    print("  PREDICTION RESULT")
    print("="*70)
    
    # Prediction label
    if prediction['prediction'] == 'helpful':
        print(f"\n  ✅ Prediction: HELPFUL")
        print(f"     Your review is likely to be helpful to other users!")
    else:
        print(f"\n  ❌ Prediction: UNHELPFUL")
        print(f"     Your review may not be as helpful to other users.")
    
    # Probability
    prob_percent = prediction['probability'] * 100
    print(f"\n  Confidence Score: {prob_percent:.1f}%")
    print(f"  Confidence Level: {prediction['confidence'].upper()}")
    
    # Visual probability bar
    bar_length = 50
    filled = int(prob_percent / 100 * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)
    print(f"  [{bar}] {prob_percent:.1f}%")
    
    # Suggestions
    if prediction.get('suggestions'):
        print(f"\n  Suggestions to improve your review:")
        for i, suggestion in enumerate(prediction['suggestions'], 1):
            print(f"     {i}. {suggestion}")
    
    # Metadata
    print(f"\n  Technical Details:")
    print(f"     • Processing Time: {prediction['processing_time_ms']:.0f}ms")
    print(f"     • Model Version: {prediction['model_version']}")
    
    if prediction.get('review_id'):
        print(f"     • Review ID: {prediction['review_id']}")
    
    print("="*70 + "\n")


def submit_feedback(review_id: str, user_id: str, predicted_label: str) -> bool:
    """Submit user feedback on the prediction."""
    print("  Was this prediction correct?")
    print("     1. Prediction is correct")
    print("     2. Prediction is wrong")
    print("     3. Skip feedback")
    
    choice = get_user_input("\n  Your choice (1-3): ", required=True)
    
    if choice == '3':
        return False
    
    # Determine actual_label based on user's choice
    if choice == '1':  # Prediction is correct
        actual_label = predicted_label
    else:  # Prediction is wrong - flip the label
        actual_label = 'helpful' if predicted_label == 'unhelpful' else 'unhelpful'
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/feedback",
            json={
                "review_id": review_id,
                "actual_label": actual_label,
                "user_id": user_id
            },
            timeout=10
        )
        
        if response.status_code == 200:
            print("\n  Thank you for your feedback! This helps improve our model.")
            print("     Your feedback will be used for online learning and model retraining.")
            return True
        else:
            print(f"\n  Error submitting feedback: {response.json().get('detail', 'Unknown error')}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"\n  Error connecting to API: {e}")
        return False


def check_api_health() -> bool:
    """Check if API is available."""
    try:
        response = requests.get(f"{API_BASE_URL}/api/v1/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            if health.get('model_loaded'):
                print(f"  API is healthy. Model version: {health.get('model_version', 'unknown')}")
                return True
            else:
                print(f"  Warning: API is available but model is not loaded.")
                return False
        else:
            print(f"  Error: API returned status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"  Error: Cannot connect to API at {API_BASE_URL}")
        print(f"     Error: {e}")
        print(f"\n  Make sure the backend server is running:")
        print(f"     cd backend && python main.py")
        return False


def main():
    """Main application loop."""
    print_header()
    
    # Check API health
    print("Checking API connection...")
    if not check_api_health():
        print("\n  Please start the backend server and try again.")
        sys.exit(1)
    
    print()
    
    # Main loop
    while True:
        try:
            # Collect review input
            review_data = collect_review_input()
            
            # Make prediction request
            print("\n  Sending request to API...")
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/v1/predict",
                    json=review_data,
                    timeout=30
                )
                
                if response.status_code == 200:
                    prediction = response.json()
                    display_prediction(prediction)
                    
                    # Ask for feedback
                    if prediction.get('review_id'):
                        submit_feedback(prediction['review_id'], review_data['user_id'], prediction['prediction'])
                else:
                    error_detail = response.json().get('detail', 'Unknown error')
                    print(f"\n  Error: {error_detail}")
            
            except requests.exceptions.RequestException as e:
                print(f"\n  Error connecting to API: {e}")
            
            # Ask if user wants to continue
            print("\n" + "-"*70)
            continue_choice = get_user_input("  Make another prediction? (y/n): ", required=False, input_type=bool)
            if not continue_choice:
                print("\n  Thank you for using Review Helpfulness Predictor!")
                print("="*70 + "\n")
                break
            
            print()
        
        except KeyboardInterrupt:
            print("\n\n  Goodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()

