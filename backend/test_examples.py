"""
Test various review examples to verify model performance.
"""

import requests
import json

API_URL = "http://localhost:8000/api/v1/predict"

test_cases = [
    {
        "name": "1. Clearly Unhelpful - Vague + Error Admission",
        "review_title": "Did not read instructions",
        "review_text": "Did not really read the instructions, so maybe that is why it did not work right for me. Kinda confusing. Probably okay for someone else, but I did not vibe with it.",
        "rating": 2,
        "verified_purchase": False,
        "expected": "unhelpful"
    },
    {
        "name": "2. Contradictory - High Rating + Error Admission",
        "review_title": "Good",
        "review_text": "Did not really read the instructions, so maybe that is why it did not work right for me. Kinda confusing. Probably okay for someone else, but I did not vibe with it.",
        "rating": 5,
        "verified_purchase": False,
        "expected": "unhelpful"
    },
    {
        "name": "3. Clearly Helpful - Detailed Review",
        "review_title": "Great software with minor issues",
        "review_text": "I've been using this software for 3 months now. The interface is intuitive and the features are comprehensive. The dashboard provides excellent analytics and the reporting function saves me hours each week. However, it can be slow at times when processing large datasets, and the customer support response time could be better (usually 24-48 hours). Overall, it's a solid product that meets most of my needs. I would recommend it to anyone looking for a reliable solution.",
        "rating": 4,
        "verified_purchase": True,
        "expected": "helpful"
    },
    {
        "name": "4. Too Short - Not Helpful",
        "review_title": "Great!",
        "review_text": "Love it!",
        "rating": 5,
        "verified_purchase": True,
        "expected": "unhelpful"
    },
    {
        "name": "5. Vague Language - Multiple Vague Phrases",
        "review_title": "Okay I guess",
        "review_text": "It's kinda okay. Maybe good for some people. I think it's probably fine. Sort of what I expected. Not really my thing though.",
        "rating": 3,
        "verified_purchase": False,
        "expected": "unhelpful"
    },
    {
        "name": "6. User Error Admission - Low Rating",
        "review_title": "My mistake",
        "review_text": "I didn't follow the instructions properly, so it didn't work for me. My fault really. Should have read the manual first.",
        "rating": 1,
        "verified_purchase": False,
        "expected": "unhelpful"
    },
    {
        "name": "7. Helpful Review - Specific Details",
        "review_title": "Excellent features, minor learning curve",
        "review_text": "This software has transformed my workflow. The automation features save me at least 5 hours per week. The integration with Slack and email is seamless. The mobile app works perfectly on both iOS and Android. The only downside is the initial setup took about 2 hours, but the documentation was clear. Customer support helped me configure the API integration. I've been using it daily for 6 months and it's been very reliable. Highly recommend for teams of 10-50 people.",
        "rating": 5,
        "verified_purchase": True,
        "expected": "helpful"
    },
    {
        "name": "8. Low Rating with Explanation - Could be Helpful",
        "review_title": "Not suitable for my use case",
        "review_text": "I purchased this expecting it to work with my existing database system (PostgreSQL), but it only supports MySQL. The interface is well-designed and the features are comprehensive, but the lack of PostgreSQL support makes it unusable for my needs. If you're using MySQL, this would be an excellent choice. The documentation is clear and the support team was responsive when I contacted them.",
        "rating": 2,
        "verified_purchase": True,
        "expected": "helpful"  # Low rating but detailed and informative
    },
    {
        "name": "9. Very Short with Low Rating",
        "review_title": "Bad",
        "review_text": "Didn't work.",
        "rating": 1,
        "verified_purchase": False,
        "expected": "unhelpful"
    },
    {
        "name": "10. High Rating but Vague",
        "review_title": "It's good",
        "review_text": "Pretty good I think. Probably works well. Seems okay to me. Maybe others will like it more.",
        "rating": 5,
        "verified_purchase": False,
        "expected": "unhelpful"
    },
    {
        "name": "11. Balanced Review - Pros and Cons",
        "review_title": "Good value with some limitations",
        "review_text": "The software does what it promises. The user interface is clean and modern. I particularly like the drag-and-drop feature for organizing tasks. The mobile app syncs perfectly with the desktop version. However, the free version is quite limited - you can only create 10 projects. The paid version at $29/month is reasonable but might be expensive for individual users. The export function works well and supports CSV, PDF, and Excel formats. Overall, I'd recommend it if you need basic project management features.",
        "rating": 4,
        "verified_purchase": True,
        "expected": "helpful"
    },
    {
        "name": "12. Error Admission with High Rating - Contradictory",
        "review_title": "Great product",
        "review_text": "I didn't read the manual and couldn't figure it out at first. My mistake. But once I got help from support, it works great!",
        "rating": 5,
        "verified_purchase": True,
        "expected": "unhelpful"  # Contradictory - admits error but gives 5 stars
    },
    {
        "name": "13. Extremely Short - One Word",
        "review_title": "Ok",
        "review_text": "Fine",
        "rating": 3,
        "verified_purchase": False,
        "expected": "unhelpful"
    },
    {
        "name": "14. Helpful - Mentions Specific Features",
        "review_title": "Powerful automation features",
        "review_text": "The workflow automation feature is excellent. I set up 15 automated tasks that run daily, saving me about 3 hours per week. The email integration works perfectly with Gmail and Outlook. The reporting dashboard shows real-time analytics which is very useful. The mobile app (version 2.3) syncs instantly. Only complaint is the API rate limit of 1000 requests per hour, but that's usually enough for my needs.",
        "rating": 5,
        "verified_purchase": True,
        "expected": "helpful"
    },
    {
        "name": "15. Vague + Short + Low Rating",
        "review_title": "Not great",
        "review_text": "Kinda bad. Didn't really like it. Probably not for me.",
        "rating": 2,
        "verified_purchase": False,
        "expected": "unhelpful"
    },
    {
        "name": "16. Helpful - Comparison Review",
        "review_title": "Better than competitors I've tried",
        "review_text": "I've used similar products from Company A and Company B, and this one is superior in several ways. The user interface is more intuitive than Company A's cluttered design. The pricing at $49/month is more reasonable than Company B's $79/month. The customer support responds within 2 hours compared to 24+ hours with others. The only area where it falls short is the mobile app, which lacks some features available on desktop. Overall, I'd recommend this over the alternatives.",
        "rating": 4,
        "verified_purchase": True,
        "expected": "helpful"
    },
    {
        "name": "17. Spam-like - All Caps",
        "review_title": "AMAZING!!!",
        "review_text": "THIS IS THE BEST PRODUCT EVER!!! BUY IT NOW!!! YOU WON'T REGRET IT!!!",
        "rating": 5,
        "verified_purchase": False,
        "expected": "unhelpful"
    },
    {
        "name": "18. Helpful - Technical Review",
        "review_title": "Solid API and documentation",
        "review_text": "As a developer, I appreciate the well-documented REST API. The authentication uses OAuth 2.0 which is standard. Response times average 200ms for GET requests. The webhook system works reliably. I've integrated it with our Node.js backend successfully. The rate limiting is clearly documented (1000 requests/hour). The only issue I encountered was with the batch upload endpoint timing out on files larger than 10MB, but the support team provided a workaround using chunked uploads.",
        "rating": 4,
        "verified_purchase": True,
        "expected": "helpful"
    },
    {
        "name": "19. Contradictory - Positive Title, Negative Text",
        "review_title": "Excellent software",
        "review_text": "Didn't really understand how to use it. Kinda confusing interface. Probably my fault for not reading the manual. Maybe it works for others.",
        "rating": 5,
        "verified_purchase": False,
        "expected": "unhelpful"
    },
    {
        "name": "20. Helpful - Use Case Specific",
        "review_title": "Perfect for small teams",
        "review_text": "Our team of 8 people has been using this for project management. The task assignment feature works great - you can assign tasks to team members and set deadlines. The file sharing allows up to 100MB per file which is sufficient for our needs. The calendar integration with Google Calendar is seamless. We use the free plan which includes 5 projects - perfect for our small team. The paid plan at $10/user/month is reasonable if you need more features.",
        "rating": 5,
        "verified_purchase": True,
        "expected": "helpful"
    },
    {
        "name": "21. Unhelpful - All Questions",
        "review_title": "Questions",
        "review_text": "Does this work? Is it good? Will I like it? Should I buy it?",
        "rating": 3,
        "verified_purchase": False,
        "expected": "unhelpful"
    },
    {
        "name": "22. Helpful - Before/After Comparison",
        "review_title": "Game changer for my productivity",
        "review_text": "Before using this, I was spending 2 hours daily on manual data entry. Now with the automation features, it takes 15 minutes. The time tracking feature shows I've saved 35 hours this month. The reporting function generates weekly summaries automatically. The integration with my existing tools (Slack, Trello, Google Drive) was straightforward. The learning curve was about 1 week, but the tutorials were helpful. Worth every penny.",
        "rating": 5,
        "verified_purchase": True,
        "expected": "helpful"
    },
    {
        "name": "23. Unhelpful - Repetitive",
        "review_title": "Good good good",
        "review_text": "Good. Good. Good. Good. Good. It's good. Really good. Very good.",
        "rating": 5,
        "verified_purchase": False,
        "expected": "unhelpful"
    },
    {
        "name": "24. Helpful - Specific Problem Solved",
        "review_title": "Solved my inventory management issues",
        "review_text": "I run a small e-commerce store and was struggling with inventory tracking across 3 sales channels (Amazon, eBay, Shopify). This software syncs inventory in real-time, preventing overselling. The low stock alerts saved me from running out of popular items twice. The barcode scanning feature works with my existing scanner. Setup took 2 hours including importing my 500+ products. The $29/month plan includes all features I need. Customer support helped me configure the API connections.",
        "rating": 5,
        "verified_purchase": True,
        "expected": "helpful"
    },
    {
        "name": "25. Unhelpful - Just Emojis",
        "review_title": "👍",
        "review_text": "👍👍👍👍👍",
        "rating": 5,
        "verified_purchase": False,
        "expected": "unhelpful"
    },
    {
        "name": "26. Helpful - Detailed Pros and Cons",
        "review_title": "Good value with room for improvement",
        "review_text": "Pros: Clean interface, fast performance, good mobile app, reliable sync, excellent customer support (responded in 1 hour), reasonable pricing at $19/month, comprehensive feature set including reporting and analytics. Cons: Limited customization options, no dark mode, mobile app missing some advanced features available on desktop, export function only supports CSV and PDF (no Excel), maximum file size of 50MB. Overall, the pros outweigh the cons for my use case. I'd recommend it for small to medium businesses.",
        "rating": 4,
        "verified_purchase": True,
        "expected": "helpful"
    },
    {
        "name": "27. Unhelpful - Copy-Paste Style",
        "review_title": "Review",
        "review_text": "This is a review. I am reviewing this product. This is my review of the product. I am writing a review.",
        "rating": 3,
        "verified_purchase": False,
        "expected": "unhelpful"
    },
    {
        "name": "28. Helpful - Version-Specific Feedback",
        "review_title": "Version 3.0 is a major improvement",
        "review_text": "I've been using this since version 1.5. Version 3.0 released last month added several features I requested: dark mode, keyboard shortcuts, and improved search. The performance improvements are noticeable - pages load 40% faster. The new collaboration features allow real-time editing which my team loves. The bug that caused crashes when exporting large files (reported in v2.8) has been fixed. The migration from v2.8 to v3.0 was smooth with no data loss. Highly recommend upgrading if you're on an older version.",
        "rating": 5,
        "verified_purchase": True,
        "expected": "helpful"
    },
    {
        "name": "29. Unhelpful - No Actual Content",
        "review_title": ".",
        "review_text": "...",
        "rating": 3,
        "verified_purchase": False,
        "expected": "unhelpful"
    },
    {
        "name": "30. Helpful - Timeline Review",
        "review_title": "Gets better with use",
        "review_text": "Week 1: Found the interface confusing, spent 3 hours learning basics. Week 2: Started to appreciate the features, created my first automated workflow. Week 3: Fully integrated into my daily routine, saving 2 hours per day. Month 2: Discovered advanced features like API integration and custom reports. Month 3: Convinced my team to adopt it, now 5 of us use it daily. The initial learning curve is worth it for the long-term productivity gains.",
        "rating": 5,
        "verified_purchase": True,
        "expected": "helpful"
    }
]

def test_review(test_case):
    """Test a single review case."""
    payload = {
        "review_title": test_case["review_title"],
        "review_text": test_case["review_text"],
        "rating": test_case["rating"],
        "verified_purchase": test_case["verified_purchase"],
        "product_id": "B09JQMJHXY",  
        "user_id": "USER001",
        "review_image_count": 0
    }
    
    try:
        response = requests.post(API_URL, json=payload, timeout=5)
        response.raise_for_status()
        result = response.json()
        
        prediction = result["prediction"]
        probability = result["probability"]
        confidence = result["confidence"]
        
        is_correct = (prediction == test_case["expected"])
        status = "✅" if is_correct else "❌"
        
        print(f"\n{status} {test_case['name']}")
        print(f"   Expected: {test_case['expected']}")
        print(f"   Got: {prediction} ({probability*100:.1f}% probability, {confidence} confidence)")
        if not is_correct:
            print(f"    MISMATCH!")
        
        return is_correct
        
    except Exception as e:
        print(f"\n{test_case['name']}")
        print(f"   Error: {e}")
        return False

def main():
    print("="*80)
    print("TESTING REVIEW MODEL WITH VARIOUS EXAMPLES")
    print("="*80)
    
    results = []
    for test_case in test_cases:
        result = test_review(test_case)
        results.append(result)
    
    print("Summary")
    correct = sum(results)
    total = len(results)
    accuracy = (correct / total) * 100
    
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {accuracy:.1f}%")

if __name__ == "__main__":
    main()

