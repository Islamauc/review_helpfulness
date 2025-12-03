import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

function App() {
  const [formData, setFormData] = useState({
    review_title: '',
    review_text: '',
    rating: 5,
    verified_purchase: false,
    product_id: 'PROD001',
    user_id: 'USER001',
    review_image_count: 0
  });
  
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [feedback, setFeedback] = useState(null);
  const [feedbackSubmitted, setFeedbackSubmitted] = useState(false);

  const handleChange = (e) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);
    setFeedback(null);
    setFeedbackSubmitted(false);

    try {
      const response = await axios.post(`${API_BASE_URL}/api/v1/predict`, formData);
      setPrediction(response.data);
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const handleFeedback = async (actualLabel) => {
    if (!prediction) {
      alert('Please make a prediction first');
      return;
    }
    
    const reviewId = prediction.review_id;
    if (!reviewId) {
      alert('Review ID not available. Please make a new prediction.');
      return;
    }

    try {
      await axios.post(`${API_BASE_URL}/api/v1/feedback`, {
        review_id: reviewId,
        actual_label: actualLabel,
        user_id: formData.user_id
      });
      setFeedbackSubmitted(true);
      setFeedback(actualLabel);
    } catch (err) {
      alert('Error submitting feedback: ' + (err.response?.data?.detail || err.message));
    }
  };

  return (
    <div className="App">
      <div className="container">
        <header className="header">
          <h1> Review Helpfulness Predictor</h1>
          <p>Predict whether your software review will be helpful to others</p>
        </header>

        <div className="content">
          <div className="form-section">
            <h2>Enter Your Review</h2>
            <form onSubmit={handleSubmit}>
              <div className="form-group">
                <label htmlFor="review_title">Review Title *</label>
                <input
                  type="text"
                  id="review_title"
                  name="review_title"
                  value={formData.review_title}
                  onChange={handleChange}
                  required
                  maxLength={200}
                  placeholder="e.g., Great software but slow"
                />
              </div>

              <div className="form-group">
                <label htmlFor="review_text">Review Text *</label>
                <textarea
                  id="review_text"
                  name="review_text"
                  value={formData.review_text}
                  onChange={handleChange}
                  required
                  rows={8}
                  maxLength={5000}
                  placeholder="Write your review here... Be specific about your experience, mention pros and cons, and provide examples."
                />
                <small>{formData.review_text.length} / 5000 characters</small>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="rating">Rating *</label>
                  <select
                    id="rating"
                    name="rating"
                    value={formData.rating}
                    onChange={handleChange}
                    required
                  >
                    <option value={1}>⭐ 1 Star</option>
                    <option value={2}>⭐⭐ 2 Stars</option>
                    <option value={3}>⭐⭐⭐ 3 Stars</option>
                    <option value={4}>⭐⭐⭐⭐ 4 Stars</option>
                    <option value={5}>⭐⭐⭐⭐⭐ 5 Stars</option>
                  </select>
                </div>

                <div className="form-group">
                  <label>
                    <input
                      type="checkbox"
                      name="verified_purchase"
                      checked={formData.verified_purchase}
                      onChange={handleChange}
                    />
                    Verified Purchase
                  </label>
                </div>
              </div>

              <div className="form-row">
                <div className="form-group">
                  <label htmlFor="product_id">Product ID</label>
                  <input
                    type="text"
                    id="product_id"
                    name="product_id"
                    value={formData.product_id}
                    onChange={handleChange}
                    placeholder="PROD001"
                  />
                </div>

                <div className="form-group">
                  <label htmlFor="user_id">User ID</label>
                  <input
                    type="text"
                    id="user_id"
                    name="user_id"
                    value={formData.user_id}
                    onChange={handleChange}
                    placeholder="USER001"
                  />
                </div>
              </div>

              <button type="submit" className="submit-btn" disabled={loading}>
                {loading ? 'Analyzing...' : 'Predict Helpfulness'}
              </button>
            </form>
          </div>

          {error && (
            <div className="error-box">
              <h3>Error</h3>
              <p>{error}</p>
            </div>
          )}

          {prediction && (
            <div className="prediction-section">
              <h2>Prediction Result</h2>
              <div className={`prediction-box ${prediction.prediction}`}>
                <div className="prediction-header">
                  <h3>
                    {prediction.prediction === 'helpful' ? '✅ Helpful' : '❌ Unhelpful'}
                  </h3>
                  <span className={`confidence-badge ${prediction.confidence}`}>
                    {prediction.confidence.toUpperCase()} Confidence
                  </span>
                </div>
                
                <div className="probability-bar">
                  <div className="probability-label">
                    Probability: {(prediction.probability * 100).toFixed(1)}%
                  </div>
                  <div className="progress-bar">
                    <div 
                      className="progress-fill"
                      style={{ width: `${prediction.probability * 100}%` }}
                    />
                  </div>
                </div>

                {prediction.suggestions && prediction.suggestions.length > 0 && (
                  <div className="suggestions">
                    <h4>Suggestions:</h4>
                    <ul>
                      {prediction.suggestions.map((suggestion, idx) => (
                        <li key={idx}>{suggestion}</li>
                      ))}
                    </ul>
                  </div>
                )}

                <div className="metadata">
                  <small>
                    Processing time: {prediction.processing_time_ms.toFixed(0)}ms | 
                    Model: {prediction.model_version}
                  </small>
                </div>

                {!feedbackSubmitted && (
                  <div className="feedback-section">
                    <h4>Was this prediction correct?</h4>
                    <p className="feedback-question">
                      Model predicted: <strong>{prediction.prediction}</strong>
                    </p>
                    <div className="feedback-buttons">
                      {prediction.prediction === 'helpful' ? (
                        <>
                          <button 
                            className="feedback-btn helpful"
                            onClick={() => handleFeedback('helpful')}
                          >
                            Prediction is correct
                          </button>
                          <button 
                            className="feedback-btn unhelpful"
                            onClick={() => handleFeedback('unhelpful')}
                          >
                            Prediction is wrong
                          </button>
                        </>
                      ) : (
                        <>
                          <button 
                            className="feedback-btn helpful"
                            onClick={() => handleFeedback('helpful')}
                          >
                            Prediction is wrong
                          </button>
                          <button 
                            className="feedback-btn unhelpful"
                            onClick={() => handleFeedback('unhelpful')}
                          >
                            Prediction is correct
                          </button>
                        </>
                      )}
                    </div>
                    <small>Your feedback helps improve our model through online learning!</small>
                  </div>
                )}

                {feedbackSubmitted && (
                  <div className="feedback-thanks">
                    <p>Thank you for your feedback!</p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        <footer className="footer">
          <p>Review Helpfulness Predictor - Powered by Neural Network ML Model</p>
          <p>Model Accuracy: 86.76% | F1 Score: 84.75%</p>
        </footer>
      </div>
    </div>
  );
}

export default App;

