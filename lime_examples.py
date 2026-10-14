"""
Comprehensive Examples: LIME On-Demand Explanation Generation

This file demonstrates various ways to use LIME for generating explanations
in real-time applications.
"""

# ============================================================================
# EXAMPLE 1: Local LIME Usage (No API, Direct Python)
# ============================================================================

print("=" * 80)
print("EXAMPLE 1: Direct LIME Usage (Local Python)")
print("=" * 80)

"""
Use this when you want to generate explanations in a script without running a server.
Good for: batch processing, notebooks, integrated applications
"""

import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from lime.lime_text import LimeTextExplainer

# Load model and tokenizer
model_dir = "/Users/bahaar/Desktop/NLP/bert_welfake_model"
tokenizer = BertTokenizer.from_pretrained(model_dir)
model = BertForSequenceClassification.from_pretrained(model_dir)
model.eval()

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Define prediction function for LIME
def predict_fn(texts):
    """Prediction function that LIME can call"""
    if isinstance(texts, str):
        texts = [texts]
    
    with torch.no_grad():
        inputs = tokenizer(texts, return_tensors="pt", padding=True, 
                          truncation=True, max_length=256)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        return probs.cpu().numpy()

# Initialize LIME explainer
explainer = LimeTextExplainer(class_names=['True News', 'Fake News'])

# Example: Generate explanation for a single text
text = "Breaking: New study confirms vaccine effectiveness despite media claims"
prediction = predict_fn(text)
predicted_label = int(np.argmax(prediction[0]))

print(f"\nText: {text}")
print(f"Prediction: {'Fake' if predicted_label == 1 else 'True'} News")
print(f"Confidence: {max(prediction[0]):.3f}")

# Generate LIME explanation
exp = explainer.explain_instance(text, predict_fn, num_features=10, top_labels=1)

print("\nTop contributing words:")
for word, weight in exp.as_list(label=predicted_label)[:5]:
    direction = "→ FAKE" if weight > 0 else "→ TRUE"
    print(f"  {word:<40} {weight:>8.4f}  {direction}")


# ============================================================================
# EXAMPLE 2: Caching Explanations for Performance
# ============================================================================

print("\n\n" + "=" * 80)
print("EXAMPLE 2: Caching Explanations for Repeated Texts")
print("=" * 80)

"""
If the same articles appear multiple times, cache the explanations.
"""

from functools import lru_cache
import hashlib

# Create a cache key function
def get_text_hash(text):
    return hashlib.md5(text.encode()).hexdigest()

# Simple in-memory cache
explanation_cache = {}

def get_explanation_cached(text, use_cache=True):
    """Generate explanation with optional caching"""
    
    if use_cache:
        cache_key = get_text_hash(text)
        if cache_key in explanation_cache:
            print(f"  [CACHED] Retrieved from cache")
            return explanation_cache[cache_key]
    
    # Generate new explanation
    print(f"  [GENERATING] Computing explanation...")
    prediction = predict_fn(text)
    predicted_label = int(np.argmax(prediction[0]))
    exp = explainer.explain_instance(text, predict_fn, num_features=10)
    
    result = {
        'text': text,
        'label': predicted_label,
        'confidence': float(max(prediction[0])),
        'explanation': exp.as_list(label=predicted_label)
    }
    
    if use_cache:
        cache_key = get_text_hash(text)
        explanation_cache[cache_key] = result
    
    return result

# Test caching
sample_text = "Scientists discover new treatment for disease"

print("\nFirst request (should generate):")
result1 = get_explanation_cached(sample_text)

print("Second request (should use cache):")
result2 = get_explanation_cached(sample_text)

print(f"Cache size: {len(explanation_cache)} entries")


# ============================================================================
# EXAMPLE 3: Adaptive Explanation Generation
# ============================================================================

print("\n\n" + "=" * 80)
print("EXAMPLE 3: Adaptive Explanation Generation")
print("=" * 80)

"""
Generate more detailed explanations for uncertain predictions,
skip for high-confidence predictions (faster).
"""

def adaptive_explain(text, confidence_threshold=0.7, 
                     simple_features=5, detailed_features=15):
    """Generate explanation based on prediction confidence"""
    
    prediction = predict_fn(text)
    predicted_label = int(np.argmax(prediction[0]))
    confidence = max(prediction[0])
    
    # Decide explanation complexity based on confidence
    if confidence >= confidence_threshold:
        # High confidence: simple explanation (faster)
        num_features = simple_features
        explanation_type = "SIMPLE"
    else:
        # Low confidence: detailed explanation (more thorough)
        num_features = detailed_features
        explanation_type = "DETAILED"
    
    exp = explainer.explain_instance(text, predict_fn, num_features=num_features)
    
    return {
        'prediction': 'Fake' if predicted_label == 1 else 'True',
        'confidence': confidence,
        'explanation_type': explanation_type,
        'features': num_features,
        'explanation': exp.as_list(label=predicted_label)
    }

text1 = "Official government announcement on policy change"  # Likely high confidence
text2 = "Report: New technology breakthrough unveiled"         # Likely uncertain

print("\nHigh-confidence example:")
result = adaptive_explain(text1)
print(f"  Prediction: {result['prediction']} ({result['confidence']:.1%})")
print(f"  Explanation type: {result['explanation_type']} ({result['features']} features)")

print("\nUncertain example:")
result = adaptive_explain(text2)
print(f"  Prediction: {result['prediction']} ({result['confidence']:.1%})")
print(f"  Explanation type: {result['explanation_type']} ({result['features']} features)")


# ============================================================================
# EXAMPLE 4: Batch Processing with Explanations
# ============================================================================

print("\n\n" + "=" * 80)
print("EXAMPLE 4: Batch Processing with Explanations")
print("=" * 80)

"""
Process multiple texts efficiently with explanations.
"""

texts = [
    "Major breakthrough in quantum computing announced",
    "New law passed to regulate social media platforms",
    "Aliens spotted near Area 51 according to leaked documents"
]

print(f"\nProcessing {len(texts)} texts...\n")

results = []
for i, text in enumerate(texts, 1):
    print(f"[{i}/{len(texts)}] {text[:50]}...")
    
    prediction = predict_fn(text)
    predicted_label = int(np.argmax(prediction[0]))
    confidence = max(prediction[0])
    
    exp = explainer.explain_instance(text, predict_fn, num_features=8)
    
    results.append({
        'text': text[:50] + '...',
        'prediction': 'Fake' if predicted_label == 1 else 'True',
        'confidence': confidence,
        'top_feature': exp.as_list(label=predicted_label)[0][0]
    })

# Display results
print("\nBatch Results:")
print(f"{'Text':<50} {'Prediction':<12} {'Confidence':<12} {'Top Feature':<30}")
print("-" * 104)
for r in results:
    print(f"{r['text']:<50} {r['prediction']:<12} {r['confidence']:<12.3f} {r['top_feature']:<30}")


# ============================================================================
# EXAMPLE 5: Handling Edge Cases
# ============================================================================

print("\n\n" + "=" * 80)
print("EXAMPLE 5: Handling Edge Cases")
print("=" * 80)

"""
Demonstrate how to handle various edge cases in production.
"""

def safe_explain(text, max_text_length=2000):
    """Generate explanation with error handling and limits"""
    
    try:
        # Validate input
        if not text or not isinstance(text, str):
            return {'error': 'Invalid input: text must be non-empty string'}
        
        # Truncate if too long
        if len(text) > max_text_length:
            text = text[:max_text_length]
            print(f"  [INFO] Text truncated to {max_text_length} characters")
        
        # Generate prediction
        prediction = predict_fn(text)
        predicted_label = int(np.argmax(prediction[0]))
        confidence = max(prediction[0])
        
        # Skip explanation if very low confidence (might be noise)
        if confidence < 0.5:
            return {
                'prediction': 'True' if predicted_label == 0 else 'Fake',
                'confidence': confidence,
                'warning': 'Low confidence prediction - explanation skipped',
                'explanation': None
            }
        
        # Generate explanation
        exp = explainer.explain_instance(text, predict_fn, num_features=10)
        
        return {
            'prediction': 'True' if predicted_label == 0 else 'Fake',
            'confidence': confidence,
            'explanation': exp.as_list(label=predicted_label),
            'num_words': len(text.split())
        }
    
    except Exception as e:
        return {'error': f'Failed to process: {str(e)}'}

# Test edge cases
print("\nEdge case 1: Empty text")
result = safe_explain("")
print(f"  Result: {result}")

print("\nEdge case 2: Very long text")
long_text = " ".join(["word"] * 5000)
result = safe_explain(long_text)
print(f"  Result: Words={result.get('num_words')}, Error={result.get('error')}")

print("\nEdge case 3: Special characters")
special_text = "Test with émojis 🚀 and üñíçødé"
result = safe_explain(special_text)
print(f"  Result: Prediction={result.get('prediction')}, Confidence={result.get('confidence'):.3f}")


# ============================================================================
# EXAMPLE 6: Real-time Streaming
# ============================================================================

print("\n\n" + "=" * 80)
print("EXAMPLE 6: Simulated Real-time Streaming")
print("=" * 80)

"""
Simulate processing a stream of news articles in real-time.
"""

import time
from collections import deque

class NewsStream:
    def __init__(self, max_queue=10):
        self.queue = deque(maxlen=max_queue)
    
    def add_article(self, text):
        """Add article to processing queue"""
        self.queue.append({'text': text, 'timestamp': time.time()})
    
    def process_stream(self):
        """Process all articles in queue"""
        results = []
        while self.queue:
            article = self.queue.popleft()
            text = article['text']
            
            prediction = predict_fn(text)
            predicted_label = int(np.argmax(prediction[0]))
            confidence = max(prediction[0])
            
            result = {
                'text': text[:40] + '...',
                'prediction': 'Fake' if predicted_label == 1 else 'True',
                'confidence': confidence,
                'processed_at': time.time()
            }
            results.append(result)
        
        return results

# Simulate incoming stream
stream = NewsStream(max_queue=5)

print("\nAdding articles to stream...")
articles = [
    "Stock market hits new high amid economic growth",
    "Celebrity announces shocking news on social media",
    "Scientists develop new sustainable energy source"
]

for article in articles:
    stream.add_article(article)
    print(f"  Added: {article[:50]}...")

print("\nProcessing stream:")
results = stream.process_stream()
for r in results:
    print(f"  {r['text']:<45} {r['prediction']:<8} {r['confidence']:.3f}")


# ============================================================================
# EXAMPLE 7: Comparison with Multiple Explanations
# ============================================================================

print("\n\n" + "=" * 80)
print("EXAMPLE 7: Comparing Different Feature Counts")
print("=" * 80)

"""
Show how changing num_features affects explanation quality and speed.
"""

text = "New vaccine approved by health authorities after successful trials"

print(f"\nText: {text}\n")
print(f"{'Num Features':<15} {'Top 3 Features':<60}")
print("-" * 75)

for num_features in [3, 5, 10, 15]:
    exp = explainer.explain_instance(text, predict_fn, num_features=num_features)
    top_features = [word for word, _ in exp.as_list()[:3]]
    features_str = ", ".join(top_features)
    print(f"{num_features:<15} {features_str:<60}")


# ============================================================================
# EXAMPLE 8: Export Results for Analysis
# ============================================================================

print("\n\n" + "=" * 80)
print("EXAMPLE 8: Export Results as DataFrame")
print("=" * 80)

"""
Export explanation results to pandas DataFrame for analysis and visualization.
"""

import pandas as pd

# Generate explanations for multiple texts
test_articles = [
    "Official statement from government ministry",
    "Unconfirmed reports from anonymous sources",
    "Research paper published in peer-reviewed journal",
    "Social media post goes viral with sensational claim"
]

results_list = []

for text in test_articles:
    prediction = predict_fn(text)
    predicted_label = int(np.argmax(prediction[0]))
    confidence = max(prediction[0])
    
    exp = explainer.explain_instance(text, predict_fn, num_features=5)
    top_feature = exp.as_list(label=predicted_label)[0][0]
    top_weight = exp.as_list(label=predicted_label)[0][1]
    
    results_list.append({
        'text': text,
        'prediction': 'Fake' if predicted_label == 1 else 'True',
        'confidence': confidence,
        'top_feature': top_feature,
        'feature_weight': top_weight
    })

# Convert to DataFrame
df = pd.DataFrame(results_list)

print("\nResults DataFrame:")
print(df.to_string(index=False))

# Save to CSV
csv_path = "/Users/bahaar/Desktop/NLP/lime_explanations_example.csv"
df.to_csv(csv_path, index=False)
print(f"\n✓ Results saved to: {csv_path}")

# Summary statistics
print("\nSummary Statistics:")
print(f"  Average confidence: {df['confidence'].mean():.3f}")
print(f"  Fake/True split: {(df['prediction'] == 'Fake').sum()} fake, {(df['prediction'] == 'True').sum()} true")


# ============================================================================
# SUMMARY
# ============================================================================

print("\n\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

summary = """
Key Takeaways for On-Demand LIME Explanations:

1. LOCAL PYTHON USAGE:
   - Import tokenizer, model, and LimeTextExplainer
   - Define prediction function LIME can call
   - Call explainer.explain_instance() for each text

2. PERFORMANCE OPTIMIZATION:
   - Cache explanations for repeated texts
   - Use adaptive complexity based on confidence
   - Batch process when possible

3. PRODUCTION CONSIDERATIONS:
   - Wrap in API (Flask/FastAPI) for web access
   - Add error handling and input validation
   - Implement caching strategies
   - Monitor performance and memory usage

4. DEPLOYMENT OPTIONS:
   - Local: Direct Python usage (notebook, script)
   - Web API: Flask/FastAPI server
   - Cloud: Docker + Google Cloud Run / AWS / Azure
   - Real-time: Kafka streaming + worker processes

5. BEST PRACTICES:
   - Always validate input text
   - Set reasonable max_length (256-512 tokens)
   - Use threading/async for concurrent requests
   - Log and monitor explanation generation
   - Cache results when appropriate

See LIME_DEPLOYMENT_GUIDE.md for comprehensive production setup.
"""

print(summary)
