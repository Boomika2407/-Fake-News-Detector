"""
Production-ready Flask API for real-time BERT predictions with LIME explanations.
Usage: python lime_api_server.py
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from lime.lime_text import LimeTextExplainer
import threading
from functools import lru_cache
import os
import json
from datetime import datetime
import sys
import argparse
import gc
import re
from collections import Counter
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

try:
    from nrclex import NRCLex
except Exception:
    NRCLex = None

# ============================================================================
# CONFIGURATION
# ============================================================================

# HuggingFace model: RoBERTa fine-tuned on WELFake dataset for fake news detection
MODEL_DIR = "himel05/fake-news-roberta"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else "cpu")
MAX_TEXT_LENGTH = 15000  # Truncate texts longer than this
LIME_NUM_FEATURES = 10  # Number of top features to explain
LIME_NUM_SAMPLES = 50   # Reduced from 150 → saves ~3× RAM on t2.micro (1 GB)
ENABLE_CACHING = True  # Cache LIME explainer initialization
DEFAULT_PORT = 5001  # Aligned with frontend and README
DEFAULT_HOST = '0.0.0.0'  # Listen on all interfaces

# ── t2.micro / low-RAM optimizations ─────────────────────────────────────────
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # No fork overhead

vader_analyzer = SentimentIntensityAnalyzer()

PRONOUNS = {
    "i", "we", "you", "he", "she", "they", "me", "us", "him", "her", "them", "my", "our", "your", "their"
}

MODAL_WORDS = {
    "may", "might", "could", "would", "should", "must", "can", "cannot", "likely", "possibly", "perhaps"
}

COMMON_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "while", "with", "to", "of", "in", "on", "for", "from",
    "by", "at", "as", "is", "are", "was", "were", "be", "been", "being", "this", "that", "these", "those",
    "it", "its", "than", "then", "into", "about", "over", "after", "before", "between", "through"
}

# ============================================================================
# GLOBAL MODELS - loaded once at startup
# ============================================================================

tokenizer = None
model = None
explainer_lime = None
lock = threading.Lock()  # For thread-safe model access


def release_runtime_memory():
    """Best-effort memory cleanup after each request."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available() and hasattr(torch, "mps"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass


def split_sentences(text):
    sentences = [sentence.strip() for sentence in re.split(r"[.!?]+", text) if sentence.strip()]
    return sentences or [text.strip()]


def tokenize_words(text):
    return re.findall(r"\b[A-Za-z][A-Za-z'-]*\b", text)


def count_syllables(word):
    word = word.lower()
    if not word:
        return 0
    vowels = "aeiouy"
    count = 0
    prev_is_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_is_vowel:
            count += 1
        prev_is_vowel = is_vowel
    if word.endswith("e") and count > 1:
        count -= 1
    return max(count, 1)


def analyze_linguistic_features(text):
    words = tokenize_words(text)
    lower_words = [word.lower() for word in words]
    word_count = len(lower_words)
    sentence_count = max(len(split_sentences(text)), 1)

    if NRCLex is not None:
        nrc = NRCLex()
        lexicon = getattr(nrc, "__lexicon__", {})
        emotion_words = Counter()
        for word in lower_words:
            for emotion in lexicon.get(word, []):
                if emotion in {"anger", "fear", "joy"}:
                    emotion_words[emotion] += 1
        emotion_total = sum(emotion_words.values()) or 1
        emotion_scores = {
            "anger": round(emotion_words.get("anger", 0) / emotion_total, 3),
            "fear": round(emotion_words.get("fear", 0) / emotion_total, 3),
            "joy": round(emotion_words.get("joy", 0) / emotion_total, 3),
        }
        lexicon_subjectivity = round(sum(1 for word in lower_words if lexicon.get(word)) / max(word_count, 1), 3)
    else:
        emotion_scores = {"anger": 0.0, "fear": 0.0, "joy": 0.0}
        lexicon_subjectivity = 0.0

    vader_scores = vader_analyzer.polarity_scores(text)
    polarity = round(vader_scores.get("compound", 0.0), 3)
    subjectivity = round(max(abs(vader_scores.get("pos", 0.0)), abs(vader_scores.get("neg", 0.0))) + lexicon_subjectivity, 3)

    syllables = sum(count_syllables(word) for word in lower_words)
    avg_sentence_length = round(word_count / max(sentence_count, 1), 2)
    flesch_score = round(206.835 - 1.015 * avg_sentence_length - 84.6 * (syllables / max(word_count, 1)), 1)

    pronoun_count = sum(1 for word in lower_words if word in PRONOUNS)
    modal_count = sum(1 for word in lower_words if word in MODAL_WORDS)
    capitalized_tokens = re.findall(r"\b[A-Z][a-z]+\b|\b[A-Z]{2,}\b", text)
    named_entities_count = sum(
        1 for token in capitalized_tokens
        if token.lower() not in COMMON_STOPWORDS and token not in {"I"}
    )
    noun_like_count = sum(
        1 for word in lower_words
        if word not in COMMON_STOPWORDS
        and word not in PRONOUNS
        and word not in MODAL_WORDS
        and len(word) > 2
    )

    repeated_words = [word for word, count in Counter(lower_words).items() if count > 2 and word not in COMMON_STOPWORDS]
    repetition_ratio = round(len(repeated_words) / max(len(set(lower_words)), 1), 3)

    emotion_max = max(emotion_scores, key=emotion_scores.get)
    emotion_strength = emotion_scores[emotion_max]
    dominant_emotion = emotion_max if emotion_strength > 0 else "none"
    emotion_interpretation = (
        "high emotional tone → aligns with fake news patterns"
        if emotion_strength >= 0.08
        else "low emotional tone → more neutral/factual framing"
    )

    sentiment_style = (
        "subjective/opinionated → less factual tone"
        if subjectivity >= 0.12
        else "more objective tone → closer to factual reporting"
    )

    readability_style = (
        "complex/formal → more like real news"
        if flesch_score < 50
        else "simple/easy-to-read → can appear more promotional or conversational"
    )

    structure_style = (
        "dense named-entity focus and high content-word ratio → narrative framing"
        if named_entities_count >= 3 or repetition_ratio >= 0.15
        else "balanced structure → fewer obvious stylistic signals"
    )

    combined_interpretation = (
        "The text shows the strongest linguistic signal in "
        f"{dominant_emotion} emotion and {sentiment_style}. "
        f"Readability is {readability_style.lower()}. "
        f"Structural cues suggest {structure_style.lower()}."
    )

    return {
        "emotion": {
            **emotion_scores,
            "dominant": dominant_emotion,
            "interpretation": emotion_interpretation,
        },
        "sentiment_subjectivity": {
            "polarity": polarity,
            "subjectivity": subjectivity,
            "interpretation": sentiment_style,
        },
        "readability_formality": {
            "flesch_score": flesch_score,
            "avg_sentence_length": avg_sentence_length,
            "interpretation": readability_style,
        },
        "structural_signals": {
            "pronoun_ratio": round(pronoun_count / max(word_count, 1), 3),
            "noun_ratio": round(noun_like_count / max(word_count, 1), 3),
            "named_entities_count": int(named_entities_count),
            "modal_ratio": round(modal_count / max(word_count, 1), 3),
            "repetition_ratio": repetition_ratio,
            "interpretation": structure_style,
        },
        "combined_insight": combined_interpretation,
    }


def generate_combined_insight(prediction, confidence, linguistic):
    """
    Generates a natural-sounding, human-friendly multi-paragraph description
    combining the BERT LLM classification results and the heuristic linguistic analysis.
    """
    class_str = "Fake News" if prediction == "Fake News" else "True News"
    conf_pct = f"{confidence * 100:.1f}%"
    
    lead_sentence = f"The fine-tuned LLM (BERT) model classified this text as **{class_str}** with a confidence score of **{conf_pct}** based on its stylistic patterns. "
    
    emotion = linguistic["emotion"]
    dom_emotion = emotion["dominant"]
    anger_score = emotion["anger"]
    fear_score = emotion["fear"]
    joy_score = emotion["joy"]
    
    if dom_emotion != "none":
        emotion_str = f"Linguistically, the text displays a distinct emotional tone, with **{dom_emotion}** being the dominant sentiment (anger: {anger_score:.2f}, fear: {fear_score:.2f}, joy: {joy_score:.2f}). "
        if dom_emotion in ["anger", "fear"]:
            emotion_str += "This aligns with common patterns in sensational or misleading news, which often use fear or anger to trigger quick emotional responses from readers. "
        else:
            emotion_str += "The presence of positive emotion suggests a promotional or celebratory tone. "
    else:
        emotion_str = "Linguistically, the text maintains a very low emotional tone, indicating a more neutral, balanced reporting style. "

    subj = linguistic["sentiment_subjectivity"]
    subjectivity_val = subj["subjectivity"]
    polarity_val = subj["polarity"]
    
    if subjectivity_val >= 0.12:
        subj_str = f"The writing style is highly opinionated and subjective (subjectivity index: {subjectivity_val:.2f}), relying heavily on personal framing or bias rather than purely objective statements. "
        if class_str == "Fake News":
            subj_str += "This highly subjective style strongly reinforces the LLM's prediction that the text is fake. "
        else:
            subj_str += "Although subjective, the LLM finds the stylistic markers consistent with authentic opinion-based writing. "
    else:
        subj_str = f"The writing is structured objectively (subjectivity index: {subjectivity_val:.2f}) with neutral framing. "
        if class_str == "True News":
            subj_str += "This objective approach aligns perfectly with standard factual reporting conventions. "
        else:
            subj_str += "However, despite the objective tone, the LLM identified underlying stylistic signals of fabrication. "

    read = linguistic["readability_formality"]
    flesch = read["flesch_score"]
    sent_len = read["avg_sentence_length"]
    
    struct = linguistic["structural_signals"]
    ne_count = struct["named_entities_count"]
    rep_ratio = struct["repetition_ratio"]
    
    read_str = f"With a Flesch Readability score of {flesch:.1f} and an average sentence length of {sent_len:.1f} words, the text is "
    if flesch < 50:
        read_str += "complex and formal, adhering to traditional journalistic structures. "
    else:
        read_str += "simple and conversational, resembling informal blog posts or social media claims. "
        
    struct_str = f"Structurally, it contains {ne_count} named entities and a repetition ratio of {rep_ratio * 100:.1f}%. "
    if ne_count >= 3 or rep_ratio >= 0.15:
        struct_str += "This dense focus and repetitive pattern suggest a narrative framing designed to drive a specific point. "
    else:
        struct_str += "This balanced distribution of content words points to a straightforward reporting format. "

    if class_str == "Fake News":
        conclusion = "In conclusion, both the LLM style classifier and the linguistic metrics point towards a sensationalized or fabricated style of writing designed for persuasion rather than raw reporting."
    else:
        conclusion = "In conclusion, the stylistic analysis from the LLM is well-supported by the formal, objective, and neutral linguistic patterns, indicating a high likelihood of genuine journalistic reporting."

    combined_insight = f"{lead_sentence}\n\n{emotion_str}\n\n{subj_str}\n\n{read_str}{struct_str}\n\n{conclusion}"
    return combined_insight


def initialize_models():
    """Load model and explainer once at startup"""
    global tokenizer, model, explainer_lime

    print(f"[INFO] Loading model '{MODEL_DIR}' from HuggingFace Hub (will download on first run)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    # low_cpu_mem_usage=True loads weights shard-by-shard → halves peak RAM during load
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_DIR,
        low_cpu_mem_usage=True   # critical for t2.micro (1 GB RAM)
    ).to(DEVICE)
    model.eval()
    gc.collect()  # Free loader buffers immediately after model is loaded
    print(f"[INFO] Model loaded. Using device: {DEVICE}")

    print("[INFO] Initializing LIME explainer...")
    explainer_lime = LimeTextExplainer(
        class_names=['True News', 'Fake News'],
        verbose=False
    )
    print("[INFO] LIME explainer ready")


# ============================================================================
# PREDICTION FUNCTION (required by LIME)
# ============================================================================

def predict_proba_batch(texts, batch_size=4):  # Smaller batch → less peak RAM on t2.micro
    """
    Predict probabilities for a batch of texts.
    Returns: numpy array of shape (n_samples, 2)
    """
    if isinstance(texts, str):
        texts = [texts]

    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            
    return np.vstack(all_probs)


# ============================================================================
# CORE PREDICTION + EXPLANATION FUNCTION
# ============================================================================

def predict_with_explanation(text, threshold=0.5):
    """
    Generate prediction and LIME explanation for a single text.

    Args:
        text: Input news text
        threshold: Confidence threshold for including explanations

    Returns:
        dict with prediction, confidence, and explanation
    """
    # Truncate if too long
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]

    with lock:
        try:
            # Get prediction
            probs = predict_proba_batch(text)
            pred_label = int(np.argmax(probs[0]))
            confidence = float(np.max(probs[0]))

            result = {
                "success": True,
                "prediction": "Fake News" if pred_label == 1 else "True News",
                "prediction_label": pred_label,
                "confidence": confidence,
                "prob_true": float(probs[0][0]),
                "prob_fake": float(probs[0][1]),
                "timestamp": datetime.now().isoformat()
            }

            linguistic = analyze_linguistic_features(text)
            linguistic["combined_insight"] = generate_combined_insight(
                result["prediction"],
                confidence,
                linguistic
            )
            result["linguistic_analysis"] = linguistic

            # Generate LIME explanation if confidence is high enough
            # (optional: skip for low-confidence predictions to save computation)
            if confidence >= threshold:
                try:
                    lime_exp = explainer_lime.explain_instance(
                        text,
                        predict_proba_batch,
                        num_features=LIME_NUM_FEATURES,
                        num_samples=LIME_NUM_SAMPLES,
                        top_labels=1
                    )

                    # Extract features and weights
                    features_weights = dict(lime_exp.as_list(label=pred_label))
                    explanation_list = [
                        {
                            "feature": feature,
                            "weight": float(weight),
                            "direction": "supports_fake" if weight > 0 else "supports_true"
                        }
                        for feature, weight in sorted(
                            features_weights.items(),
                            key=lambda x: abs(x[1]),
                            reverse=True
                        )
                    ]

                    result["explanation"] = {
                        "method": "LIME",
                        "num_features": len(explanation_list),
                        "features": explanation_list,
                        "interpretation": (
                            "Positive weights support 'Fake News' prediction. "
                            "Negative weights support 'True News' prediction."
                        )
                    }

                except Exception as e:
                    result["explanation"] = {
                        "error": f"Could not generate explanation: {str(e)[:100]}"
                    }
            else:
                result["explanation"] = {
                    "skipped": f"Confidence ({confidence:.3f}) below threshold ({threshold})"
                }

            return result

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        finally:
            release_runtime_memory()


# ============================================================================
# FLASK API
# ============================================================================

app = Flask(__name__)
CORS(app)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_device": str(DEVICE),
        "timestamp": datetime.now().isoformat()
    })


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    Predict and explain a single news article.

    Request JSON:
    {
        "text": "Article text here...",
        "threshold": 0.5  (optional, default 0.5)
    }

    Response:
    {
        "prediction": "Fake News" | "True News",
        "confidence": 0.95,
        "explanation": {
            "method": "LIME",
            "features": [
                {"feature": "word", "weight": 0.15, "direction": "supports_fake"},
                ...
            ]
        }
    }
    """
    try:
        data = request.get_json()

        if not data or "text" not in data:
            return jsonify({"error": "Missing 'text' field"}), 400

        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "Text cannot be empty"}), 400

        threshold = float(data.get("threshold", 0.5))
        threshold = max(0.0, min(1.0, threshold))  # Clamp to [0, 1]

        result = predict_with_explanation(text, threshold)
        status_code = 200 if result.get("success", False) else 400

        return jsonify(result), status_code

    except Exception as e:
        return jsonify({
            "error": f"Server error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch_endpoint():
    """
    Predict and explain multiple articles (batch mode).

    Request JSON:
    {
        "texts": ["Article 1...", "Article 2...", ...],
        "threshold": 0.5  (optional)
    }

    Response: Array of prediction results
    """
    try:
        data = request.get_json()

        if not data or "texts" not in data:
            return jsonify({"error": "Missing 'texts' field"}), 400

        texts = data.get("texts", [])
        if not isinstance(texts, list) or len(texts) == 0:
            return jsonify({"error": "texts must be a non-empty list"}), 400

        threshold = float(data.get("threshold", 0.5))
        threshold = max(0.0, min(1.0, threshold))

        results = [predict_with_explanation(text, threshold) for text in texts]

        return jsonify({
            "success": True,
            "num_samples": len(results),
            "predictions": results,
            "timestamp": datetime.now().isoformat()
        }), 200

    except Exception as e:
        return jsonify({
            "error": f"Server error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500


@app.route('/info', methods=['GET'])
def info_endpoint():
    """Get API information and configuration"""
    return jsonify({
        "api": "BERT News Classification with LIME Explanations",
        "version": "1.0",
        "model": "BERT (fine-tuned on WELFake dataset)",
        "device": str(DEVICE),
        "max_text_length": MAX_TEXT_LENGTH,
        "lime_features": LIME_NUM_FEATURES,
        "lime_num_samples": LIME_NUM_SAMPLES,
        "linguistic_analysis": "Lightweight heuristic analysis of emotion, sentiment, readability, and structure",
        "endpoints": {
            "/health": "GET - Health check",
            "/info": "GET - API information",
            "/predict": "POST - Single prediction with explanation",
            "/predict/batch": "POST - Batch predictions"
        },
        "example_request": {
            "text": "Your news article text here...",
            "threshold": 0.5
        }
    })


# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='BERT News Classification API with LIME Explanations')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, 
                        help=f'Port to run the API on (default: {DEFAULT_PORT})')
    parser.add_argument('--host', type=str, default=DEFAULT_HOST,
                        help=f'Host to bind to (default: {DEFAULT_HOST})')
    parser.add_argument('--debug', action='store_true',
                        help='Run in debug mode')
    args = parser.parse_args()

    print("="*70)
    print("BERT News Classification API - LIME Explanations")
    print("="*70)

    # Initialize models
    initialize_models()

    print("\n[INFO] Starting Flask API server...")
    print(f"[INFO] API will be available at http://localhost:{args.port}")
    print(f"[INFO] Host: {args.host}")
    print("\n[INFO] Endpoints:")
    print("  GET  /health     - Health check")
    print("  GET  /info       - API information")
    print("  POST /predict    - Single prediction + explanation")
    print("  POST /predict/batch - Batch predictions")
    print("\n" + "="*70 + "\n")

    # Run Flask app
    try:
        app.run(debug=args.debug, host=args.host, port=args.port, threaded=True)
    except OSError as e:
        if "Address already in use" in str(e):
            print(f"\n❌ ERROR: Port {args.port} is already in use!")
            print(f"\nTo use a different port, run:")
            print(f"  python3 lime_api_server.py --port 5001")
            print(f"\nOr to kill the process using port {args.port}:")
            print(f"  lsof -ti:{args.port} | xargs kill -9")
            sys.exit(1)
        else:
            raise
