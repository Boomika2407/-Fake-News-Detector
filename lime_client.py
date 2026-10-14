"""
Python client library for the BERT News Classification API with LIME explanations.

Usage:
    from lime_client import NewsClassificationClient
    
    client = NewsClassificationClient(api_url="http://localhost:5000")
    result = client.predict("Your news text here...")
    
    print(result['prediction'])
    print(result['confidence'])
    for feature in result['explanation']['features']:
        print(f"  {feature['feature']}: {feature['weight']:.4f}")
"""

import requests
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PredictionResult:
    """Structured result from prediction"""
    prediction: str
    prediction_label: int
    confidence: float
    prob_true: float
    prob_fake: float
    explanation: Optional[Dict[str, Any]] = None
    linguistic_analysis: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    success: bool = True
    error: Optional[str] = None

    def __repr__(self):
        return (
            f"PredictionResult(prediction='{self.prediction}', "
            f"confidence={self.confidence:.4f})"
        )


class NewsClassificationClient:
    """Client for interacting with the News Classification API"""

    def __init__(self, api_url: str = "http://localhost:5001", timeout: int = 30):
        """
        Initialize the client.

        Args:
            api_url: Base URL of the API server
            timeout: Request timeout in seconds
        """
        self.api_url = api_url.rstrip('/')
        self.timeout = timeout

    def health_check(self) -> bool:
        """Check if the API server is running"""
        try:
            response = requests.get(
                f"{self.api_url}/health",
                timeout=self.timeout
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False

    def get_info(self) -> Dict[str, Any]:
        """Get API information and available endpoints"""
        try:
            response = requests.get(
                f"{self.api_url}/info",
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            raise RuntimeError(f"Failed to get API info: {e}")

    def predict(
        self,
        text: str,
        threshold: float = 0.5,
        return_raw: bool = False
    ) -> PredictionResult:
        """
        Predict and explain a single news article.

        Args:
            text: News article text
            threshold: Confidence threshold for generating explanations (0-1)
            return_raw: If True, return raw API response as dict

        Returns:
            PredictionResult object or raw dict if return_raw=True
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        threshold = max(0.0, min(1.0, float(threshold)))

        try:
            response = requests.post(
                f"{self.api_url}/predict",
                json={"text": text.strip(), "threshold": threshold},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            if return_raw:
                return data

            return PredictionResult(**data)

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"API request failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to parse response: {e}")

    def predict_batch(
        self,
        texts: List[str],
        threshold: float = 0.5,
        return_raw: bool = False
    ) -> List[PredictionResult]:
        """
        Predict and explain multiple articles.

        Args:
            texts: List of news article texts
            threshold: Confidence threshold for generating explanations
            return_raw: If True, return raw API response

        Returns:
            List of PredictionResult objects
        """
        if not texts:
            raise ValueError("texts list cannot be empty")

        threshold = max(0.0, min(1.0, float(threshold)))

        try:
            response = requests.post(
                f"{self.api_url}/predict/batch",
                json={"texts": texts, "threshold": threshold},
                timeout=self.timeout * len(texts)  # Scale timeout for batch
            )
            response.raise_for_status()
            data = response.json()

            if return_raw:
                return data

            predictions = data.get('predictions', [])
            return [PredictionResult(**pred) for pred in predictions]

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Batch API request failed: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to parse batch response: {e}")

    def format_explanation(self, result: PredictionResult) -> str:
        """Format explanation as readable text"""
        if not result.explanation:
            return "No explanation available"

        if 'error' in result.explanation:
            return f"Error: {result.explanation['error']}"

        if 'skipped' in result.explanation:
            return f"Explanation skipped: {result.explanation['skipped']}"

        lines = [f"Prediction: {result.prediction} ({result.confidence:.1%} confidence)\n"]
        lines.append("Top contributing features:")

        for feature in result.explanation.get('features', [])[:10]:
            direction = "→ FAKE" if feature['direction'] == 'supports_fake' else "→ TRUE"
            lines.append(f"  • \"{feature['feature']:<40}\" {feature['weight']:>8.4f}  {direction}")

        linguistic = result.linguistic_analysis or result.explanation.get('linguistic_analysis')
        if linguistic:
            lines.append("\nLinguistic analysis:")
            emotion = linguistic.get('emotion', {})
            sentiment = linguistic.get('sentiment_subjectivity', {})
            readability = linguistic.get('readability_formality', {})
            structure = linguistic.get('structural_signals', {})

            lines.append(
                f"  Emotion: anger={emotion.get('anger', 0):.3f}, fear={emotion.get('fear', 0):.3f}, joy={emotion.get('joy', 0):.3f}"
            )
            lines.append(
                f"  Sentiment: polarity={sentiment.get('polarity', 0):.3f}, subjectivity={sentiment.get('subjectivity', 0):.3f}"
            )
            lines.append(
                f"  Readability: Flesch={readability.get('flesch_score', 0):.1f}, avg sentence length={readability.get('avg_sentence_length', 0):.1f}"
            )
            lines.append(
                f"  Structure: pronoun ratio={structure.get('pronoun_ratio', 0):.3f}, noun ratio={structure.get('noun_ratio', 0):.3f}, named entities={structure.get('named_entities_count', 0)}"
            )
            lines.append(f"  Combined insight: {linguistic.get('combined_insight', '')}")

        return "\n".join(lines)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: Check API connection
    print("=" * 70)
    print("EXAMPLE 1: Health Check")
    print("=" * 70)

    client = NewsClassificationClient()

    if client.health_check():
        print("✓ API server is running")
    else:
        print("✗ API server is not reachable. Start it first:")
        print("  python lime_api_server.py")
        exit(1)

    # Example 2: Get API info
    print("\n" + "=" * 70)
    print("EXAMPLE 2: API Information")
    print("=" * 70)

    info = client.get_info()
    print(f"API Version: {info.get('version')}")
    print(f"Model: {info.get('model')}")
    print(f"Device: {info.get('device')}")

    # Example 3: Single prediction
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Single Prediction")
    print("=" * 70)

    sample_text = """
    Breaking News: Scientists announce breakthrough in renewable energy technology.
    The team has developed a more efficient solar panel that can generate electricity
    even on cloudy days. This could revolutionize the energy industry.
    """

    try:
        result = client.predict(sample_text, threshold=0.5)
        print(f"\n✓ Prediction successful:")
        print(f"  Prediction: {result.prediction}")
        print(f"  Confidence: {result.confidence:.1%}")
        print(f"  True news prob: {result.prob_true:.1%}")
        print(f"  Fake news prob: {result.prob_fake:.1%}")

        if result.explanation and 'features' in result.explanation:
            print(f"\n  Top 5 influential features:")
            for feature in result.explanation['features'][:5]:
                direction = "→ FAKE" if feature['direction'] == 'supports_fake' else "→ TRUE"
                print(f"    • \"{feature['feature']:<40}\" {feature['weight']:>8.4f}  {direction}")

    except Exception as e:
        print(f"✗ Prediction failed: {e}")

    # Example 4: Batch prediction
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Batch Prediction")
    print("=" * 70)

    sample_texts = [
        "Major government scandal uncovered by investigators.",
        "COVID vaccine contains microchip designed by aliens.",
        "New study shows benefits of Mediterranean diet for heart health."
    ]

    try:
        results = client.predict_batch(sample_texts, threshold=0.5)
        print(f"\n✓ Batch prediction successful ({len(results)} samples):")
        for i, result in enumerate(results, 1):
            print(f"\n  Sample {i}: {result.prediction} ({result.confidence:.1%})")

    except Exception as e:
        print(f"✗ Batch prediction failed: {e}")
