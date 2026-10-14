# 📰 Fake News Classification with LIME & Linguistic Explanations

This repository contains a production-ready system for real-time fake news detection combining a deep learning model (BERT) with model-agnostic explainability (LIME) and heuristic linguistic analysis.

---

## 🛠️ Components

1. **`lime_api_server.py`**: A Flask API server that loads the fine-tuned BERT model, initializes the LIME explainer, processes texts, and computes linguistic characteristics.
2. **`lime_web_frontend.html`**: An interactive web frontend that sends text to the Flask API, displays predictions, and visualizes word importance (LIME) and linguistic metrics in real time.
3. **`requirements.txt`**: List of Python dependencies needed to run the API.
4. **`.gitignore`**: Excludes large training datasets and trained model weights from Git to comply with GitHub's file size limits.

---

## 🚀 Quick Start Guide

To run the application locally:

### 1. Install Dependencies
Ensure you have Python 3.8+ installed. Open a terminal in the project directory and run:

```bash
pip install -r requirements.txt
```

### 2. Run the API Server
Start the Flask backend by running:

```bash
python lime_api_server.py --port 5001
```

*Note: The frontend is configured to communicate with the API on port **5001**.*

### 3. Open the Frontend
Double-click **`lime_web_frontend.html`** or open it in any web browser. You can now paste news articles, adjust prediction thresholds, and view predictions along with LIME highlights and linguistic analysis!

---

## 📝 Project Details

* **BERT Classification Model**: A sequence classifier fine-tuned on the WELFake dataset to distinguish genuine reporting styles from sensationalized fake news.
* **LIME (Local Interpretable Model-agnostic Explanations)**: Highlights specific words in the text that drove the model's decision (red highlights support "Fake News", green highlights support "True News").
* **Linguistic Analysis**: Computes lightweight, explainable NLP metrics including:
  * **Emotion**: Ratios of anger, fear, and joy (using `nrclex`).
  * **Sentiment & Subjectivity**: Sentiment polarity and objectivity indicators (using `vaderSentiment`).
  * **Readability**: Flesch reading ease score and average sentence length.
  * **Structural Signals**: Pronoun/noun ratios and named entity density.
* **Combined Insight**: A natural language summary explaining the model's prediction alongside linguistic signals.

---

## ⚠️ Repository Notes

* **Model Weights & Datasets**: The model directories (`bert_welfake_model/`, `roberta_welfake_model/`) and dataset CSV files are excluded from this repository via the **`.gitignore`** file because they exceed GitHub's 100MB file size limit. 
* To run the API server locally, ensure you have the fine-tuned model files stored in a directory named `bert_welfake_model` in the root of the project.
