# ============================================================
# Dockerfile — Fake News Detector API (Flask + RoBERTa + LIME)
# ============================================================

# Use official Python slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements-docker.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-docker.txt


# Copy application source code
COPY lime_api_server.py .

# Expose the API port
EXPOSE 5001

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5001/health')" || exit 1

# Run the Flask API server
CMD ["python", "lime_api_server.py", "--host", "0.0.0.0", "--port", "5001"]
