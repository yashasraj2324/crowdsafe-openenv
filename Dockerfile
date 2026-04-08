FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY openenv.yaml .
COPY inference.py .

# Create non-root user (HF Spaces best practice)
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port 7860 (HuggingFace Spaces default)
EXPOSE 7860

# Environment defaults (override at runtime)
ENV API_BASE_URL="https://openrouter.ai/api/v1"
ENV MODEL_NAME="meta-llama/llama-3.1-8b-instruct:free"
ENV HF_TOKEN=""

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:7860/health')"

# Start FastAPI server
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
