# syntax=docker/dockerfile:1
# ──────────────────────────────────────────────────────────────
# Multi-stage Dockerfile for Smithy Agentic RAG
# ──────────────────────────────────────────────────────────────

FROM python:3.14-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ──────────────────────────────────────────────────────────────
# Development stage
# ──────────────────────────────────────────────────────────────
FROM base AS dev

# Install optional + dev dependencies
RUN pip install --no-cache-dir \
    PyPDF2 \
    sentence-transformers \
    rouge-score \
    python-dotenv \
    debugpy

# Copy source
COPY . .

# Create data directory and non-root user
RUN mkdir -p /app/data/lancedb \
    && useradd -m appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 7860 5678

CMD ["python", "app.py"]

# ──────────────────────────────────────────────────────────────
# Production stage
# ──────────────────────────────────────────────────────────────
FROM base AS production

# Install only production dependencies
RUN pip install --no-cache-dir \
    PyPDF2 \
    python-dotenv \
    gunicorn

# Copy only necessary application code
COPY agentic_rag/ /app/agentic_rag/
COPY app.py setup.py Makefile /app/

# Create directories and non-root user
RUN mkdir -p /app/data/lancedb \
    && mkdir -p /app/.cache \
    && useradd -m appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --retries=3 --start-period=40s \
    CMD curl -f http://localhost:7860 || exit 1

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--workers", "1", "--timeout", "120", "--access-logfile", "-", "app:app"]
