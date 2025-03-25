FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install optional dependencies
RUN pip install --no-cache-dir \
    PyPDF2 \
    sentence-transformers \
    rouge-score \
    python-dotenv

# Create directories
RUN mkdir -p /app/data/lancedb

# Copy application code
COPY . .

# Create a non-root user to run the application
RUN useradd -m appuser
RUN chown -R appuser:appuser /app
USER appuser

# Environment variables will be provided at runtime through .env file or docker run

EXPOSE 7860

# Default command runs the demo with provided environment variables
CMD ["python", "app.py"]
