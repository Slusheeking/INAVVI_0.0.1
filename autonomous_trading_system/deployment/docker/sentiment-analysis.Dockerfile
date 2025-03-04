FROM python:3.10-slim

LABEL maintainer="Autonomous Trading System"
LABEL description="FinBERT Sentiment Analysis for Financial News"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libpq-dev \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CPU support (use CUDA version if GPU is available)
RUN pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

# Install dependencies with fixed versions to avoid build issues
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    transformers==4.35.2 \
    sqlalchemy==2.0.23 \
    psycopg2-binary==2.9.9 \
    pandas==2.1.3 \
    numpy==1.26.2 \
    scipy==1.11.4 \
    tqdm==4.66.1 \
    redis==5.0.1 \
    python-dotenv==1.0.0 \
    loguru==0.7.2 \
    nltk==3.8.1

# Install spaCy separately with specific version to avoid build issues
RUN pip install --no-cache-dir spacy==3.7.2 --no-build-isolation

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy requirements files
COPY requirements-sentiment.txt .
COPY requirements-dev.txt .

# Create logs directory
RUN mkdir -p /app/logs

# Install dependencies
RUN pip install --no-cache-dir -r requirements-sentiment.txt

# Copy source code
COPY . /app

# Set Python path
ENV PYTHONPATH=/app

# Set environment variables
ENV TIMESCALEDB_HOST=timescaledb
ENV TIMESCALEDB_PORT=5432
ENV TIMESCALEDB_USER=ats_user
ENV TIMESCALEDB_PASSWORD=your_database_password_here
ENV TIMESCALEDB_DATABASE=ats_db
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV USE_GPU=false

# Run the sentiment analysis script with default parameters
CMD ["python", "src/scripts/run_sentiment_analysis.py", "--days", "7", "--continuous"]