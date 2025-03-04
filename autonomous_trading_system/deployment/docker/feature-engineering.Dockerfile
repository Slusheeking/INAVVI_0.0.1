# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    libpq-dev \
    gcc

# Install PyTorch with CPU support
RUN pip install --no-cache-dir torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    transformers==4.35.2 \
    sqlalchemy==2.0.23 \
    psycopg2-binary==2.9.9 \
    redis==5.0.1

# Install spaCy and download English model
RUN pip install --no-cache-dir spacy==3.7.2 --no-build-isolation
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

# Set environment variables
ENV PYTHONPATH=/app
ENV LOG_LEVEL=INFO
ENV TIMESCALEDB_HOST=timescaledb
ENV TIMESCALEDB_PORT=5432
ENV TIMESCALEDB_USER=ats_user
ENV TIMESCALEDB_PASSWORD=your_database_password_here
ENV TIMESCALEDB_DATABASE=ats_db
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379

# Set entrypoint
ENTRYPOINT ["python", "src/scripts/run_feature_engineering.py"]

# Default command
CMD ["--days", "7", "--continuous", "--skip-sentiment"]