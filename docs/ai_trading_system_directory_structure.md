# AI Trading System Directory Structure and Setup Guide

## Directory Structure Overview

The AI Trading System follows a modular directory structure that organizes code by functionality. This document outlines the complete directory structure and provides setup instructions for each component.

```
ai_trading_system/
├── src/                           # Source code
│   ├── config/                    # Configuration files
│   ├── data_collection/           # Data collection components
│   │   ├── market_data/           # Market data collection
│   │   ├── polygon_client.py      # Polygon.io API client
│   │   ├── unusual_whales_client.py # Unusual Whales API client
│   │   └── production_data_pipeline.py # Production data pipeline
│   ├── model_training/            # Model training components
│   │   ├── feature_engineering.py # Feature engineering
│   │   ├── feature_store.py       # Feature store
│   │   ├── lstm_model.py          # LSTM model
│   │   ├── mixed_precision_adapter.py # Mixed precision adapter
│   │   ├── model_trainer.py       # Model trainer
│   │   └── walk_forward_validation.py # Walk-forward validation
│   ├── trading/                   # Trading components
│   │   ├── alpaca/                # Alpaca integration
│   │   │   ├── alpaca_client.py   # Alpaca API client
│   │   │   ├── alpaca_integration.py # Alpaca integration
│   │   │   └── alpaca_trade_executor.py # Alpaca trade executor
│   │   ├── dollar_profit_optimizer.py # Dollar profit optimizer
│   │   ├── dynamic_ticker_selector.py # Dynamic ticker selector
│   │   ├── peak_detector.py       # Peak detector
│   │   ├── position_sizer.py      # Position sizer
│   │   └── timeframe_selector.py  # Timeframe selector
│   ├── monitoring/                # Monitoring components
│   │   ├── alerting/              # Alerting components
│   │   │   └── slack_notifier.py  # Slack notifier
│   │   ├── collectors/            # Metrics collectors
│   │   │   ├── base_collector.py  # Base collector
│   │   │   ├── data_pipeline_metrics_collector.py # Data pipeline metrics
│   │   │   ├── model_metrics_collector.py # Model metrics
│   │   │   ├── system_metrics_collector.py # System metrics
│   │   │   └── trading_metrics_collector.py # Trading metrics
│   │   ├── dashboard/             # Dashboard components
│   │   │   ├── app.py             # Dashboard app
│   │   │   └── test_dashboard_app.py # Dashboard tests
│   │   ├── exporters/             # Metrics exporters
│   │   │   ├── prometheus_client.py # Prometheus client
│   │   │   └── timescaledb_client.py # TimescaleDB client
│   │   └── monitoring_manager.py  # Monitoring manager
│   ├── utils/                     # Utility functions
│   │   ├── database.py            # Database utilities
│   │   ├── query_timescale_db.py  # TimescaleDB query utilities
│   │   └── redis/                 # Redis utilities
│   │       ├── cache_manager.py   # Redis cache manager
│   │       ├── distributed_lock.py # Distributed lock
│   │       ├── feature_store_cache.py # Feature store cache
│   │       └── pubsub.py          # Pub/sub messaging
│   ├── scripts/                   # Scripts
│   │   ├── fetch_market_data.py   # Fetch market data
│   │   ├── run_alpaca_trading.py  # Run Alpaca trading
│   │   ├── run_grafana_monitoring.py # Run Grafana monitoring
│   │   └── schedule_data_pipeline.py # Schedule data pipeline
│   ├── data_collector.py          # Main data collector
│   └── main.py                    # Main entry point
├── tests/                         # Tests
│   ├── data_collection/           # Data collection tests
│   │   └── test_data_pipeline.py  # Data pipeline tests
│   └── trading/                   # Trading tests
│       └── alpaca/                # Alpaca tests
│           └── test_alpaca_api.py # Alpaca API tests
├── docs/                          # Documentation
│   ├── AI_TRADING_SYSTEM_README.md # Main README
│   ├── alpaca_api_integration_guide.md # Alpaca integration guide
│   ├── data_pipeline_improvements_todo.md # Data pipeline improvements
│   ├── dollar_profit_optimizer_gpu_guide.md # Dollar profit optimizer guide
│   ├── dynamic_ticker_selection_and_position_sizing.md # Ticker selection guide
│   ├── grafana_monitoring_guide.md # Grafana monitoring guide
│   ├── model_training_monitoring.md # Model training monitoring guide
│   ├── redis_implementation_guide.md # Redis implementation guide
│   └── STARTUP_SCRIPTS_README.md # Startup scripts guide
├── grafana_monitoring/            # Grafana monitoring
│   ├── provisioning/              # Grafana provisioning
│   │   ├── dashboards/            # Dashboard provisioning
│   │   │   ├── dashboards.yaml    # Dashboard config
│   │   │   └── trading_system_dashboard.json # Dashboard definition
│   │   └── datasources/           # Datasource provisioning
│   │       └── datasources.yaml   # Datasource config
│   ├── docker-compose.yml         # Docker Compose file
│   ├── grafana.ini                # Grafana config
│   ├── README.md                  # Grafana README
│   ├── run_grafana.sh             # Run Grafana script
│   ├── run_local_grafana.sh       # Run local Grafana script
│   ├── start_grafana_with_ngrok.sh # Start Grafana with ngrok
│   └── stop_grafana_with_ngrok.sh # Stop Grafana with ngrok
├── .env.sample                    # Sample environment variables
├── .dockerignore                  # Docker ignore file
├── Dockerfile                     # Dockerfile
├── requirements.txt               # Python dependencies
├── setup_prometheus.sh            # Setup Prometheus script
├── start_trading_system.sh        # Start trading system script
├── start_trading_system_all.sh    # Start all trading system components
├── stop_trading_system.sh         # Stop trading system script
├── stop_trading_system_all.sh     # Stop all trading system components
├── start_alpaca_trading.sh        # Start Alpaca trading script
├── stop_alpaca_trading.sh         # Stop Alpaca trading script
├── run_alpaca_test.py             # Run Alpaca test script
├── test_alpaca_direct.py          # Test Alpaca direct script
├── run_data_apis_directly.sh      # Run data APIs directly script
├── stop_data_apis.sh              # Stop data APIs script
├── check_system_status.sh         # Check system status script
└── DATA_APIS_README.md            # Data APIs README
```

## Setup Instructions

### 1. Environment Setup

#### 1.1 Clone the Repository

```bash
git clone https://github.com/your-org/ai_trading_system.git
cd ai_trading_system
```

#### 1.2 Set Up Environment Variables

```bash
cp .env.sample .env
```

Edit the `.env` file to include your API keys and configuration:

```
# API Keys
POLYGON_API_KEY=your_polygon_api_key
UNUSUAL_WHALES_API_KEY=your_unusual_whales_api_key
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_API_SECRET=your_alpaca_api_secret

# Database Configuration
DB_HOST=localhost
DB_PORT=5433
DB_NAME=inavvi
DB_USER=inavvi_user
DB_PASSWORD=postgres

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Slack Configuration
SLACK_WEBHOOK_URL=your_slack_webhook_url

# System Configuration
MAX_ACTIVE_TICKERS=150
FOCUS_UNIVERSE_SIZE=40
MIN_VOLUME=500000
MIN_VOLATILITY=0.01
OPPORTUNITY_THRESHOLD=0.5
ACCOUNT_SIZE=5000.0
RISK_PERCENTAGE=0.02
MAX_POSITION_SIZE=2500.0
```

#### 1.3 Install Dependencies

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### 1.4 Set Up NVIDIA CUDA for GPU Acceleration

```bash
# Install NVIDIA drivers
sudo apt-get update
sudo apt-get install -y nvidia-driver-525

# Install CUDA Toolkit
wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda_12.0.0_525.60.13_linux.run
sudo sh cuda_12.0.0_525.60.13_linux.run

# Add CUDA to PATH
echo 'export PATH=/usr/local/cuda-12.0/bin${PATH:+:${PATH}}' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
source ~/.bashrc
```

#### 1.5 Install Docker and Docker Compose

```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.15.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 2. Database Setup

#### 2.1 Set Up TimescaleDB

```bash
# Pull and run TimescaleDB container
docker run -d --name timescaledb \
  -p 5433:5432 \
  -e POSTGRES_USER=inavvi_user \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=inavvi \
  -v timescaledb_data:/var/lib/postgresql/data \
  timescale/timescaledb:latest-pg14

# Create tables and hypertables
python -c "from src.utils.database import DatabaseManager; DatabaseManager().create_tables()"
```

#### 2.2 Set Up Redis

```bash
# Pull and run Redis container
docker run -d --name redis \
  -p 6379:6379 \
  -v redis_data:/data \
  redis:latest
```

### 3. Monitoring Setup

#### 3.1 Set Up Prometheus

```bash
# Run the Prometheus setup script
chmod +x setup_prometheus.sh
./setup_prometheus.sh
```

#### 3.2 Set Up Grafana

```bash
# Run the Grafana setup script
cd grafana_monitoring
chmod +x run_grafana.sh
./run_grafana.sh
```

### 4. System Startup

#### 4.1 Start the Trading System

```bash
# Start the complete trading system
chmod +x start_trading_system_all.sh
./start_trading_system_all.sh
```

#### 4.2 Start Individual Components

```bash
# Start data collection
chmod +x run_data_apis_directly.sh
./run_data_apis_directly.sh

# Start Alpaca trading
chmod +x start_alpaca_trading.sh
./start_alpaca_trading.sh

# Start Grafana monitoring
cd grafana_monitoring
chmod +x start_grafana_with_ngrok.sh
./start_grafana_with_ngrok.sh
```

#### 4.3 Check System Status

```bash
# Check the status of all system components
chmod +x check_system_status.sh
./check_system_status.sh
```

### 5. Development Setup

#### 5.1 Set Up Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pre-commit install
```

#### 5.2 Run Tests

```bash
# Run all tests
pytest

# Run specific tests
pytest tests/data_collection/test_data_pipeline.py
pytest tests/trading/alpaca/test_alpaca_api.py
```

## Component Configuration

### 1. Data Collection Configuration

The data collection components are configured in `src/config/config.py`:

```python
# API Keys
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
UNUSUAL_WHALES_API_KEY = os.getenv("UNUSUAL_WHALES_API_KEY")

# Database Configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5433"),
    "dbname": os.getenv("DB_NAME", "inavvi"),
    "user": os.getenv("DB_USER", "inavvi_user"),
    "password": os.getenv("DB_PASSWORD", "postgres")
}

# Market Hours Configuration
MARKET_HOURS = {
    "open_time": "09:30",
    "close_time": "16:00",
    "timezone": "America/New_York"
}
```

### 2. Trading Configuration

The trading components are configured in `src/config/alpaca_config.py`:

```python
# Alpaca API Configuration
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
ALPACA_DATA_URL = os.getenv("ALPACA_DATA_URL", "https://data.alpaca.markets")
ALPACA_STREAM_URL = os.getenv("ALPACA_STREAM_URL", "wss://stream.data.alpaca.markets")

# Trading Configuration
MAX_ACTIVE_TICKERS = int(os.getenv("MAX_ACTIVE_TICKERS", "150"))
FOCUS_UNIVERSE_SIZE = int(os.getenv("FOCUS_UNIVERSE_SIZE", "40"))
MIN_VOLUME = int(os.getenv("MIN_VOLUME", "500000"))
MIN_VOLATILITY = float(os.getenv("MIN_VOLATILITY", "0.01"))
OPPORTUNITY_THRESHOLD = float(os.getenv("OPPORTUNITY_THRESHOLD", "0.5"))
ACCOUNT_SIZE = float(os.getenv("ACCOUNT_SIZE", "5000.0"))
RISK_PERCENTAGE = float(os.getenv("RISK_PERCENTAGE", "0.02"))
MAX_POSITION_SIZE = float(os.getenv("MAX_POSITION_SIZE", "2500.0"))
```

### 3. Monitoring Configuration

The monitoring components are configured in `grafana_monitoring/grafana.ini` and `grafana_monitoring/provisioning/datasources/datasources.yaml`.

## Docker Deployment

### 1. Build the Docker Image

```bash
docker build -t ai_trading_system:latest .
```

### 2. Run the Docker Container

```bash
docker run -d --name ai_trading_system \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/.env:/app/.env \
  ai_trading_system:latest
```

### 3. Run with Docker Compose

```bash
docker-compose up -d
```

## Troubleshooting

### 1. Database Connection Issues

If you encounter database connection issues:

```bash
# Check if TimescaleDB container is running
docker ps | grep timescaledb

# Check TimescaleDB logs
docker logs timescaledb

# Restart TimescaleDB container
docker restart timescaledb
```

### 2. API Connection Issues

If you encounter API connection issues:

```bash
# Check API keys in .env file
cat .env | grep API_KEY

# Test Polygon API connection
python -c "from src.data_collection.polygon_client import PolygonClient; client = PolygonClient(api_key='your_api_key'); print(client.get_ticker_details('AAPL'))"

# Test Alpaca API connection
python -c "from src.trading.alpaca.alpaca_client import AlpacaClient; client = AlpacaClient(api_key='your_api_key', api_secret='your_api_secret'); print(client.get_account_status())"
```

### 3. GPU Acceleration Issues

If you encounter GPU acceleration issues:

```bash
# Check NVIDIA driver installation
nvidia-smi

# Check CUDA installation
nvcc --version

# Check TensorFlow GPU support
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Conclusion

This document provides a comprehensive overview of the AI Trading System directory structure and setup instructions. By following these instructions, you can set up and run the complete AI Trading System with all its components.

For more detailed information about specific components, refer to the documentation in the `docs/` directory.