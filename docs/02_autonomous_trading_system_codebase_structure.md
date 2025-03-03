# Autonomous Trading System Codebase Structure

## Overview

This document outlines the complete directory structure of the Autonomous Trading System (ATS) codebase, providing a clear map of how the system is organized. It includes detailed descriptions of each directory and file, along with setup instructions for getting the system up and running.

The ATS codebase follows a modular, component-based architecture that separates concerns and promotes maintainability. Each subsystem is contained in its own directory with clear interfaces to other components.

## Directory Structure

```
autonomous_trading_system/
├── src/                           # Source code
│   ├── config/                    # Configuration files
│   ├── data_acquisition/          # Data acquisition components
│   ├── feature_engineering/       # Feature engineering components
│   ├── model_training/            # Model training components
│   ├── trading_strategy/          # Trading strategy components
│   ├── monitoring/                # Monitoring components
│   ├── continuous_learning/       # Continuous learning components
│   ├── utils/                     # Utility functions
│   └── scripts/                   # Utility scripts
├── tests/                         # Tests
├── docs/                          # Documentation
├── deployment/                    # Deployment configurations
│   ├── docker/                    # Docker configurations
│   ├── kubernetes/                # Kubernetes configurations
│   └── monitoring/                # Monitoring configurations
├── notebooks/                     # Jupyter notebooks for analysis
└── tools/                         # Development tools
```

## Source Code Structure (`src/`)

### Configuration (`src/config/`)

| File | Description |
|------|-------------|
| `system_config.py` | Main system configuration |
| `logging_config.py` | Logging configuration |
| `api_config.py` | API configuration (Polygon, Unusual Whales, Alpaca) |
| `database_config.py` | Database configuration (TimescaleDB, Redis) |
| `trading_config.py` | Trading configuration |
| `model_config.py` | Model configuration |
| `monitoring_config.py` | Monitoring configuration |

### Data Acquisition (`src/data_acquisition/`)

| File/Directory | Description |
|----------------|-------------|
| `api/` | API client implementations |
| `api/polygon_client.py` | Polygon.io API client |
| `api/unusual_whales_client.py` | Unusual Whales API client |
| `api/alpaca_market_data_client.py` | Alpaca market data client |
| `collectors/` | Data collector implementations |
| `collectors/price_collector.py` | Price data collector |
| `collectors/quote_collector.py` | Quote data collector |
| `collectors/trade_collector.py` | Trade data collector |
| `collectors/options_collector.py` | Options data collector |
| `validation/` | Data validation components |
| `validation/data_validator.py` | Data validator |
| `validation/validation_rules.py` | Validation rules |
| `pipeline/` | Data pipeline components |
| `pipeline/data_pipeline.py` | Main data pipeline |
| `pipeline/pipeline_scheduler.py` | Pipeline scheduler |
| `storage/` | Data storage components |
| `storage/timescale_manager.py` | TimescaleDB manager |
| `storage/data_schema.py` | Database schema definitions |

### Feature Engineering (`src/feature_engineering/`)

| File/Directory | Description |
|----------------|-------------|
| `calculators/` | Feature calculator implementations |
| `calculators/price_features.py` | Price-based feature calculators |
| `calculators/volume_features.py` | Volume-based feature calculators |
| `calculators/volatility_features.py` | Volatility feature calculators |
| `calculators/momentum_features.py` | Momentum feature calculators |
| `calculators/trend_features.py` | Trend feature calculators |
| `calculators/pattern_features.py` | Pattern recognition feature calculators |
| `calculators/microstructure_features.py` | Market microstructure feature calculators |
| `store/` | Feature store components |
| `store/feature_store.py` | Feature store implementation |
| `store/feature_registry.py` | Feature registry |
| `store/feature_cache.py` | Feature cache with Redis |
| `analysis/` | Feature analysis components |
| `analysis/feature_importance.py` | Feature importance analyzer |
| `analysis/feature_correlation.py` | Feature correlation analyzer |
| `pipeline/` | Feature pipeline components |
| `pipeline/feature_pipeline.py` | Feature calculation pipeline |
| `pipeline/multi_timeframe_processor.py` | Multi-timeframe feature processor |

### Model Training (`src/model_training/`)

| File/Directory | Description |
|----------------|-------------|
| `models/` | Model implementations |
| `models/xgboost_model.py` | XGBoost model implementation |
| `models/lstm_model.py` | LSTM model implementation |
| `models/attention_model.py` | Attention model implementation |
| `models/ensemble_model.py` | Ensemble model implementation |
| `optimization/` | Optimization components |
| `optimization/dollar_profit_objective.py` | Dollar profit objective function |
| `optimization/hyperparameter_tuner.py` | Hyperparameter tuner |
| `optimization/mixed_precision_adapter.py` | Mixed precision adapter for GPU acceleration |
| `validation/` | Validation components |
| `validation/cross_timeframe_validator.py` | Cross-timeframe validator |
| `validation/walk_forward_validator.py` | Walk-forward validator |
| `registry/` | Model registry components |
| `registry/model_registry.py` | Model registry implementation |
| `registry/model_metadata.py` | Model metadata definitions |
| `inference/` | Inference components |
| `inference/model_server.py` | Model serving infrastructure |
| `inference/prediction_confidence.py` | Prediction confidence calculator |

### Trading Strategy (`src/trading_strategy/`)

| File/Directory | Description |
|----------------|-------------|
| `selection/` | Selection components |
| `selection/ticker_selector.py` | Dynamic ticker selector |
| `selection/timeframe_selector.py` | Timeframe selector |
| `sizing/` | Position sizing components |
| `sizing/position_sizer.py` | Risk-based position sizer |
| `sizing/portfolio_allocator.py` | Portfolio allocator |
| `execution/` | Execution components |
| `execution/order_generator.py` | Smart order generator |
| `execution/order_type_selector.py` | Order type selector |
| `execution/execution_quality_analyzer.py` | Execution quality analyzer |
| `risk/` | Risk management components |
| `risk/stop_loss_manager.py` | Adaptive stop-loss manager |
| `risk/profit_target_manager.py` | Dynamic profit target manager |
| `risk/portfolio_risk_manager.py` | Portfolio-level risk manager |
| `alpaca/` | Alpaca integration components |
| `alpaca/alpaca_client.py` | Alpaca API client |
| `alpaca/alpaca_trade_executor.py` | Alpaca trade executor |
| `alpaca/alpaca_position_manager.py` | Alpaca position manager |
| `alpaca/alpaca_integration.py` | High-level Alpaca integration |
| `signals/` | Signal generation components |
| `signals/peak_detector.py` | Peak detector for exit timing |
| `signals/entry_signal_generator.py` | Entry signal generator |
| `signals/holding_period_optimizer.py` | Holding period optimizer |

### Monitoring (`src/monitoring/`)

| File/Directory | Description |
|----------------|-------------|
| `collectors/` | Metrics collector implementations |
| `collectors/system_metrics_collector.py` | System metrics collector |
| `collectors/trading_metrics_collector.py` | Trading metrics collector |
| `collectors/model_metrics_collector.py` | Model metrics collector |
| `collectors/data_pipeline_metrics_collector.py` | Data pipeline metrics collector |
| `exporters/` | Metrics exporter implementations |
| `exporters/prometheus_exporter.py` | Prometheus exporter |
| `exporters/timescaledb_exporter.py` | TimescaleDB exporter |
| `alerting/` | Alerting components |
| `alerting/slack_notifier.py` | Slack notifier |
| `alerting/alert_manager.py` | Alert manager |
| `dashboard/` | Dashboard components |
| `dashboard/app.py` | Dashboard application |
| `dashboard/performance_dashboard.py` | Performance dashboard |
| `dashboard/system_dashboard.py` | System dashboard |
| `analysis/` | Analysis components |
| `analysis/dollar_profit_analyzer.py` | Dollar profit analyzer |
| `analysis/performance_analyzer.py` | Performance analyzer |
| `analysis/attribution_analyzer.py` | Attribution analyzer |
| `monitoring_manager.py` | Monitoring manager |

### Continuous Learning (`src/continuous_learning/`)

| File/Directory | Description |
|----------------|-------------|
| `analysis/` | Analysis components |
| `analysis/performance_analyzer.py` | Performance analyzer |
| `analysis/market_regime_detector.py` | Market regime detector |
| `retraining/` | Retraining components |
| `retraining/model_retrainer.py` | Model retrainer |
| `retraining/retraining_scheduler.py` | Retraining scheduler |
| `adaptation/` | Adaptation components |
| `adaptation/parameter_tuner.py` | Adaptive parameter tuner |
| `adaptation/ensemble_weighter.py` | Ensemble weighter |
| `pipeline/` | Pipeline components |
| `pipeline/continuous_learning_pipeline.py` | Continuous learning pipeline |
| `pipeline/feature_importance_tracker.py` | Feature importance tracker |

### Utilities (`src/utils/`)

| File/Directory | Description |
|----------------|-------------|
| `database/` | Database utilities |
| `database/timescaledb_utils.py` | TimescaleDB utilities |
| `database/redis_utils.py` | Redis utilities |
| `logging/` | Logging utilities |
| `logging/logger.py` | Logger implementation |
| `logging/log_formatter.py` | Log formatter |
| `concurrency/` | Concurrency utilities |
| `concurrency/thread_pool.py` | Thread pool implementation |
| `concurrency/process_pool.py` | Process pool implementation |
| `concurrency/distributed_lock.py` | Distributed lock implementation |
| `serialization/` | Serialization utilities |
| `serialization/json_serializer.py` | JSON serializer |
| `serialization/pickle_serializer.py` | Pickle serializer |
| `time/` | Time utilities |
| `time/market_calendar.py` | Market calendar |
| `time/time_utils.py` | Time utilities |
| `metrics/` | Metrics utilities |
| `metrics/performance_metrics.py` | Performance metrics |
| `metrics/system_metrics.py` | System metrics |

### Scripts (`src/scripts/`)

| File | Description |
|------|-------------|
| `run_data_acquisition.py` | Script to run data acquisition |
| `run_feature_engineering.py` | Script to run feature engineering |
| `run_model_training.py` | Script to run model training |
| `run_trading_strategy.py` | Script to run trading strategy |
| `run_monitoring.py` | Script to run monitoring |
| `run_continuous_learning.py` | Script to run continuous learning |
| `system_controller.py` | System controller script |

## Tests Structure (`tests/`)

| Directory | Description |
|-----------|-------------|
| `unit/` | Unit tests |
| `integration/` | Integration tests |
| `system/` | System tests |
| `performance/` | Performance tests |
| `fixtures/` | Test fixtures |

## Documentation Structure (`docs/`)

| File | Description |
|------|-------------|
| `01_autonomous_trading_system_architecture.md` | System architecture overview |
| `02_autonomous_trading_system_codebase_structure.md` | Codebase structure and setup guide |
| `03_autonomous_trading_system_workflow.md` | System workflow diagrams |
| `04_autonomous_trading_system_component_reference.md` | Component reference |
| `05_data_acquisition_subsystem.md` | Data acquisition subsystem documentation |
| `06_feature_engineering_subsystem.md` | Feature engineering subsystem documentation |
| `07_model_training_subsystem.md` | Model training subsystem documentation |
| `08_trading_strategy_subsystem.md` | Trading strategy subsystem documentation |
| `09_monitoring_subsystem.md` | Monitoring subsystem documentation |
| `10_continuous_learning_subsystem.md` | Continuous learning subsystem documentation |
| `11_deployment_guide.md` | Deployment guide |
| `12_operations_guide.md` | Operations guide |
| `13_development_guide.md` | Development guide |
| `14_api_reference.md` | API reference |
| `15_database_schema.md` | Database schema reference |
| `16_configuration_reference.md` | Configuration reference |
| `17_troubleshooting_guide.md` | Troubleshooting guide |
| `18_performance_tuning_guide.md` | Performance tuning guide |
| `19_security_guide.md` | Security guide |
| `20_production_readiness_checklist.md` | Production readiness checklist |

## Deployment Structure (`deployment/`)

### Docker (`deployment/docker/`)

| File | Description |
|------|-------------|
| `Dockerfile` | Main Dockerfile |
| `docker-compose.yml` | Docker Compose configuration |
| `docker-compose.dev.yml` | Development Docker Compose configuration |
| `docker-compose.prod.yml` | Production Docker Compose configuration |
| `.dockerignore` | Docker ignore file |

### Kubernetes (`deployment/kubernetes/`)

| File | Description |
|------|-------------|
| `deployment.yaml` | Kubernetes deployment configuration |
| `service.yaml` | Kubernetes service configuration |
| `configmap.yaml` | Kubernetes ConfigMap configuration |
| `secret.yaml` | Kubernetes Secret configuration |
| `ingress.yaml` | Kubernetes Ingress configuration |
| `pv.yaml` | Kubernetes PersistentVolume configuration |
| `pvc.yaml` | Kubernetes PersistentVolumeClaim configuration |

### Monitoring (`deployment/monitoring/`)

| File/Directory | Description |
|----------------|-------------|
| `prometheus/` | Prometheus configuration |
| `prometheus/prometheus.yml` | Prometheus configuration file |
| `prometheus/rules/` | Prometheus alerting rules |
| `grafana/` | Grafana configuration |
| `grafana/grafana.ini` | Grafana configuration file |
| `grafana/provisioning/` | Grafana provisioning |
| `grafana/provisioning/dashboards/` | Grafana dashboard provisioning |
| `grafana/provisioning/datasources/` | Grafana datasource provisioning |
| `grafana/dashboards/` | Grafana dashboard definitions |
| `grafana/dashboards/system_dashboard.json` | System dashboard definition |
| `grafana/dashboards/trading_dashboard.json` | Trading dashboard definition |
| `grafana/dashboards/model_dashboard.json` | Model dashboard definition |
| `grafana/dashboards/data_pipeline_dashboard.json` | Data pipeline dashboard definition |

## Setup Instructions

### 1. Environment Setup

#### 1.1 Clone the Repository

```bash
git clone https://github.com/your-org/autonomous-trading-system.git
cd autonomous-trading-system
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
TIMESCALEDB_HOST=localhost
TIMESCALEDB_PORT=5432
TIMESCALEDB_DATABASE=ats_db
TIMESCALEDB_USER=ats_user
TIMESCALEDB_PASSWORD=your_password

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Monitoring Configuration
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
SLACK_WEBHOOK_URL=your_slack_webhook_url

# Trading Configuration
MAX_ACTIVE_TICKERS=150
FOCUS_UNIVERSE_SIZE=40
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
  -p 5432:5432 \
  -e POSTGRES_USER=ats_user \
  -e POSTGRES_PASSWORD=your_password \
  -e POSTGRES_DB=ats_db \
  -v timescaledb_data:/var/lib/postgresql/data \
  timescale/timescaledb:latest-pg14

# Create tables and hypertables
python -m src.scripts.setup_database
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

#### 3.1 Set Up Prometheus and Grafana

```bash
# Start Prometheus and Grafana using Docker Compose
cd deployment/monitoring
docker-compose up -d
```

### 4. System Startup

#### 4.1 Start the Complete System

```bash
# Start the complete system using Docker Compose
docker-compose up -d
```

#### 4.2 Start Individual Components

```bash
# Start data acquisition
python -m src.scripts.run_data_acquisition

# Start feature engineering
python -m src.scripts.run_feature_engineering

# Start model training
python -m src.scripts.run_model_training

# Start trading strategy
python -m src.scripts.run_trading_strategy

# Start monitoring
python -m src.scripts.run_monitoring

# Start continuous learning
python -m src.scripts.run_continuous_learning
```

#### 4.3 Check System Status

```bash
# Check the status of all system components
python -m src.scripts.check_system_status
```

## Development Setup

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
pytest tests/unit/data_acquisition/
pytest tests/integration/trading_strategy/
```

## Conclusion

This document provides a comprehensive overview of the Autonomous Trading System codebase structure and setup instructions. The modular architecture allows for easy maintenance and extension of the system, while the clear separation of concerns ensures that each component can be developed and tested independently.

For more detailed information about specific components, refer to the documentation in the `docs/` directory.