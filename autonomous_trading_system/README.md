# Autonomous Trading System

A sophisticated, modular trading platform designed to maximize dollar profit through advanced market data analysis, dynamic timeframe selection, intelligent position sizing, and optimized trade execution.

## Overview

The Autonomous Trading System (ATS) is a comprehensive platform that leverages state-of-the-art machine learning models and real-time market data to make trading decisions across multiple timeframes and asset classes. The system is built with a modular architecture that enables continuous learning and adaptation to changing market conditions.

## Key Features

- **Multi-timeframe Analysis**: Analyzes data across multiple timeframes (1-min, 5-min, 15-min, hourly, daily)
- **Dollar Profit Optimization**: Optimizes for absolute dollar profit rather than percentage returns
- **Dynamic Ticker Selection**: Selects the most promising tickers from a large universe
- **Risk-Based Position Sizing**: Sizes positions based on risk parameters (2% risk rule)
- **Adaptive Exit Timing**: Identifies optimal exit points based on market conditions
- **GPU-Accelerated Model Training**: Leverages NVIDIA GPUs for high-performance model training
- **Comprehensive Monitoring**: Tracks system performance and health with Prometheus and Grafana
- **Continuous Learning**: Adapts to changing market conditions through automated retraining
- **Automated Deployment**: Ensures consistent and reliable deployment with CI/CD pipeline

## System Architecture

The system follows a modular architecture with seven primary subsystems:

1. **Data Acquisition & Processing**: Gathers and validates market data from various sources
2. **Feature Engineering & Storage**: Calculates and stores features from raw market data
3. **Model Training & Inference**: Trains and manages machine learning models
4. **Trading Strategy & Execution**: Executes trades based on model predictions
5. **Monitoring & Analytics**: Tracks system performance and health
6. **Continuous Learning & Adaptation**: Adapts the system to changing market conditions
7. **CI/CD Pipeline**: Automates building, testing, and deployment of the system

## Installation

### Prerequisites

- Python 3.8+
- Docker and Docker Compose
- NVIDIA GPU with CUDA support (preferably NVIDIA GH200 Grace Hopper Superchip)
- 32+ GB RAM
- 1+ TB SSD storage
- High-speed internet connection

### API Requirements

- Polygon.io API key
- Unusual Whales API key
- Alpaca API key and secret

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-org/autonomous-trading-system.git
   cd autonomous-trading-system
   ```

2. Set up environment variables:
   ```bash
   cp .env.sample .env
   # Edit the .env file with your API keys and configuration
   ```

3. Install dependencies:
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

4. Set up databases:
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
   
   # Pull and run Redis container
   docker run -d --name redis \
     -p 6379:6379 \
     -v redis_data:/data \
     redis:latest
   ```

5. Set up monitoring:
   ```bash
   # Start Prometheus and Grafana using Docker Compose
   cd deployment/monitoring
   docker-compose up -d
   ```

## Usage

### Starting the System

Start the complete system using Docker Compose:
```bash
docker-compose up -d
```

Or start individual components:
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

### Checking System Status

```bash
# Check the status of all system components
python -m src.scripts.check_system_status
```

### Running Backtests

```bash
# Run a backtest with default parameters
python -m src.scripts.run_backtest

# Run a backtest with custom parameters
python -m src.scripts.run_backtest --tickers AAPL,MSFT,GOOGL --timeframe 5m --start-date 2023-01-01 --end-date 2023-01-31
```

### Generating Performance Reports

```bash
# Generate a performance report
python -m src.scripts.generate_performance_report
```

## Documentation

The system includes comprehensive documentation:

- **Documentation Guide**: [docs/00_autonomous_trading_system_documentation_guide.md](docs/00_autonomous_trading_system_documentation_guide.md)
- **System Architecture**: [docs/01_autonomous_trading_system_architecture.md](docs/01_autonomous_trading_system_architecture.md)
- **Codebase Structure**: [docs/02_autonomous_trading_system_codebase_structure.md](docs/02_autonomous_trading_system_codebase_structure.md)
- **System Workflow**: [docs/03_autonomous_trading_system_workflow.md](docs/03_autonomous_trading_system_workflow.md)
- **Component Reference**: [docs/04_autonomous_trading_system_component_reference.md](docs/04_autonomous_trading_system_component_reference.md)

Additional documentation is available for each subsystem and specific topics like backtesting, emergency procedures, and API endpoints.

## Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pip install pre-commit
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific tests
pytest tests/unit/data_acquisition/
pytest tests/integration/trading_strategy/
```

### CI/CD Pipeline

The system uses GitHub Actions for CI/CD. The workflow files are located in `.github/workflows/`.

## Technologies Used

- **Programming Languages**: Python, SQL
- **Machine Learning**: XGBoost, TensorFlow, PyTorch
- **Data Storage**: TimescaleDB, Redis
- **Monitoring**: Prometheus, Grafana
- **Containerization**: Docker, Kubernetes
- **CI/CD**: GitHub Actions
- **APIs**: Polygon.io, Unusual Whales, Alpaca

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational and research purposes only. It is not intended to be used for live trading without proper risk management and oversight. The authors and contributors are not responsible for any financial losses incurred from using this system.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request