l# Configuration Reference

This document provides a comprehensive reference for all configuration options in the Autonomous Trading System.

## Configuration Structure

The system uses a hierarchical configuration structure with multiple configuration files located in the `src/config` directory:

```
src/config/
├── system_config.py       # System-wide configuration
├── logging_config.py      # Logging configuration
├── api_config.py          # API keys and endpoints
├── database_config.py     # Database connection settings
├── trading_config.py      # Trading parameters
├── model_config.py        # ML model parameters
├── monitoring_config.py   # Monitoring and alerting
├── ci_cd_config.py        # CI/CD pipeline settings
├── connections.py         # External service connections
├── service_discovery.py   # Service discovery settings
├── network_config.py      # Network settings
└── hardware_config.py     # Hardware resource allocation
```

## Environment Variables

The system uses environment variables for sensitive configuration and deployment-specific settings:

| Variable | Description | Default |
|----------|-------------|---------|
| `ATS_ENV` | Environment (development, staging, production) | development |
| `ATS_LOG_LEVEL` | Logging level | INFO |
| `ATS_DB_HOST` | Database host | localhost |
| `ATS_DB_PORT` | Database port | 5432 |
| `ATS_DB_NAME` | Database name | ats_db |
| `ATS_DB_USER` | Database username | ats_user |
| `ATS_DB_PASSWORD` | Database password | None |
| `ATS_REDIS_HOST` | Redis host | localhost |
| `ATS_REDIS_PORT` | Redis port | 6379 |
| `ATS_REDIS_PASSWORD` | Redis password | None |
| `POLYGON_API_KEY` | Polygon.io API key | None |
| `UNUSUAL_WHALES_API_KEY` | Unusual Whales API key | None |
| `ALPACA_API_KEY` | Alpaca API key | None |
| `ALPACA_API_SECRET` | Alpaca API secret | None |
| `SLACK_WEBHOOK_URL` | Slack webhook URL for notifications | None |

## System Configuration

The `system_config.py` file contains system-wide configuration:

```python
# From src/config/system_config.py
SYSTEM_CONFIG = {
    "name": "Autonomous Trading System",
    "version": "0.1.0",
    "environment": os.getenv("ATS_ENV", "development"),
    "timezone": "UTC",
    "max_workers": 8,
    "heartbeat_interval_seconds": 30,
    "health_check_interval_seconds": 60,
    "shutdown_timeout_seconds": 30,
    "startup_delay_seconds": 5,
    "component_dependencies": {
        "data_acquisition": [],
        "feature_engineering": ["data_acquisition"],
        "model_training": ["feature_engineering"],
        "trading_strategy": ["model_training"],
        "monitoring": []
    }
}
```

## Logging Configuration

The `logging_config.py` file configures the logging system:

```python
# From src/config/logging_config.py
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
        "json": {
            "format": "json",
            "class": "autonomous_trading_system.src.utils.logging.log_formatter.JsonFormatter"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": os.getenv("ATS_LOG_LEVEL", "INFO"),
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "INFO",
            "formatter": "json",
            "filename": "logs/ats.log",
            "maxBytes": 10485760,  # 10 MB
            "backupCount": 10
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": os.getenv("ATS_LOG_LEVEL", "INFO"),
            "propagate": True
        }
    }
}
```

## API Configuration

The `api_config.py` file contains API keys and endpoints:

```python
# From src/config/api_config.py
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
POLYGON_BASE_URL = "https://api.polygon.io"
POLYGON_WS_URL = "wss://socket.polygon.io"

UNUSUAL_WHALES_API_KEY = os.getenv("UNUSUAL_WHALES_API_KEY")
UNUSUAL_WHALES_BASE_URL = "https://api.unusualwhales.com"

ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_API_SECRET = os.getenv("ALPACA_API_SECRET")
ALPACA_BASE_URL = "https://api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"
ALPACA_PAPER_TRADING = True  # Set to False for live trading
```

## Database Configuration

The `database_config.py` file configures database connections:

```python
# From src/config/database_config.py
TIMESCALEDB_CONFIG = {
    "host": os.getenv("ATS_DB_HOST", "localhost"),
    "port": int(os.getenv("ATS_DB_PORT", "5432")),
    "database": os.getenv("ATS_DB_NAME", "ats_db"),
    "user": os.getenv("ATS_DB_USER", "ats_user"),
    "password": os.getenv("ATS_DB_PASSWORD", ""),
    "min_connections": 5,
    "max_connections": 20,
    "connection_timeout": 30,
    "application_name": "ats",
    "sslmode": "prefer"
}

REDIS_CONFIG = {
    "host": os.getenv("ATS_REDIS_HOST", "localhost"),
    "port": int(os.getenv("ATS_REDIS_PORT", "6379")),
    "password": os.getenv("ATS_REDIS_PASSWORD", ""),
    "db": 0,
    "decode_responses": True,
    "socket_timeout": 5,
    "socket_connect_timeout": 5,
    "retry_on_timeout": True,
    "health_check_interval": 30
}
```

## Trading Configuration

The `trading_config.py` file contains trading parameters:

```python
# From src/config/trading_config.py
TRADING_CONFIG = {
    "trading_hours": {
        "start": "09:30",  # Eastern Time
        "end": "16:00"     # Eastern Time
    },
    "extended_hours_trading": False,
    "max_positions": 20,
    "max_position_size_usd": 5000.0,
    "max_position_size_percentage": 0.05,  # 5% of portfolio
    "risk_per_trade": 0.01,  # 1% of portfolio
    "stop_loss_percentage": 0.02,  # 2% stop loss
    "take_profit_percentage": 0.06,  # 6% take profit
    "trailing_stop_activation_percentage": 0.03,  # Activate at 3% profit
    "trailing_stop_distance_percentage": 0.02,  # 2% trailing distance
    "order_types": {
        "entry": "market",  # market, limit
        "exit": "market",   # market, limit, stop, stop_limit
        "stop_loss": "stop"  # stop, stop_limit
    },
    "position_sizing_method": "risk_based",  # risk_based, fixed, kelly, portfolio_optimization
    "universe_selection": {
        "method": "market_cap",  # market_cap, liquidity, sector, custom
        "max_symbols": 100,
        "min_market_cap": 1000000000,  # $1B
        "min_avg_volume": 500000,
        "sectors": ["Technology", "Healthcare", "Consumer Cyclical", "Financial Services"],
        "refresh_frequency": "daily"  # daily, weekly
    }
}
```

## Model Configuration

The `model_config.py` file configures machine learning models:

```python
# From src/config/model_config.py
MODEL_CONFIG = {
    "default_model": "xgboost",
    "models": {
        "xgboost": {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
            "objective": "binary:logistic",
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0,
            "min_child_weight": 1
        },
        "lstm": {
            "units": 64,
            "dropout": 0.2,
            "recurrent_dropout": 0.2,
            "activation": "tanh",
            "recurrent_activation": "sigmoid",
            "return_sequences": False,
            "sequence_length": 10
        },
        "attention": {
            "num_heads": 4,
            "key_dim": 32,
            "dropout": 0.1,
            "sequence_length": 20
        },
        "ensemble": {
            "models": ["xgboost", "lstm"],
            "weights": [0.7, 0.3],
            "aggregation_method": "weighted_average"  # weighted_average, voting, stacking
        }
    },
    "training": {
        "batch_size": 64,
        "epochs": 50,
        "validation_split": 0.2,
        "early_stopping_patience": 10,
        "learning_rate_reduction_patience": 5,
        "learning_rate_reduction_factor": 0.5,
        "shuffle": True,
        "use_gpu": True,
        "mixed_precision": True
    },
    "validation": {
        "method": "walk_forward",  # walk_forward, cross_validation
        "n_splits": 5,
        "test_size": 0.2,
        "gap": 0  # Time gap between train and test sets
    },
    "hyperparameter_tuning": {
        "method": "bayesian",  # grid, random, bayesian
        "n_trials": 50,
        "n_jobs": -1,
        "cv": 3,
        "scoring": "dollar_profit"  # custom scoring function
    },
    "feature_importance": {
        "method": "shap",  # shap, permutation
        "n_repeats": 10
    }
}
```

## Monitoring Configuration

The `monitoring_config.py` file configures monitoring and alerting:

```python
# From src/config/monitoring_config.py
MONITORING_CONFIG = {
    "metrics_collection_interval": 60,  # seconds
    "system_metrics": {
        "cpu_usage": True,
        "memory_usage": True,
        "disk_usage": True,
        "network_io": True,
        "process_stats": True,
        "gpu_usage": True
    },
    "trading_metrics": {
        "pnl": True,
        "win_rate": True,
        "sharpe_ratio": True,
        "max_drawdown": True,
        "position_count": True,
        "exposure": True
    },
    "model_metrics": {
        "prediction_accuracy": True,
        "feature_importance": True,
        "inference_time": True,
        "confidence_distribution": True
    },
    "data_pipeline_metrics": {
        "data_freshness": True,
        "data_completeness": True,
        "processing_time": True,
        "error_rate": True
    },
    "alerting": {
        "channels": {
            "slack": {
                "enabled": True, # Only Slack notifications are used
                "webhook_url": os.getenv("SLACK_WEBHOOK_URL")
            },
            "sms": {
                "enabled": False,
                "recipients": [os.getenv("EMERGENCY_CONTACT_PHONE")]
            }
        },
        "thresholds": {
            "system": {
                "cpu_usage": 90,  # percentage
                "memory_usage": 90,  # percentage
                "disk_usage": 90,  # percentage
                "error_rate": 0.01  # 1% of requests
            },
            "trading": {
                "daily_loss": 0.05,  # 5% of portfolio
                "drawdown": 0.15,  # 15% drawdown
                "exposure": 0.8  # 80% of portfolio
            },
            "model": {
                "accuracy_drop": 0.1,  # 10% drop in accuracy
                "inference_time": 1.0  # 1 second
            },
            "data": {
                "freshness": 300,  # 5 minutes
                "completeness": 0.95  # 95% complete
            }
        }
    },
    "emergency_stop": {
        "enabled": True,
        "triggers": {
            "daily_loss": 0.07,  # 7% of portfolio
            "drawdown": 0.2,  # 20% drawdown
            "error_rate": 0.05,  # 5% of requests
            "data_freshness": 600  # 10 minutes
        },
        "actions": {
            "close_all_positions": True,
            "stop_trading": True,
            "notify_slack": True
        }
    }
}
```

## Hardware Configuration

The `hardware_config.py` file configures hardware resource allocation:

```python
# From src/config/hardware_config.py
HARDWARE_CONFIG = {
    "cpu": {
        "data_acquisition": 2,  # CPU cores
        "feature_engineering": 4,
        "model_training": 8,
        "trading_strategy": 4,
        "monitoring": 2
    },
    "memory": {
        "data_acquisition": "2Gi",
        "feature_engineering": "4Gi",
        "model_training": "8Gi",
        "trading_strategy": "4Gi",
        "monitoring": "2Gi"
    },
    "gpu": {
        "enabled": True,
        "model_training": 1,  # Number of GPUs
        "inference": 0.5  # Fractional GPU allocation
    },
    "disk": {
        "data_acquisition": "20Gi",
        "feature_engineering": "20Gi",
        "model_training": "50Gi",
        "trading_strategy": "10Gi",
        "monitoring": "50Gi"
    },
    "network": {
        "data_acquisition": {
            "ingress": "100m",  # 100 Mbps
            "egress": "10m"
        },
        "trading_strategy": {
            "ingress": "50m",
            "egress": "50m"
        }
    }
}
```

## Configuration Loading

The system uses a centralized configuration loader that handles environment-specific overrides:

```python
# From src/config/config_loader.py
def load_config(config_name):
    """Load configuration with environment-specific overrides.
    
    Args:
        config_name: Name of the configuration to load
        
    Returns:
        Merged configuration dictionary
    """
    # Load base configuration
    base_config = importlib.import_module(f"autonomous_trading_system.src.config.{config_name}")
    
    # Get environment
    env = os.getenv("ATS_ENV", "development")
    
    # Try to load environment-specific configuration
    try:
        env_config = importlib.import_module(f"autonomous_trading_system.src.config.{env}.{config_name}")
        # Merge configurations
        return deep_merge(base_config, env_config)
    except ImportError:
        # No environment-specific configuration found
        return base_config
```

## Configuration Validation

The system validates configuration at startup to ensure all required settings are present and valid:

```python
# From src/config/config_validator.py
def validate_config():
    """Validate all configuration settings.
    
    Raises:
        ConfigValidationError: If any configuration is invalid
    """
    # Validate system config
    validate_system_config()
    
    # Validate API config
    validate_api_config()
    
    # Validate database config
    validate_database_config()
    
    # Validate trading config
    validate_trading_config()
    
    # Validate model config
    validate_model_config()
    
    # Validate monitoring config
    validate_monitoring_config()
```

## Configuration Updates

The system supports dynamic configuration updates for certain parameters:

```python
# From src/config/config_manager.py
class ConfigManager:
    """Manager for dynamic configuration updates."""
    
    def __init__(self):
        """Initialize the configuration manager."""
        self.configs = {}
        self.listeners = {}
        
    def register_listener(self, config_name, listener):
        """Register a listener for configuration changes.
        
        Args:
            config_name: Name of the configuration to listen for
            listener: Callback function to call when configuration changes
        """
        if config_name not in self.listeners:
            self.listeners[config_name] = []
        self.listeners[config_name].append(listener)
        
    def update_config(self, config_name, updates):
        """Update configuration and notify listeners.
        
        Args:
            config_name: Name of the configuration to update
            updates: Dictionary of updates to apply
        """
        # Apply updates
        if config_name not in self.configs:
            self.configs[config_name] = {}
        deep_update(self.configs[config_name], updates)
        
        # Notify listeners
        if config_name in self.listeners:
            for listener in self.listeners[config_name]:
                listener(self.configs[config_name])
```

## Conclusion

This configuration reference provides a comprehensive overview of all configuration options in the Autonomous Trading System. The system's modular configuration structure allows for flexible customization while maintaining sensible defaults. Environment-specific overrides and dynamic configuration updates enable the system to adapt to different deployment environments and changing requirements.