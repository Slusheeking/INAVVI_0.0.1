"""
Database Configuration Module

This module provides configuration settings and connection utilities for the database components
of the Autonomous Trading System, including TimescaleDB and Redis.
"""

import os
from typing import Dict, Any, Optional
from urllib.parse import quote_plus
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# TimescaleDB Configuration
TIMESCALEDB_CONFIG = {
    'host': os.getenv('TIMESCALEDB_HOST', 'localhost'),
    'port': int(os.getenv('TIMESCALEDB_PORT', '5433')),
    'database': os.getenv('TIMESCALEDB_DATABASE', 'ats_db'),
    'user': os.getenv('TIMESCALEDB_USER', 'ats_user'),
    'password': os.getenv('TIMESCALEDB_PASSWORD', 'your_database_password_here'),
}

# Redis Configuration
REDIS_CONFIG = {
    'host': os.getenv('REDIS_HOST', 'localhost'),
    'port': int(os.getenv('REDIS_PORT', '6379')),
    'db': int(os.getenv('REDIS_DB', '0')),
    'decode_responses': True,
}

# SQLAlchemy Connection String
def get_db_connection_string(config: Optional[Dict[str, Any]] = None) -> str:
    """
    Generate a SQLAlchemy connection string for TimescaleDB.
    
    Args:
        config: Optional database configuration override
        
    Returns:
        SQLAlchemy connection string
    """
    if config is None:
        config = TIMESCALEDB_CONFIG
    
    # Safely encode password for URL
    password = quote_plus(config['password']) if config['password'] else ''
    
    return (
        f"postgresql://{config['user']}:{password}@"
        f"{config['host']}:{config['port']}/{config['database']}"
    )

# Database Pool Configuration
DB_POOL_CONFIG = {
    'pool_size': int(os.getenv('DB_POOL_SIZE', '10')),
    'max_overflow': int(os.getenv('DB_MAX_OVERFLOW', '20')),
    'pool_timeout': int(os.getenv('DB_POOL_TIMEOUT', '30')),
    'pool_recycle': int(os.getenv('DB_POOL_RECYCLE', '1800')),
}

# TimescaleDB Hypertable Configuration
HYPERTABLE_CONFIG = {
    'stock_aggs': {
        'chunk_interval': '1 day',
        'time_column': 'timestamp',
    },
    'crypto_aggs': {
        'chunk_interval': '1 day',
        'time_column': 'timestamp',
    },
    'quotes': {
        'chunk_interval': '1 hour',
        'time_column': 'timestamp',
    },
    'trades': {
        'chunk_interval': '1 hour',
        'time_column': 'timestamp',
    },
    'options_aggs': {
        'chunk_interval': '1 day',
        'time_column': 'timestamp',
    },
    'unusual_whales_options': {
        'chunk_interval': '1 day',
        'time_column': 'timestamp',
    },
    'options_flow': {
        'chunk_interval': '1 day',
        'time_column': 'timestamp',
    },
    'features': {
        'chunk_interval': '1 day',
        'time_column': 'timestamp',
    },
    'system_metrics': {
        'chunk_interval': '1 hour',
        'time_column': 'timestamp',
    },
    'trading_metrics': {
        'chunk_interval': '1 day',
        'time_column': 'timestamp',
    },
    'news_sentiment': {
        'chunk_interval': '1 day',
        'time_column': 'timestamp',
    },
}

# Compression Configuration
COMPRESSION_CONFIG = {
    'enabled': False,  # Temporarily disabled due to syntax issues
    'compress_after': '7 days',
    'compression_level': 9,  # 0-9, where 9 is highest compression
}

# Retention Policy Configuration
RETENTION_CONFIG = {
    'stock_aggs': {
        'enabled': True,
        'retention_period': '2 years',
    },
    'crypto_aggs': {
        'enabled': True,
        'retention_period': '2 years',
    },
    'quotes': {
        'enabled': True,
        'retention_period': '30 days',
    },
    'trades': {
        'enabled': True,
        'retention_period': '30 days',
    },
    'options_flow': {
        'enabled': True,
        'retention_period': '1 year',
    },
    'unusual_whales_options': {
        'enabled': True,
        'retention_period': '1 year',
    },
    'system_metrics': {
        'enabled': True,
        'retention_period': '90 days',
    },
    'news_sentiment': {
        'enabled': True,
        'retention_period': '1 year',
    },
}

# Database Migration Configuration
MIGRATION_CONFIG = {
    'script_location': 'src/utils/database/migrations',
    'version_table': 'alembic_version',
}

def get_connection_params() -> Dict[str, Any]:
    """
    Get database connection parameters for psycopg2.
    
    Returns:
        Dictionary of connection parameters
    """
    return TIMESCALEDB_CONFIG.copy()

def get_redis_connection_params() -> Dict[str, Any]:
    """
    Get Redis connection parameters.
    
    Returns:
        Dictionary of Redis connection parameters
    """
    return REDIS_CONFIG.copy()