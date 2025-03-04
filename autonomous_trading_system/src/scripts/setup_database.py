#!/usr/bin/env python3
"""
Database Setup Script

This script initializes the TimescaleDB database for the Autonomous Trading System.
It creates all necessary tables, indexes, and TimescaleDB hypertables.

Usage:
    python setup_database.py [--drop-existing] [--sample-data]

Options:
    --drop-existing    Drop existing tables before creating new ones
    --sample-data      Load sample data for testing
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import redis
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.config.database_config import (
    get_connection_params, 
    get_redis_connection_params,
    HYPERTABLE_CONFIG,
    COMPRESSION_CONFIG,
    RETENTION_CONFIG,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# SQL statements for creating tables
CREATE_TABLES_SQL = {
    # Market Data Tables
    'stock_aggs': """
    CREATE TABLE IF NOT EXISTS stock_aggs (
        timestamp TIMESTAMPTZ NOT NULL,
        symbol VARCHAR(16) NOT NULL,
        open NUMERIC(16,6) NOT NULL,
        high NUMERIC(16,6) NOT NULL,
        low NUMERIC(16,6) NOT NULL,
        close NUMERIC(16,6) NOT NULL,
        volume BIGINT NOT NULL,
        vwap NUMERIC(16,6),
        transactions INTEGER,
        timeframe VARCHAR(8) NOT NULL,
        source VARCHAR(32) DEFAULT 'polygon',
        multiplier INTEGER DEFAULT 1,
        timespan_unit VARCHAR(16) DEFAULT 'minute',
        adjusted BOOLEAN DEFAULT FALSE,
        otc BOOLEAN DEFAULT FALSE
    );
    """,
    
    'crypto_aggs': """
    CREATE TABLE IF NOT EXISTS crypto_aggs (
        timestamp TIMESTAMPTZ NOT NULL,
        symbol VARCHAR(16) NOT NULL,
        open NUMERIC(20,8) NOT NULL,
        high NUMERIC(20,8) NOT NULL,
        low NUMERIC(20,8) NOT NULL,
        close NUMERIC(20,8) NOT NULL,
        volume NUMERIC(24,8) NOT NULL,
        vwap NUMERIC(20,8),
        transactions INTEGER,
        timeframe VARCHAR(8) NOT NULL,
        source VARCHAR(32) DEFAULT 'polygon',
        multiplier INTEGER DEFAULT 1,
        timespan_unit VARCHAR(16) DEFAULT 'minute',
        exchange VARCHAR(16)
    );
    """,
    
    'quotes': """
    CREATE TABLE IF NOT EXISTS quotes (
        timestamp TIMESTAMPTZ NOT NULL,
        symbol VARCHAR(16) NOT NULL,
        bid_price NUMERIC(16,6) NOT NULL,
        ask_price NUMERIC(16,6) NOT NULL,
        bid_size INTEGER NOT NULL,
        ask_size INTEGER NOT NULL,
        exchange VARCHAR(8),
        conditions VARCHAR[],
        sequence_number BIGINT,
        tape CHAR(1),
        source VARCHAR(32) DEFAULT 'polygon'
    );
    """,
    
    'trades': """
    CREATE TABLE IF NOT EXISTS trades (
        timestamp TIMESTAMPTZ NOT NULL,
        symbol VARCHAR(16) NOT NULL,
        price NUMERIC(16,6) NOT NULL,
        size INTEGER NOT NULL,
        exchange VARCHAR(8) NOT NULL,
        conditions VARCHAR[],
        tape CHAR(1),
        sequence_number BIGINT,
        trade_id VARCHAR(32),
        source VARCHAR(32) DEFAULT 'polygon'
    );
    """,
    
    'options_aggs': """
    CREATE TABLE IF NOT EXISTS options_aggs (
        timestamp TIMESTAMPTZ NOT NULL,
        symbol VARCHAR(32) NOT NULL,
        underlying VARCHAR(16) NOT NULL,
        expiration DATE NOT NULL,
        strike NUMERIC(16,6) NOT NULL,
        option_type CHAR(1) NOT NULL,
        open NUMERIC(16,6),
        high NUMERIC(16,6),
        low NUMERIC(16,6),
        close NUMERIC(16,6),
        volume INTEGER,
        open_interest INTEGER,
        timeframe VARCHAR(8) NOT NULL,
        multiplier INTEGER DEFAULT 1,
        timespan_unit VARCHAR(16) DEFAULT 'minute',
        source VARCHAR(32) DEFAULT 'polygon'
    );
    """,
    
    'unusual_whales_options': """
    CREATE TABLE IF NOT EXISTS unusual_whales_options (
        timestamp TIMESTAMPTZ NOT NULL,
        symbol VARCHAR(32) NOT NULL,
        underlying VARCHAR(16) NOT NULL,
        underlying_price NUMERIC(16,6),
        expiration DATE NOT NULL,
        strike NUMERIC(16,6) NOT NULL,
        option_type CHAR(1) NOT NULL,
        premium NUMERIC(16,6) NOT NULL,
        premium_type VARCHAR(8) NOT NULL,
        sentiment VARCHAR(8) NOT NULL,
        unusual_score INTEGER,
        volume INTEGER NOT NULL,
        open_interest INTEGER,
        volume_oi_ratio NUMERIC(10,6),
        implied_volatility NUMERIC(10,6),
        days_to_expiration INTEGER NOT NULL,
        trade_type VARCHAR(16),
        size_notation VARCHAR(16),
        alert_id VARCHAR(64),
        sector VARCHAR(32),
        source VARCHAR(32) DEFAULT 'unusual_whales'
    );
    """,
    
    'options_flow': """
    DROP TABLE IF EXISTS options_flow CASCADE;
    CREATE TABLE IF NOT EXISTS options_flow (
        id TEXT NOT NULL,
        timestamp TIMESTAMPTZ NOT NULL,
        symbol TEXT NOT NULL,
        contract_type TEXT,
        strike NUMERIC,
        expiration_date TIMESTAMPTZ,
        premium NUMERIC,
        size NUMERIC,
        open_interest NUMERIC,
        implied_volatility NUMERIC,
        delta NUMERIC,
        gamma NUMERIC,
        theta NUMERIC,
        vega NUMERIC,
        sentiment TEXT,
        trade_type TEXT,
        source TEXT DEFAULT 'unusual_whales',
        PRIMARY KEY (id, timestamp)
    );
    """,
    
    # Reference Data Tables
    'ticker_details': """
    CREATE TABLE IF NOT EXISTS ticker_details (
        ticker VARCHAR(16) PRIMARY KEY,
        name VARCHAR(128) NOT NULL,
        market VARCHAR(32) NOT NULL,
        locale VARCHAR(16) NOT NULL,
        type VARCHAR(16) NOT NULL,
        currency VARCHAR(8) NOT NULL,
        active BOOLEAN DEFAULT TRUE,
        primary_exchange VARCHAR(32),
        last_updated TIMESTAMPTZ NOT NULL,
        description TEXT,
        sic_code VARCHAR(8),
        sic_description VARCHAR(256),
        ticker_root VARCHAR(16),
        homepage_url VARCHAR(256),
        total_employees INTEGER,
        list_date DATE,
        share_class_shares_outstanding BIGINT,
        weighted_shares_outstanding BIGINT,
        market_cap BIGINT,
        phone_number VARCHAR(32),
        address JSONB,
        metadata JSONB
    );
    """,
    
    'market_holidays': """
    CREATE TABLE IF NOT EXISTS market_holidays (
        date DATE NOT NULL,
        name VARCHAR(64) NOT NULL,
        market VARCHAR(32) NOT NULL,
        status VARCHAR(16) NOT NULL,
        open_time TIME,
        close_time TIME,
        year INTEGER NOT NULL,
        PRIMARY KEY (date, market)
    );
    """,
    
    'market_status': """
    CREATE TABLE IF NOT EXISTS market_status (
        timestamp TIMESTAMPTZ NOT NULL,
        market VARCHAR(32) NOT NULL,
        status VARCHAR(16) NOT NULL,
        next_open TIMESTAMPTZ,
        next_close TIMESTAMPTZ,
        early_close BOOLEAN DEFAULT FALSE,
        late_open BOOLEAN DEFAULT FALSE,
        PRIMARY KEY (timestamp, market)
    );
    """,
    
    'news_articles': """
    DROP TABLE IF EXISTS news_articles CASCADE;
    CREATE TABLE IF NOT EXISTS news_articles (
        article_id VARCHAR(64) PRIMARY KEY,
        published_utc TIMESTAMPTZ NOT NULL,
        title VARCHAR(512) NOT NULL,
        author VARCHAR(128),
        article_url VARCHAR(512) NOT NULL,
        tickers VARCHAR[],
        image_url VARCHAR(512),
        description TEXT,
        keywords VARCHAR[],
        source VARCHAR(64) NOT NULL
    );
    """,
    
    # Feature Engineering Tables
    'news_sentiment': """
    CREATE TABLE IF NOT EXISTS news_sentiment (
        article_id VARCHAR(64) REFERENCES news_articles(article_id),
        timestamp TIMESTAMPTZ NOT NULL,
        symbol VARCHAR(16) NOT NULL,
        sentiment_score NUMERIC(5,4) NOT NULL,
        sentiment_label VARCHAR(16) NOT NULL,
        confidence NUMERIC(5,4) NOT NULL,
        entity_mentions JSONB,
        keywords JSONB,
        model_version VARCHAR(32) NOT NULL,
        PRIMARY KEY (article_id, symbol, timestamp)
    );
    """,
    
    'features': """
    CREATE TABLE IF NOT EXISTS features (
        timestamp TIMESTAMPTZ NOT NULL,
        symbol VARCHAR(16) NOT NULL,
        feature_name VARCHAR(64) NOT NULL,
        feature_value NUMERIC(20,8) NOT NULL,
        timeframe VARCHAR(8) NOT NULL,
        feature_group VARCHAR(32) NOT NULL
    );
    """,
    
    'feature_metadata': """
    CREATE TABLE IF NOT EXISTS feature_metadata (
        feature_name VARCHAR(64) PRIMARY KEY,
        description TEXT NOT NULL,
        formula TEXT,
        parameters JSONB,
        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL,
        version VARCHAR(16) NOT NULL,
        is_active BOOLEAN DEFAULT TRUE
    );
    """,
    
    # Model Training Tables
    'models': """
    CREATE TABLE IF NOT EXISTS models (
        model_id UUID PRIMARY KEY,
        model_name VARCHAR(64) NOT NULL,
        model_type VARCHAR(32) NOT NULL,
        target VARCHAR(64) NOT NULL,
        features VARCHAR[] NOT NULL,
        parameters JSONB NOT NULL,
        metrics JSONB NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        trained_at TIMESTAMPTZ NOT NULL,
        version VARCHAR(16) NOT NULL,
        status VARCHAR(16) NOT NULL,
        file_path VARCHAR(256) NOT NULL
    );
    """,
    
    'model_training_runs': """
    CREATE TABLE IF NOT EXISTS model_training_runs (
        run_id UUID PRIMARY KEY,
        model_id UUID REFERENCES models(model_id),
        start_time TIMESTAMPTZ NOT NULL,
        end_time TIMESTAMPTZ,
        status VARCHAR(16) NOT NULL,
        parameters JSONB NOT NULL,
        metrics JSONB,
        logs TEXT
    );
    """,
    
    # Trading Strategy Tables
    'trading_signals': """
    CREATE TABLE IF NOT EXISTS trading_signals (
        signal_id UUID PRIMARY KEY,
        timestamp TIMESTAMPTZ NOT NULL,
        symbol VARCHAR(16) NOT NULL,
        signal_type VARCHAR(16) NOT NULL,
        confidence NUMERIC(5,4) NOT NULL,
        model_id UUID REFERENCES models(model_id),
        timeframe VARCHAR(8) NOT NULL,
        parameters JSONB,
        features_snapshot JSONB
    );
    """,
    
    'orders': """
    CREATE TABLE IF NOT EXISTS orders (
        order_id UUID PRIMARY KEY,
        external_order_id VARCHAR(64) UNIQUE,
        timestamp TIMESTAMPTZ NOT NULL,
        symbol VARCHAR(16) NOT NULL,
        order_type VARCHAR(16) NOT NULL,
        side VARCHAR(8) NOT NULL,
        quantity NUMERIC(16,6) NOT NULL,
        price NUMERIC(16,6),
        status VARCHAR(16) NOT NULL,
        signal_id UUID REFERENCES trading_signals(signal_id),
        strategy_id UUID,
        filled_quantity NUMERIC(16,6) DEFAULT 0,
        filled_price NUMERIC(16,6),
        commission NUMERIC(10,6) DEFAULT 0,
        updated_at TIMESTAMPTZ NOT NULL
    );
    """,
    
    'positions': """
    CREATE TABLE IF NOT EXISTS positions (
        position_id UUID PRIMARY KEY,
        symbol VARCHAR(16) NOT NULL,
        quantity NUMERIC(16,6) NOT NULL,
        entry_price NUMERIC(16,6) NOT NULL,
        current_price NUMERIC(16,6) NOT NULL,
        entry_time TIMESTAMPTZ NOT NULL,
        last_update TIMESTAMPTZ NOT NULL,
        strategy_id UUID,
        status VARCHAR(16) NOT NULL,
        pnl NUMERIC(16,6) NOT NULL,
        pnl_percentage NUMERIC(10,6) NOT NULL,
        metadata JSONB
    );
    """,
    
    # Monitoring Tables
    'system_metrics': """
    CREATE TABLE IF NOT EXISTS system_metrics (
        timestamp TIMESTAMPTZ NOT NULL,
        metric_name VARCHAR(64) NOT NULL,
        metric_value NUMERIC(20,8) NOT NULL,
        component VARCHAR(32) NOT NULL,
        host VARCHAR(64) NOT NULL,
        tags JSONB
    );
    """,
    
    'trading_metrics': """
    CREATE TABLE IF NOT EXISTS trading_metrics (
        timestamp TIMESTAMPTZ NOT NULL,
        metric_name VARCHAR(64) NOT NULL,
        metric_value NUMERIC(20,8) NOT NULL,
        symbol VARCHAR(16),
        strategy_id UUID,
        timeframe VARCHAR(8),
        tags JSONB
    );
    """,
    
    'alerts': """
    CREATE TABLE IF NOT EXISTS alerts (
        alert_id UUID PRIMARY KEY,
        timestamp TIMESTAMPTZ NOT NULL,
        alert_type VARCHAR(32) NOT NULL,
        severity VARCHAR(16) NOT NULL,
        message TEXT NOT NULL,
        component VARCHAR(32) NOT NULL,
        status VARCHAR(16) NOT NULL,
        resolved_at TIMESTAMPTZ,
        metadata JSONB
    );
    """,
    
    # Emergency Stop Tables
    'emergency_events': """
    CREATE TABLE IF NOT EXISTS emergency_events (
        event_id UUID PRIMARY KEY,
        timestamp TIMESTAMPTZ NOT NULL,
        trigger_type VARCHAR(32) NOT NULL,
        severity VARCHAR(16) NOT NULL,
        description TEXT NOT NULL,
        action_taken VARCHAR(32) NOT NULL,
        positions_closed INTEGER,
        total_value NUMERIC(20,8),
        resolution_time TIMESTAMPTZ,
        triggered_by VARCHAR(64) NOT NULL,
        metadata JSONB
    );
    """
}

# SQL statements for creating indexes
CREATE_INDEXES_SQL = {
    'stock_aggs_idx1': """
    CREATE INDEX IF NOT EXISTS idx_stock_aggs_symbol_timestamp_timeframe 
    ON stock_aggs (symbol, timestamp, timeframe);
    """,
    
    'stock_aggs_idx2': """
    CREATE INDEX IF NOT EXISTS idx_stock_aggs_adjusted_symbol_timestamp 
    ON stock_aggs (adjusted, symbol, timestamp);
    """,
    
    'crypto_aggs_idx1': """
    CREATE INDEX IF NOT EXISTS idx_crypto_aggs_symbol_timestamp_timeframe 
    ON crypto_aggs (symbol, timestamp, timeframe);
    """,
    
    'crypto_aggs_idx2': """
    CREATE INDEX IF NOT EXISTS idx_crypto_aggs_exchange_symbol_timestamp 
    ON crypto_aggs (exchange, symbol, timestamp);
    """,
    
    'quotes_idx1': """
    CREATE INDEX IF NOT EXISTS idx_quotes_symbol_timestamp 
    ON quotes (symbol, timestamp);
    """,
    
    'trades_idx1': """
    CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp 
    ON trades (symbol, timestamp);
    """,
    
    'trades_idx2': """
    CREATE INDEX IF NOT EXISTS idx_trades_exchange_timestamp 
    ON trades (exchange, timestamp);
    """,
    
    'options_aggs_idx1': """
    CREATE INDEX IF NOT EXISTS idx_options_aggs_underlying_expiration_strike_type 
    ON options_aggs (underlying, expiration, strike, option_type);
    """,
    
    'options_aggs_idx2': """
    CREATE INDEX IF NOT EXISTS idx_options_aggs_symbol_timestamp 
    ON options_aggs (symbol, timestamp);
    """,
    
    'unusual_whales_options_idx1': """
    CREATE INDEX IF NOT EXISTS idx_unusual_whales_underlying_expiration_strike_type 
    ON unusual_whales_options (underlying, expiration, strike, option_type);
    """,
    
    'unusual_whales_options_idx2': """
    CREATE INDEX IF NOT EXISTS idx_unusual_whales_symbol_timestamp 
    ON unusual_whales_options (symbol, timestamp);
    """,
    
    'unusual_whales_options_idx3': """
    CREATE INDEX IF NOT EXISTS idx_unusual_whales_unusual_score 
    ON unusual_whales_options (unusual_score) 
    WHERE unusual_score IS NOT NULL;
    """,
    
    'options_flow_idx1': """
    CREATE INDEX IF NOT EXISTS idx_options_flow_symbol_timestamp 
    ON options_flow (symbol, timestamp);
    """,
    
    'options_flow_idx2': """
    CREATE INDEX IF NOT EXISTS idx_options_flow_expiration_strike_type 
    ON options_flow (expiration_date, strike, contract_type);
    """,
    
    'options_flow_idx3': """
    CREATE INDEX IF NOT EXISTS idx_options_flow_sentiment 
    ON options_flow (sentiment);
    """,
    
    'ticker_details_idx1': """
    CREATE INDEX IF NOT EXISTS idx_ticker_details_market_active 
    ON ticker_details (market, active);
    """,
    
    'ticker_details_idx2': """
    CREATE INDEX IF NOT EXISTS idx_ticker_details_type_active 
    ON ticker_details (type, active);
    """,
    
    'market_holidays_idx1': """
    CREATE INDEX IF NOT EXISTS idx_market_holidays_year_market 
    ON market_holidays (year, market);
    """,
    
    'market_status_idx1': """
    CREATE INDEX IF NOT EXISTS idx_market_status_market_status 
    ON market_status (market, status);
    """,
    
    'news_articles_idx1': """
    CREATE INDEX IF NOT EXISTS idx_news_articles_published_utc 
    ON news_articles (published_utc);
    """,
    
    'news_articles_idx2': """
    CREATE INDEX IF NOT EXISTS idx_news_articles_tickers 
    ON news_articles USING GIN (tickers);
    """,
    
    'news_sentiment_idx1': """
    CREATE INDEX IF NOT EXISTS idx_news_sentiment_symbol_timestamp 
    ON news_sentiment (symbol, timestamp);
    """,
    
    'news_sentiment_idx2': """
    CREATE INDEX IF NOT EXISTS idx_news_sentiment_label_confidence 
    ON news_sentiment (sentiment_label, confidence);
    """,
    
    'news_sentiment_idx3': """
    CREATE INDEX IF NOT EXISTS idx_news_sentiment_score 
    ON news_sentiment (sentiment_score);
    """,
    
    'features_idx1': """
    CREATE INDEX IF NOT EXISTS idx_features_symbol_name_timestamp_timeframe 
    ON features (symbol, feature_name, timestamp, timeframe);
    """,
    
    'features_idx2': """
    CREATE INDEX IF NOT EXISTS idx_features_group_symbol_timestamp 
    ON features (feature_group, symbol, timestamp);
    """,
    
    'models_idx1': """
    CREATE INDEX IF NOT EXISTS idx_models_name_version 
    ON models (model_name, version);
    """,
    
    'models_idx2': """
    CREATE INDEX IF NOT EXISTS idx_models_status_type 
    ON models (status, model_type);
    """,
    
    'model_training_runs_idx1': """
    CREATE INDEX IF NOT EXISTS idx_model_training_runs_model_start 
    ON model_training_runs (model_id, start_time);
    """,
    
    'model_training_runs_idx2': """
    CREATE INDEX IF NOT EXISTS idx_model_training_runs_status 
    ON model_training_runs (status);
    """,
    
    'trading_signals_idx1': """
    CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol_timestamp_type 
    ON trading_signals (symbol, timestamp, signal_type);
    """,
    
    'trading_signals_idx2': """
    CREATE INDEX IF NOT EXISTS idx_trading_signals_model_timestamp 
    ON trading_signals (model_id, timestamp);
    """,
    
    'orders_idx1': """
    CREATE INDEX IF NOT EXISTS idx_orders_symbol_timestamp 
    ON orders (symbol, timestamp);
    """,
    
    'orders_idx2': """
    CREATE INDEX IF NOT EXISTS idx_orders_status_timestamp 
    ON orders (status, timestamp);
    """,
    
    'orders_idx3': """
    CREATE INDEX IF NOT EXISTS idx_orders_signal_id 
    ON orders (signal_id);
    """,
    
    'orders_idx4': """
    CREATE INDEX IF NOT EXISTS idx_orders_strategy_timestamp 
    ON orders (strategy_id, timestamp);
    """,
    
    'positions_idx1': """
    CREATE INDEX IF NOT EXISTS idx_positions_symbol_status 
    ON positions (symbol, status);
    """,
    
    'positions_idx2': """
    CREATE INDEX IF NOT EXISTS idx_positions_strategy_status 
    ON positions (strategy_id, status);
    """,
    
    'system_metrics_idx1': """
    CREATE INDEX IF NOT EXISTS idx_system_metrics_name_component_timestamp 
    ON system_metrics (metric_name, component, timestamp);
    """,
    
    'system_metrics_idx2': """
    CREATE INDEX IF NOT EXISTS idx_system_metrics_host_timestamp 
    ON system_metrics (host, timestamp);
    """,
    
    'trading_metrics_idx1': """
    CREATE INDEX IF NOT EXISTS idx_trading_metrics_name_timestamp 
    ON trading_metrics (metric_name, timestamp);
    """,
    
    'trading_metrics_idx2': """
    CREATE INDEX IF NOT EXISTS idx_trading_metrics_symbol_name_timestamp 
    ON trading_metrics (symbol, metric_name, timestamp) 
    WHERE symbol IS NOT NULL;
    """,
    
    'trading_metrics_idx3': """
    CREATE INDEX IF NOT EXISTS idx_trading_metrics_strategy_name_timestamp 
    ON trading_metrics (strategy_id, metric_name, timestamp) 
    WHERE strategy_id IS NOT NULL;
    """,
    
    'alerts_idx1': """
    CREATE INDEX IF NOT EXISTS idx_alerts_timestamp_severity 
    ON alerts (timestamp, severity);
    """,
    
    'alerts_idx2': """
    CREATE INDEX IF NOT EXISTS idx_alerts_status_timestamp 
    ON alerts (status, timestamp);
    """,
    
    'alerts_idx3': """
    CREATE INDEX IF NOT EXISTS idx_alerts_component_timestamp 
    ON alerts (component, timestamp);
    """,
    
    'emergency_events_idx1': """
    CREATE INDEX IF NOT EXISTS idx_emergency_events_timestamp_severity 
    ON emergency_events (timestamp, severity);
    """,
    
    'emergency_events_idx2': """
    CREATE INDEX IF NOT EXISTS idx_emergency_events_trigger_timestamp 
    ON emergency_events (trigger_type, timestamp);
    """
}

def create_hypertable(conn, table_name: str, config: Dict[str, Any]) -> None:
    """
    Convert a regular table to a TimescaleDB hypertable.
    
    Args:
        conn: Database connection
        table_name: Name of the table to convert
        config: Hypertable configuration
    """
    cursor = conn.cursor()
    
    # Check if table is already a hypertable
    cursor.execute(
        "SELECT * FROM timescaledb_information.hypertables WHERE hypertable_name = %s",
        (table_name,)
    )
    if cursor.fetchone() is not None:
        logger.info(f"Table {table_name} is already a hypertable")
        return
    
    # Convert to hypertable
    time_column = config.get('time_column', 'timestamp')
    chunk_interval = config.get('chunk_interval', '1 day')
    
    cursor.execute(
        f"SELECT create_hypertable('{table_name}', '{time_column}', chunk_time_interval => interval '{chunk_interval}');"
    )
    logger.info(f"Converted {table_name} to a hypertable with chunk interval {chunk_interval}")
    
    # Set up compression if enabled
    if COMPRESSION_CONFIG.get('enabled', False):
        compress_after = COMPRESSION_CONFIG.get('compress_after', '7 days')
        compression_level = COMPRESSION_CONFIG.get('compression_level', 9)
        
        cursor.execute(
            f"""
            ALTER TABLE {table_name} SET (
                timescaledb.compress,
                timescaledb.compress_segmentby = 'symbol',
                timescaledb.compress_orderby = '{time_column}',
                timescaledb.compress_chunk_time_interval = interval '{compress_after}'
            );
            """
        )
        logger.info(f"Enabled compression for {table_name}")
        
        # Add compression policy
        cursor.execute(
            f"""
            SELECT add_compression_policy('{table_name}', interval '{compress_after}');
            """
        )
        logger.info(f"Added compression policy for {table_name} after {compress_after}")
    
    # Set up retention policy if enabled for this table
    retention_config = RETENTION_CONFIG.get(table_name, {})
    if retention_config.get('enabled', False):
        retention_period = retention_config.get('retention_period', '1 year')
        
        cursor.execute(
            f"""
            SELECT add_retention_policy('{table_name}', interval '{retention_period}');
            """
        )
        logger.info(f"Added retention policy for {table_name} with period {retention_period}")
    
    cursor.close()

def setup_redis(config: Dict[str, Any]) -> None:
    """
    Set up Redis for caching and real-time operations.
    
    Args:
        config: Redis connection parameters
    """
    try:
        r = redis.Redis(**config)
        r.ping()  # Test connection
        
        # Clear existing keys (only in development/testing)
        if os.getenv('ENVIRONMENT', 'development') != 'production':
            r.flushall()
            logger.info("Cleared Redis database")
        
        # Set up key expiration policies
        # This is just an example, adjust based on your needs
        logger.info("Redis connection successful")
        
    except redis.ConnectionError as e:
        logger.error(f"Failed to connect to Redis: {e}")
        logger.warning("Continuing without Redis setup")

def main() -> None:
    """Main function to set up the database."""
    parser = argparse.ArgumentParser(description='Set up the TimescaleDB database for the Autonomous Trading System')
    parser.add_argument('--drop-existing', action='store_true', help='Drop existing tables before creating new ones')
    parser.add_argument('--sample-data', action='store_true', help='Load sample data for testing')
    args = parser.parse_args()
    
    # Load environment variables
    load_dotenv()
    
    # Get database connection parameters
    db_params = get_connection_params()
    
    try:
        # Connect to PostgreSQL server
        logger.info(f"Connecting to PostgreSQL server at {db_params['host']}:{db_params['port']}")
        conn = psycopg2.connect(
            host=db_params['host'],
            port=db_params['port'],
            database='postgres',
            user=db_params['user'],
            password=db_params['password']
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if database exists, create if it doesn't
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_params['database']}'")
        if not cursor.fetchone():
            logger.info(f"Creating database {db_params['database']}")
            cursor.execute(f"CREATE DATABASE {db_params['database']}")
        else:
            logger.info(f"Database {db_params['database']} already exists")
        
        # Close connection to server
        cursor.close()
        conn.close()
        
        # Connect to the specific database
        logger.info(f"Connecting to database {db_params['database']}")
        conn = psycopg2.connect(**db_params)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if TimescaleDB extension is installed
        cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'timescaledb'")
        if not cursor.fetchone():
            logger.info("Creating TimescaleDB extension")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
        else:
            logger.info("TimescaleDB extension already installed")
        
        # Drop existing tables if requested
        if args.drop_existing:
            logger.warning("Dropping existing tables")
            for table_name in reversed(list(CREATE_TABLES_SQL.keys())):
                cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
        
        # Create tables
        for table_name, sql in CREATE_TABLES_SQL.items():
            logger.info(f"Creating table {table_name}")
            cursor.execute(sql)
        
        # Create indexes
        for index_name, sql in CREATE_INDEXES_SQL.items():
            logger.info(f"Creating index {index_name}")
            cursor.execute(sql)
        
        # Convert tables to hypertables
        for table_name, config in HYPERTABLE_CONFIG.items():
            logger.info(f"Setting up hypertable for {table_name}")
            create_hypertable(conn, table_name, config)
        
        # Load sample data if requested
        if args.sample_data:
            logger.info("Loading sample data")
            # This would be implemented separately
            # load_sample_data(conn)
        
        cursor.close()
        conn.close()
        
        # Set up Redis
        logger.info("Setting up Redis")
        redis_params = get_redis_connection_params()
        setup_redis(redis_params)
        
        logger.info("Database setup completed successfully")
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()