#!/usr/bin/env python3
"""
Database Setup Script

This script initializes the TimescaleDB database for the Autonomous Trading System.
It creates all necessary tables, indexes, and TimescaleDB hypertables.

Usage:
    python setup_database.py [--drop-existing] [--sample-data] [--config-only]

Options:
    --drop-existing    Drop existing tables before creating new ones
    --sample-data      Load sample data for testing
    --config-only      Only update configuration, do not create tables
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, Any, List, Tuple
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
    get_redis_persistence_config,
    get_backup_config,
    COMPRESSION_CONFIG,
    RETENTION_CONFIG,
    REDIS_PERSISTENCE_CONFIG
)

# Set up logging
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
    CREATE INDEX IF NOT EXISTS idx_stock_aggs_symbol_timestamp_timeframe 
    ON stock_aggs (symbol, timestamp, timeframe);
    CREATE INDEX IF NOT EXISTS idx_stock_aggs_adjusted_symbol_timestamp 
    ON stock_aggs (adjusted, symbol, timestamp);
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
    CREATE INDEX IF NOT EXISTS idx_crypto_aggs_symbol_timestamp_timeframe 
    ON crypto_aggs (symbol, timestamp, timeframe);
    CREATE INDEX IF NOT EXISTS idx_crypto_aggs_exchange_symbol_timestamp 
    ON crypto_aggs (exchange, symbol, timestamp);
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
    CREATE INDEX IF NOT EXISTS idx_quotes_symbol_timestamp 
    ON quotes (symbol, timestamp);
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
    CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp 
    ON trades (symbol, timestamp);
    CREATE INDEX IF NOT EXISTS idx_trades_exchange_timestamp 
    ON trades (exchange, timestamp);
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
    CREATE INDEX IF NOT EXISTS idx_options_aggs_underlying_expiration_strike_type 
    ON options_aggs (underlying, expiration, strike, option_type);
    CREATE INDEX IF NOT EXISTS idx_options_aggs_symbol_timestamp 
    ON options_aggs (symbol, timestamp);
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
    CREATE INDEX IF NOT EXISTS idx_unusual_whales_underlying_expiration_strike_type 
    ON unusual_whales_options (underlying, expiration, strike, option_type);
    CREATE INDEX IF NOT EXISTS idx_unusual_whales_symbol_timestamp 
    ON unusual_whales_options (symbol, timestamp);
    CREATE INDEX IF NOT EXISTS idx_unusual_whales_unusual_score 
    ON unusual_whales_options (unusual_score) 
    WHERE unusual_score IS NOT NULL;
    """,
    
    'options_flow': """
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
    CREATE INDEX IF NOT EXISTS idx_options_flow_symbol_timestamp 
    ON options_flow (symbol, timestamp);
    CREATE INDEX IF NOT EXISTS idx_options_flow_expiration_strike_type 
    ON options_flow (expiration_date, strike, contract_type);
    CREATE INDEX IF NOT EXISTS idx_options_flow_sentiment 
    ON options_flow (sentiment);
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
    CREATE INDEX IF NOT EXISTS idx_ticker_details_market_active 
    ON ticker_details (market, active);
    CREATE INDEX IF NOT EXISTS idx_ticker_details_type_active 
    ON ticker_details (type, active);
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
    CREATE INDEX IF NOT EXISTS idx_market_holidays_year_market 
    ON market_holidays (year, market);
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
    CREATE INDEX IF NOT EXISTS idx_market_status_market_status 
    ON market_status (market, status);
    """,
    
    'news_articles': """
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
    CREATE INDEX IF NOT EXISTS idx_news_articles_published_utc 
    ON news_articles (published_utc);
    CREATE INDEX IF NOT EXISTS idx_news_articles_tickers 
    ON news_articles USING GIN (tickers);
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
    CREATE INDEX IF NOT EXISTS idx_news_sentiment_symbol_timestamp 
    ON news_sentiment (symbol, timestamp);
    CREATE INDEX IF NOT EXISTS idx_news_sentiment_label_confidence 
    ON news_sentiment (sentiment_label, confidence);
    CREATE INDEX IF NOT EXISTS idx_news_sentiment_score 
    ON news_sentiment (sentiment_score);
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
    CREATE INDEX IF NOT EXISTS idx_features_symbol_name_timestamp_timeframe 
    ON features (symbol, feature_name, timestamp, timeframe);
    CREATE INDEX IF NOT EXISTS idx_features_group_symbol_timestamp 
    ON features (feature_group, symbol, timestamp);
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
    CREATE INDEX IF NOT EXISTS idx_models_name_version 
    ON models (model_name, version);
    CREATE INDEX IF NOT EXISTS idx_models_status_type 
    ON models (status, model_type);
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
    CREATE INDEX IF NOT EXISTS idx_model_training_runs_model_start 
    ON model_training_runs (model_id, start_time);
    CREATE INDEX IF NOT EXISTS idx_model_training_runs_status 
    ON model_training_runs (status);
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
    CREATE INDEX IF NOT EXISTS idx_trading_signals_symbol_timestamp_type 
    ON trading_signals (symbol, timestamp, signal_type);
    CREATE INDEX IF NOT EXISTS idx_trading_signals_model_timestamp 
    ON trading_signals (model_id, timestamp);
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
    CREATE INDEX IF NOT EXISTS idx_orders_symbol_timestamp 
    ON orders (symbol, timestamp);
    CREATE INDEX IF NOT EXISTS idx_orders_status_timestamp 
    ON orders (status, timestamp);
    CREATE INDEX IF NOT EXISTS idx_orders_signal_id 
    ON orders (signal_id);
    CREATE INDEX IF NOT EXISTS idx_orders_strategy_timestamp 
    ON orders (strategy_id, timestamp);
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
    CREATE INDEX IF NOT EXISTS idx_positions_symbol_status 
    ON positions (symbol, status);
    CREATE INDEX IF NOT EXISTS idx_positions_strategy_status 
    ON positions (strategy_id, status);
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
    CREATE INDEX IF NOT EXISTS idx_system_metrics_name_component_timestamp 
    ON system_metrics (metric_name, component, timestamp);
    CREATE INDEX IF NOT EXISTS idx_system_metrics_host_timestamp 
    ON system_metrics (host, timestamp);
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
    CREATE INDEX IF NOT EXISTS idx_trading_metrics_name_timestamp 
    ON trading_metrics (metric_name, timestamp);
    CREATE INDEX IF NOT EXISTS idx_trading_metrics_symbol_name_timestamp 
    ON trading_metrics (symbol, metric_name, timestamp) 
    WHERE symbol IS NOT NULL;
    CREATE INDEX IF NOT EXISTS idx_trading_metrics_strategy_name_timestamp 
    ON trading_metrics (strategy_id, metric_name, timestamp) 
    WHERE strategy_id IS NOT NULL;
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
    CREATE INDEX IF NOT EXISTS idx_alerts_timestamp_severity 
    ON alerts (timestamp, severity);
    CREATE INDEX IF NOT EXISTS idx_alerts_status_timestamp 
    ON alerts (status, timestamp);
    CREATE INDEX IF NOT EXISTS idx_alerts_component_timestamp 
    ON alerts (component, timestamp);
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
    CREATE INDEX IF NOT EXISTS idx_emergency_events_timestamp_severity 
    ON emergency_events (timestamp, severity);
    CREATE INDEX IF NOT EXISTS idx_emergency_events_trigger_timestamp 
    ON emergency_events (trigger_type, timestamp);
    """,
    
    # Validation Tables
    'validation_rules': """
    CREATE TABLE IF NOT EXISTS validation_rules (
        rule_id UUID PRIMARY KEY,
        rule_name VARCHAR(64) NOT NULL,
        data_type VARCHAR(32) NOT NULL,
        description TEXT NOT NULL,
        parameters JSONB NOT NULL,
        is_active BOOLEAN DEFAULT TRUE,
        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_validation_rules_data_type 
    ON validation_rules (data_type);
    CREATE INDEX IF NOT EXISTS idx_validation_rules_is_active 
    ON validation_rules (is_active);
    """,
    
    'data_quality_issues': """
    CREATE TABLE IF NOT EXISTS data_quality_issues (
        issue_id UUID PRIMARY KEY,
        timestamp TIMESTAMPTZ NOT NULL,
        data_type VARCHAR(32) NOT NULL,
        symbol VARCHAR(16) NOT NULL,
        issue_type VARCHAR(32) NOT NULL,
        description TEXT NOT NULL,
        severity VARCHAR(16) NOT NULL,
        is_resolved BOOLEAN DEFAULT FALSE,
        resolved_at TIMESTAMPTZ,
        resolution_notes TEXT,
        metadata JSONB
    );
    CREATE INDEX IF NOT EXISTS idx_data_quality_issues_timestamp 
    ON data_quality_issues (timestamp);
    CREATE INDEX IF NOT EXISTS idx_data_quality_issues_symbol_type 
    ON data_quality_issues (symbol, issue_type);
    CREATE INDEX IF NOT EXISTS idx_data_quality_issues_is_resolved 
    ON data_quality_issues (is_resolved);
    """,
    
    # Batch Processing Tables
    'data_batches': """
    CREATE TABLE IF NOT EXISTS data_batches (
        batch_id UUID PRIMARY KEY,
        data_type VARCHAR(32) NOT NULL,
        start_time TIMESTAMPTZ NOT NULL,
        end_time TIMESTAMPTZ NOT NULL,
        status VARCHAR(16) NOT NULL,
        record_count INTEGER NOT NULL,
        processed_count INTEGER NOT NULL,
        error_count INTEGER NOT NULL,
        created_at TIMESTAMPTZ NOT NULL,
        updated_at TIMESTAMPTZ NOT NULL,
        metadata JSONB
    );
    CREATE INDEX IF NOT EXISTS idx_data_batches_data_type_status 
    ON data_batches (data_type, status);
    CREATE INDEX IF NOT EXISTS idx_data_batches_time_range 
    ON data_batches (start_time, end_time);
    """,
    
    # Performance Metrics Tables
    'performance_metrics': """
    CREATE TABLE IF NOT EXISTS performance_metrics (
        timestamp TIMESTAMPTZ NOT NULL,
        component VARCHAR(32) NOT NULL,
        operation VARCHAR(64) NOT NULL,
        execution_time INTEGER NOT NULL,
        cpu_usage NUMERIC(5,2),
        memory_usage NUMERIC(5,2),
        record_count INTEGER,
        host VARCHAR(64) NOT NULL,
        metadata JSONB
    );
    CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp 
    ON performance_metrics (timestamp);
    CREATE INDEX IF NOT EXISTS idx_performance_metrics_component_operation 
    ON performance_metrics (component, operation);
    """
}

# Hypertable configuration
HYPERTABLE_CONFIG = {
    'stock_aggs': {'time_column': 'timestamp', 'chunk_interval': '1 day'},
    'crypto_aggs': {'time_column': 'timestamp', 'chunk_interval': '1 day'},
    'quotes': {'time_column': 'timestamp', 'chunk_interval': '1 hour'},
    'trades': {'time_column': 'timestamp', 'chunk_interval': '1 hour'},
    'options_aggs': {'time_column': 'timestamp', 'chunk_interval': '1 day'},
    'unusual_whales_options': {'time_column': 'timestamp', 'chunk_interval': '1 day'},
    'options_flow': {'time_column': 'timestamp', 'chunk_interval': '1 day'},
    'news_sentiment': {'time_column': 'timestamp', 'chunk_interval': '1 week'},
    'features': {'time_column': 'timestamp', 'chunk_interval': '1 day'},
    'system_metrics': {'time_column': 'timestamp', 'chunk_interval': '1 hour'},
    'trading_metrics': {'time_column': 'timestamp', 'chunk_interval': '1 day'},
    'performance_metrics': {'time_column': 'timestamp', 'chunk_interval': '1 day'}
}

# Sample data for testing
def load_sample_data(conn) -> None:
    """
    Load sample data for testing.
    
    Args:
        conn: Database connection
    """
    cursor = conn.cursor()
    
    # Sample ticker details
    cursor.execute("""
    INSERT INTO ticker_details (ticker, name, market, locale, type, currency, active, primary_exchange, last_updated)
    VALUES 
        ('AAPL', 'Apple Inc.', 'stocks', 'us', 'CS', 'USD', TRUE, 'NASDAQ', NOW()),
        ('MSFT', 'Microsoft Corporation', 'stocks', 'us', 'CS', 'USD', TRUE, 'NASDAQ', NOW()),
        ('GOOGL', 'Alphabet Inc.', 'stocks', 'us', 'CS', 'USD', TRUE, 'NASDAQ', NOW()),
        ('AMZN', 'Amazon.com Inc.', 'stocks', 'us', 'CS', 'USD', TRUE, 'NASDAQ', NOW()),
        ('TSLA', 'Tesla Inc.', 'stocks', 'us', 'CS', 'USD', TRUE, 'NASDAQ', NOW())
    ON CONFLICT (ticker) DO NOTHING;
    """)
    
    # Sample stock aggregates
    for symbol in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']:
        for i in range(10):
            timestamp = f"2023-01-{i+1:02d} 09:30:00-05:00"
            cursor.execute(f"""
            INSERT INTO stock_aggs (timestamp, symbol, open, high, low, close, volume, vwap, transactions, timeframe)
            VALUES ('{timestamp}', '{symbol}', 150.0, 155.0, 148.0, 152.0, 1000000, 151.5, 5000, '1m')
            ON CONFLICT DO NOTHING;
            """)
    
    # Sample feature metadata
    cursor.execute("""
    INSERT INTO feature_metadata (feature_name, description, formula, created_at, updated_at, version, is_active)
    VALUES 
        ('rsi_14', 'Relative Strength Index (14 periods)', 'RSI = 100 - (100 / (1 + RS))', NOW(), NOW(), '1.0', TRUE),
        ('macd_12_26_9', 'Moving Average Convergence Divergence', 'MACD = EMA(12) - EMA(26)', NOW(), NOW(), '1.0', TRUE),
        ('bb_20_2', 'Bollinger Bands (20 periods, 2 std dev)', 'BB = MA(20) ± 2 * StdDev(20)', NOW(), NOW(), '1.0', TRUE)
    ON CONFLICT (feature_name) DO NOTHING;
    """)
    
    # Sample features
    for symbol in ['AAPL', 'MSFT', 'GOOGL']:
        for i in range(5):
            timestamp = f"2023-01-{i+1:02d} 09:30:00-05:00"
            cursor.execute(f"""
            INSERT INTO features (timestamp, symbol, feature_name, feature_value, timeframe, feature_group)
            VALUES 
                ('{timestamp}', '{symbol}', 'rsi_14', 65.5, '1h', 'momentum'),
                ('{timestamp}', '{symbol}', 'macd_12_26_9', 0.75, '1h', 'momentum'),
                ('{timestamp}', '{symbol}', 'bb_20_2', 0.85, '1h', 'volatility')
            ON CONFLICT DO NOTHING;
            """)
    
    # Sample validation rules
    cursor.execute("""
    INSERT INTO validation_rules (rule_id, rule_name, data_type, description, parameters, is_active, created_at, updated_at)
    VALUES 
        (gen_random_uuid(), 'price_range', 'stock_aggs', 'Validate price range', '{"min_price": 0.01, "max_price": 10000.0}', TRUE, NOW(), NOW()),
        (gen_random_uuid(), 'volume_range', 'stock_aggs', 'Validate volume range', '{"min_volume": 1, "max_volume": 1000000000}', TRUE, NOW(), NOW()),
        (gen_random_uuid(), 'price_consistency', 'stock_aggs', 'Validate price consistency', '{"check_high_low": true, "check_open_close": true}', TRUE, NOW(), NOW())
    ON CONFLICT DO NOTHING;
    """)
    
    conn.commit()
    cursor.close()
    logger.info("Sample data loaded successfully")

def create_database_if_not_exists(conn, db_name: str) -> None:
    """
    Create the database if it doesn't exist.
    
    Args:
        conn: Database connection
        db_name: Name of the database to create
    """
    cursor = conn.cursor()
    cursor.execute(f"SELECT 1 FROM pg_database WHERE datname = '{db_name}'")
    if not cursor.fetchone():
        cursor.execute(f"CREATE DATABASE {db_name}")
        logger.info(f"Database {db_name} created")
    else:
        logger.info(f"Database {db_name} already exists")
    cursor.close()

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
        
        try:
            cursor.execute(
                f"""
                ALTER TABLE {table_name} SET (
                    timescaledb.compress,
                    timescaledb.compress_segmentby = 'symbol',
                    timescaledb.compress_orderby = '{time_column}'
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
        except psycopg2.Error as e:
            logger.warning(f"Error setting up compression for {table_name}: {e}")
    
    # Set up retention policy if enabled for this table
    retention_config = RETENTION_CONFIG.get(table_name, {})
    if retention_config.get('enabled', False):
        retention_period = retention_config.get('retention_period', '1 year')
        
        try:
            cursor.execute(
                f"""
                SELECT add_retention_policy('{table_name}', interval '{retention_period}');
                """
            )
            logger.info(f"Added retention policy for {table_name} with period {retention_period}")
        except psycopg2.Error as e:
            logger.warning(f"Error setting up retention policy for {table_name}: {e}")
    
    cursor.close()

def setup_redis(config: Dict[str, Any]) -> None:
    """
    Set up Redis for caching and real-time operations.
    This function configures Redis with appropriate persistence and memory settings.
    
    Args:
        config: Redis connection parameters
    """
    try:
        r = redis.Redis(**config)
        r.ping()  # Test connection
        
        # Clear existing keys (only in development/testing)
        environment = os.getenv('ENVIRONMENT', 'development')
        if environment != 'production':
            r.flushall()
            logger.info("Cleared Redis database")
        
        # Configure persistence
        persistence_config = get_redis_persistence_config()
        try:
            # Configure RDB persistence
            for seconds, changes in persistence_config['save_intervals']:
                r.config_set('save', f"{seconds} {changes}")  # e.g., "900 1" (save after 900 sec if at least 1 change)
            
            # Configure AOF persistence
            r.config_set('appendonly', 'yes' if persistence_config['appendonly'] else 'no')
            
            # Configure memory limits
            r.config_set('maxmemory', persistence_config['maxmemory'])
            r.config_set('maxmemory-policy', persistence_config['maxmemory_policy'])
            
            logger.info("Redis persistence configured successfully")
        except redis.ResponseError as e:
            if "protected mode" in str(e).lower():
                logger.warning("Unable to configure Redis persistence (running in protected mode)")
            else:
                logger.warning(f"Error configuring Redis persistence: {e}")
        
        logger.info("Redis setup completed successfully")
        
        # Initialize validation rules in Redis
        setup_validation_rules_in_redis(r)
        
    except redis.ConnectionError as e:
        logger.error(f"Failed to connect to Redis: {e}")
        logger.warning("Continuing without Redis setup")

def setup_validation_rules_in_redis(r: redis.Redis) -> None:
    """
    Set up validation rules in Redis for quick access.
    
    Args:
        r: Redis client
    """
    try:
        # Import validation rules
        from src.data_acquisition.validation.validation_rules import (
            get_stock_aggs_rules,
            get_quote_rules,
            get_trade_rules,
            get_options_rules,
            get_news_sentiment_rules,
            get_feature_rules,
            get_model_prediction_rules,
            get_trading_signal_rules,
            get_hft_validation_rules,
            get_day_trading_validation_rules,
            get_swing_trading_validation_rules,
            get_market_making_validation_rules
        )
        
        # Store validation rules in Redis
        r.set('validation_rules:stock_aggs', json.dumps(get_stock_aggs_rules()))
        r.set('validation_rules:quotes', json.dumps(get_quote_rules()))
        r.set('validation_rules:trades', json.dumps(get_trade_rules()))
        r.set('validation_rules:options', json.dumps(get_options_rules()))
        r.set('validation_rules:news_sentiment', json.dumps(get_news_sentiment_rules()))
        r.set('validation_rules:features', json.dumps(get_feature_rules()))
        r.set('validation_rules:model_predictions', json.dumps(get_model_prediction_rules()))
        r.set('validation_rules:trading_signals', json.dumps(get_trading_signal_rules()))
        
        # Store trading strategy-specific validation rules
        r.set('validation_rules:hft', json.dumps(get_hft_validation_rules()))
        r.set('validation_rules:day_trading', json.dumps(get_day_trading_validation_rules()))
        r.set('validation_rules:swing_trading', json.dumps(get_swing_trading_validation_rules()))
        r.set('validation_rules:market_making', json.dumps(get_market_making_validation_rules()))
        
        logger.info("Validation rules stored in Redis")
    except ImportError as e:
        logger.warning(f"Could not import validation rules: {e}")
    except redis.RedisError as e:
        logger.error(f"Failed to store validation rules in Redis: {e}")

def export_validation_rules_to_json(rules: Dict, filepath: str) -> None:
    """
    Export validation rules to a JSON file.
    
    Args:
        rules: Validation rules dictionary
        filepath: Path to the output JSON file
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(rules, f, indent=2)
        logger.info(f"Validation rules exported to {filepath}")
    except Exception as e:
        logger.error(f"Failed to export validation rules to {filepath}: {e}")

def main():
    """Main entry point for database setup."""
    parser = argparse.ArgumentParser(description='Set up the database for the Autonomous Trading System')
    parser.add_argument('--config-only', action='store_true', help='Only update configuration, do not create tables')
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
        
        # Create database if it doesn't exist
        create_database_if_not_exists(conn, db_params['database'])
        
        # If config-only flag is set, skip table creation
        if args.config_only:
            logger.info("Config-only flag set, skipping table creation")
            setup_redis(get_redis_connection_params())
            return
        
        # Connect to the specific database
        conn = psycopg2.connect(
            host=db_params['host'],
            port=db_params['port'],
            database=db_params['database'],
            user=db_params['user'],
            password=db_params['password']
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
        
        # Check if TimescaleDB extension is installed
        cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'timescaledb'")
        if not cursor.fetchone():
            logger.info("Creating TimescaleDB extension")
            cursor.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
        else:
            logger.info("TimescaleDB extension already installed")
        
        # Create extension for JSON operations
        cursor.execute("CREATE EXTENSION IF NOT EXISTS jsonb_plperl CASCADE")
        
        # Create extension for full-text search
        cursor.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm CASCADE")
        
        # Drop existing tables if requested
        if args.drop_existing:
            logger.warning("Dropping existing tables (--drop-existing flag specified)")
            for table_name in reversed(list(CREATE_TABLES_SQL.keys())):
                cursor.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
        
        # Create tables
        for table_name, sql in CREATE_TABLES_SQL.items():
            logger.info(f"Creating table: {table_name}")
            cursor.execute(sql)
            conn.commit()
        
        # Create hypertables
        for table_name, config in HYPERTABLE_CONFIG.items():
            logger.info(f"Setting up hypertable for {table_name}")
            create_hypertable(conn, table_name, config)
        
        # Load sample data if requested
        if args.sample_data:
            logger.info("Loading sample data")
            load_sample_data(conn)
        
        # Export validation rules to JSON files
        validation_rules_dir = os.path.join(os.path.dirname(__file__), '../../data/validation_rules')
        os.makedirs(validation_rules_dir, exist_ok=True)
        
        try:
            from src.data_acquisition.validation.validation_rules import (
                get_stock_aggs_rules,
                get_quote_rules,
                get_trade_rules,
                get_options_rules
            )
            
            export_validation_rules_to_json(get_stock_aggs_rules(), os.path.join(validation_rules_dir, 'stock_aggs_rules.json'))
            export_validation_rules_to_json(get_quote_rules(), os.path.join(validation_rules_dir, 'quote_rules.json'))
            export_validation_rules_to_json(get_trade_rules(), os.path.join(validation_rules_dir, 'trade_rules.json'))
            export_validation_rules_to_json(get_options_rules(), os.path.join(validation_rules_dir, 'options_rules.json'))
        except ImportError as e:
            logger.warning(f"Could not import validation rules for export: {e}")
        
        # Close connection
        conn.close()
        
        # Set up Redis
        logger.info("Setting up Redis")
        redis_params = get_redis_connection_params()
        setup_redis(redis_params)
        
        # Log backup configuration
        backup_config = get_backup_config()
        logger.info(f"Backup configuration: enabled={backup_config['enabled']}, schedule='{backup_config['schedule']}', retention={backup_config['retention_days']} days")
        
        logger.info("Database setup completed successfully")
        logger.info("Run with --sample-data flag to load sample data for testing")
        
    except Exception as e:
        logger.error(f"Error setting up database: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()