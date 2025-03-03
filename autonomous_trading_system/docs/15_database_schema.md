# Autonomous Trading System - Database Schema Documentation

## Overview

This document provides a comprehensive overview of the database schema used in the Autonomous Trading System. The system uses TimescaleDB (PostgreSQL extension) as its primary database for time-series data storage and Redis for caching and real-time operations.

## Database Architecture

The database architecture follows a hybrid approach:

1. **TimescaleDB (PostgreSQL)**: Primary database for persistent storage of time-series data, trading records, model metadata, and system configuration.
2. **Redis**: In-memory database for caching, real-time feature calculations, and message queuing.

## TimescaleDB Schema

### Market Data Tables

#### `stock_aggs` (Polygon Aggregates)
| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `timestamp` | TIMESTAMPTZ | Time of the price data point | NOT NULL |
| `symbol` | VARCHAR(16) | Ticker symbol | NOT NULL |
| `open` | NUMERIC(16,6) | Opening price | NOT NULL |
| `high` | NUMERIC(16,6) | Highest price | NOT NULL |
| `low` | NUMERIC(16,6) | Lowest price | NOT NULL |
| `close` | NUMERIC(16,6) | Closing price | NOT NULL |
| `volume` | BIGINT | Trading volume | NOT NULL |
| `vwap` | NUMERIC(16,6) | Volume-weighted average price | |
| `transactions` | INTEGER | Number of transactions | |
| `timeframe` | VARCHAR(8) | Data timeframe (1m, 5m, 15m, 1h, 1d) | NOT NULL |
| `source` | VARCHAR(32) | Data source (e.g., 'polygon') | DEFAULT 'polygon' |
| `multiplier` | INTEGER | Timespan multiplier from API | DEFAULT 1 |
| `timespan_unit` | VARCHAR(16) | Timespan unit (minute, hour, day, week, month, quarter, year) | DEFAULT 'minute' |
| `adjusted` | BOOLEAN | Whether prices are adjusted for splits/dividends | DEFAULT FALSE |
| `otc` | BOOLEAN | Whether the stock is OTC | DEFAULT FALSE |

**Indexes**:
- Primary hypertable partition on `timestamp` with chunk interval of 1 day
- Index on (`symbol`, `timestamp`, `timeframe`)
- Index on (`adjusted`, `symbol`, `timestamp`)

#### `crypto_aggs` (Polygon Crypto Aggregates)
| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `timestamp` | TIMESTAMPTZ | Time of the price data point | NOT NULL |
| `symbol` | VARCHAR(16) | Crypto pair symbol (e.g., 'X:BTCUSD') | NOT NULL |
| `open` | NUMERIC(20,8) | Opening price | NOT NULL |
| `high` | NUMERIC(20,8) | Highest price | NOT NULL |
| `low` | NUMERIC(20,8) | Lowest price | NOT NULL |
| `close` | NUMERIC(20,8) | Closing price | NOT NULL |
| `volume` | NUMERIC(24,8) | Trading volume | NOT NULL |
| `vwap` | NUMERIC(20,8) | Volume-weighted average price | |
| `transactions` | INTEGER | Number of transactions | |
| `timeframe` | VARCHAR(8) | Data timeframe (1m, 5m, 15m, 1h, 1d) | NOT NULL |
| `source` | VARCHAR(32) | Data source (e.g., 'polygon') | DEFAULT 'polygon' |
| `multiplier` | INTEGER | Timespan multiplier from API | DEFAULT 1 |
| `timespan_unit` | VARCHAR(16) | Timespan unit (minute, hour, day, week, month, quarter, year) | DEFAULT 'minute' |
| `exchange` | VARCHAR(16) | Exchange identifier | |

**Indexes**:
- Primary hypertable partition on `timestamp` with chunk interval of 1 day
- Index on (`symbol`, `timestamp`, `timeframe`)
- Index on (`exchange`, `symbol`, `timestamp`)

#### `quotes` (Polygon Quotes)
| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `timestamp` | TIMESTAMPTZ | Time of the quote | NOT NULL |
| `symbol` | VARCHAR(16) | Ticker symbol | NOT NULL |
| `bid_price` | NUMERIC(16,6) | Bid price | NOT NULL |
| `ask_price` | NUMERIC(16,6) | Ask price | NOT NULL |
| `bid_size` | INTEGER | Bid size | NOT NULL |
| `ask_size` | INTEGER | Ask size | NOT NULL |
| `exchange` | VARCHAR(8) | Exchange code | |
| `conditions` | VARCHAR[] | Quote conditions | |
| `sequence_number` | BIGINT | Sequence number from exchange | |
| `tape` | CHAR(1) | Tape identifier | |
| `source` | VARCHAR(32) | Data source (e.g., 'polygon') | DEFAULT 'polygon' |

**Indexes**:
- Primary hypertable partition on `timestamp` with chunk interval of 1 hour
- Index on (`symbol`, `timestamp`)

#### `trades` (Polygon Trades)
| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `timestamp` | TIMESTAMPTZ | Time of the trade | NOT NULL |
| `symbol` | VARCHAR(16) | Ticker symbol | NOT NULL |
| `price` | NUMERIC(16,6) | Trade price | NOT NULL |
| `size` | INTEGER | Trade size | NOT NULL |
| `exchange` | VARCHAR(8) | Exchange code | NOT NULL |
| `conditions` | VARCHAR[] | Trade conditions | |
| `tape` | CHAR(1) | Tape (A/B/C) | |
| `sequence_number` | BIGINT | Sequence number from exchange | |
| `trade_id` | VARCHAR(32) | Unique trade identifier | |
| `source` | VARCHAR(32) | Data source (e.g., 'polygon') | DEFAULT 'polygon' |

**Indexes**:
- Primary hypertable partition on `timestamp` with chunk interval of 1 hour
- Index on (`symbol`, `timestamp`)
- Index on (`exchange`, `timestamp`)

#### `options_aggs` (Polygon Options Aggregates)
| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `timestamp` | TIMESTAMPTZ | Time of the data point | NOT NULL |
| `symbol` | VARCHAR(32) | Option symbol (OCC format) | NOT NULL |
| `underlying` | VARCHAR(16) | Underlying asset symbol | NOT NULL |
| `expiration` | DATE | Expiration date | NOT NULL |
| `strike` | NUMERIC(16,6) | Strike price | NOT NULL |
| `option_type` | CHAR(1) | Option type (C/P) | NOT NULL |
| `open` | NUMERIC(16,6) | Opening price | |
| `high` | NUMERIC(16,6) | Highest price | |
| `low` | NUMERIC(16,6) | Lowest price | |
| `close` | NUMERIC(16,6) | Closing price | |
| `volume` | INTEGER | Trading volume | |
| `open_interest` | INTEGER | Open interest | |
| `timeframe` | VARCHAR(8) | Data timeframe | NOT NULL |
| `multiplier` | INTEGER | Timespan multiplier from API | DEFAULT 1 |
| `timespan_unit` | VARCHAR(16) | Timespan unit (minute, hour, day, week, month, quarter, year) | DEFAULT 'minute' |
| `source` | VARCHAR(32) | Data source (e.g., 'polygon') | DEFAULT 'polygon' |

**Indexes**:
- Primary hypertable partition on `timestamp` with chunk interval of 1 day
- Index on (`underlying`, `expiration`, `strike`, `option_type`)
- Index on (`symbol`, `timestamp`)

#### `unusual_whales_options` (Unusual Whales Alerts)
| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `timestamp` | TIMESTAMPTZ | Time of the data point | NOT NULL |
| `symbol` | VARCHAR(32) | Option symbol | NOT NULL |
| `underlying` | VARCHAR(16) | Underlying asset symbol | NOT NULL |
| `underlying_price` | NUMERIC(16,6) | Price of underlying at time of alert | |
| `expiration` | DATE | Expiration date | NOT NULL |
| `strike` | NUMERIC(16,6) | Strike price | NOT NULL |
| `option_type` | CHAR(1) | Option type (C/P) | NOT NULL |
| `premium` | NUMERIC(16,6) | Option premium | NOT NULL |
| `premium_type` | VARCHAR(8) | Premium type (debit/credit) | NOT NULL |
| `sentiment` | VARCHAR(8) | Bullish/Bearish sentiment | NOT NULL |
| `unusual_score` | INTEGER | Unusual Whales proprietary score (1-100) | |
| `volume` | INTEGER | Volume at time of alert | NOT NULL |
| `open_interest` | INTEGER | Open interest at time of alert | |
| `volume_oi_ratio` | NUMERIC(10,6) | Volume to open interest ratio | |
| `implied_volatility` | NUMERIC(10,6) | Implied volatility at time of alert | |
| `days_to_expiration` | INTEGER | Days until expiration at time of alert | NOT NULL |
| `trade_type` | VARCHAR(16) | Type of trade (sweep, block, etc.) | |
| `size_notation` | VARCHAR(16) | Size notation (small, medium, large, etc.) | |
| `alert_id` | VARCHAR(64) | Unique alert identifier from Unusual Whales | |
| `sector` | VARCHAR(32) | Sector of the underlying | |
| `source` | VARCHAR(32) | Data source | DEFAULT 'unusual_whales' |

**Indexes**:
- Primary hypertable partition on `timestamp` with chunk interval of 1 day
- Index on (`underlying`, `expiration`, `strike`, `option_type`)
- Index on (`symbol`, `timestamp`)
- Index on (`unusual_score`) WHERE unusual_score IS NOT NULL

### Reference Data Tables

#### `ticker_details`
| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `ticker` | VARCHAR(16) | Ticker symbol | PRIMARY KEY |
| `name` | VARCHAR(128) | Company/Asset name | NOT NULL |
| `market` | VARCHAR(32) | Market type (stocks, crypto, forex, etc.) | NOT NULL |
| `locale` | VARCHAR(16) | Locale (us, global) | NOT NULL |
| `type` | VARCHAR(16) | Security type (CS, ETF, etc.) | NOT NULL |
| `currency` | VARCHAR(8) | Trading currency | NOT NULL |
| `active` | BOOLEAN | Whether the ticker is active | DEFAULT TRUE |
| `primary_exchange` | VARCHAR(32) | Primary exchange | |
| `last_updated` | TIMESTAMPTZ | Last update timestamp | NOT NULL |
| `description` | TEXT | Company/Asset description | |
| `sic_code` | VARCHAR(8) | Standard Industrial Classification code | |
| `sic_description` | VARCHAR(256) | SIC description | |
| `ticker_root` | VARCHAR(16) | Root symbol | |
| `homepage_url` | VARCHAR(256) | Company homepage URL | |
| `total_employees` | INTEGER | Number of employees | |
| `list_date` | DATE | Date listed on exchange | |
| `share_class_shares_outstanding` | BIGINT | Shares outstanding | |
| `weighted_shares_outstanding` | BIGINT | Weighted shares outstanding | |
| `market_cap` | BIGINT | Market capitalization | |
| `phone_number` | VARCHAR(32) | Company phone number | |
| `address` | JSONB | Company address | |
| `metadata` | JSONB | Additional metadata | |

**Indexes**:
- Index on (`market`, `active`)
- Index on (`type`, `active`)

#### `market_holidays`
| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `date` | DATE | Holiday date | NOT NULL |
| `name` | VARCHAR(64) | Holiday name | NOT NULL |
| `market` | VARCHAR(32) | Market (us_equity, us_options, forex, crypto) | NOT NULL |
| `status` | VARCHAR(16) | Market status (closed, early_close, late_open) | NOT NULL |
| `open_time` | TIME | Market open time (for early close/late open) | |
| `close_time` | TIME | Market close time (for early close) | |
| `year` | INTEGER | Year of the holiday | NOT NULL |

**Indexes**:
- Primary key on (`date`, `market`)
- Index on (`year`, `market`)

#### `market_status`
| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `timestamp` | TIMESTAMPTZ | Status timestamp | NOT NULL |
| `market` | VARCHAR(32) | Market (us_equity, us_options, forex, crypto) | NOT NULL |
| `status` | VARCHAR(16) | Current status (open, closed, extended_hours) | NOT NULL |
| `next_open` | TIMESTAMPTZ | Next market open time | |
| `next_close` | TIMESTAMPTZ | Next market close time | |
| `early_close` | BOOLEAN | Whether today has an early close | DEFAULT FALSE |
| `late_open` | BOOLEAN | Whether today has a late open | DEFAULT FALSE |

**Indexes**:
- Primary key on (`timestamp`, `market`)
- Index on (`market`, `status`)

#### `news_articles`
| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `article_id` | VARCHAR(64) | Unique article identifier | PRIMARY KEY |
| `published_utc` | TIMESTAMPTZ | Publication timestamp | NOT NULL |
| `title` | VARCHAR(512) | Article title | NOT NULL |
| `author` | VARCHAR(128) | Article author | |
| `article_url` | VARCHAR(512) | URL to the article | NOT NULL |
| `tickers` | VARCHAR[] | Related ticker symbols | |
| `image_url` | VARCHAR(512) | URL to article image | |
| `description` | TEXT | Article description/summary | |
| `keywords` | VARCHAR[] | Article keywords | |
| `source` | VARCHAR(64) | News source | NOT NULL |

**Indexes**:
- Index on (`published_utc`)
- Index on (`tickers`) using GIN

### Feature Engineering Tables

#### `features`
| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `timestamp` | TIMESTAMPTZ | Time of the feature calculation | NOT NULL |
| `symbol` | VARCHAR(16) | Ticker symbol | NOT NULL |
| `feature_name` | VARCHAR(64) | Name of the feature | NOT NULL |
| `feature_value` | NUMERIC(20,8) | Value of the feature | NOT NULL |
| `timeframe` | VARCHAR(8) | Data timeframe | NOT NULL |
| `feature_group` | VARCHAR(32) | Group the feature belongs to | NOT NULL |

**Indexes**:
- Primary hypertable partition on `timestamp` with chunk interval of 1 day
- Index on (`symbol`, `feature_name`, `timestamp`, `timeframe`)
- Index on (`feature_group`, `symbol`, `timestamp`)

#### `feature_metadata`
| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `feature_name` | VARCHAR(64) | Name of the feature | PRIMARY KEY |
| `description` | TEXT | Description of the feature | NOT NULL |
| `formula` | TEXT | Formula or algorithm used | |
| `parameters` | JSONB | Parameters used in calculation | |
| `created_at` | TIMESTAMPTZ | Creation timestamp | NOT NULL |
| `updated_at` | TIMESTAMPTZ | Last update timestamp | NOT NULL |
| `version` | VARCHAR(16) | Version of the feature | NOT NULL |
| `is_active` | BOOLEAN | Whether the feature is active | DEFAULT TRUE |

### Model Training Tables

#### `models`
| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `model_id` | UUID | Unique model identifier | PRIMARY KEY |
| `model_name` | VARCHAR(64) | Name of the model | NOT NULL |
| `model_type` | VARCHAR(32) | Type of model (e.g., 'xgboost', 'lstm') | NOT NULL |
| `target` | VARCHAR(64) | Target variable | NOT NULL |
| `features` | VARCHAR[] | Features used | NOT NULL |
| `parameters` | JSONB | Model parameters | NOT NULL |
| `metrics` | JSONB | Performance metrics | NOT NULL |
| `created_at` | TIMESTAMPTZ | Creation timestamp | NOT NULL |
| `trained_at` | TIMESTAMPTZ | Training completion timestamp | NOT NULL |
| `version` | VARCHAR(16) | Model version | NOT NULL |
| `status` | VARCHAR(16) | Model status | NOT NULL |
| `file_path` | VARCHAR(256) | Path to model file | NOT NULL |

**Indexes**:
- Index on (`model_name`, `version`)
- Index on (`status`, `model_type`)

#### `model_training_runs`
| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `run_id` | UUID | Unique run identifier | PRIMARY KEY |
| `model_id` | UUID | Reference to model | FOREIGN KEY |
| `start_time` | TIMESTAMPTZ | Start of training | NOT NULL |
| `end_time` | TIMESTAMPTZ | End of training | |
| `status` | VARCHAR(16) | Run status | NOT NULL |
| `parameters` | JSONB | Training parameters | NOT NULL |
| `metrics` | JSONB | Training metrics | |
| `logs` | TEXT | Training logs | |

**Indexes**:
- Index on (`model_id`, `start_time`)
- Index on (`status`)

### Trading Strategy Tables

#### `trading_signals`
| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `signal_id` | UUID | Unique signal identifier | PRIMARY KEY |
| `timestamp` | TIMESTAMPTZ | Signal generation time | NOT NULL |
| `symbol` | VARCHAR(16) | Ticker symbol | NOT NULL |
| `signal_type` | VARCHAR(16) | Type of signal (buy/sell) | NOT NULL |
| `confidence` | NUMERIC(5,4) | Signal confidence (0-1) | NOT NULL |
| `model_id` | UUID | Model that generated the signal | FOREIGN KEY |
| `timeframe` | VARCHAR(8) | Signal timeframe | NOT NULL |
| `parameters` | JSONB | Signal parameters | |
| `features_snapshot` | JSONB | Feature values at signal time | |

**Indexes**:
- Index on (`symbol`, `timestamp`, `signal_type`)
- Index on (`model_id`, `timestamp`)

#### `orders`
| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `order_id` | UUID | Unique order identifier | PRIMARY KEY |
| `external_order_id` | VARCHAR(64) | ID from broker (e.g., Alpaca) | UNIQUE |
| `timestamp` | TIMESTAMPTZ | Order creation time | NOT NULL |
| `symbol` | VARCHAR(16) | Ticker symbol | NOT NULL |
| `order_type` | VARCHAR(16) | Type of order | NOT NULL |
| `side` | VARCHAR(8) | Buy/Sell | NOT NULL |
| `quantity` | NUMERIC(16,6) | Order quantity | NOT NULL |
| `price` | NUMERIC(16,6) | Order price | |
| `status` | VARCHAR(16) | Order status | NOT NULL |
| `signal_id` | UUID | Signal that triggered the order | FOREIGN KEY |
| `strategy_id` | UUID | Strategy that placed the order | FOREIGN KEY |
| `filled_quantity` | NUMERIC(16,6) | Quantity filled | DEFAULT 0 |
| `filled_price` | NUMERIC(16,6) | Average fill price | |
| `commission` | NUMERIC(10,6) | Order commission | DEFAULT 0 |
| `updated_at` | TIMESTAMPTZ | Last update timestamp | NOT NULL |

**Indexes**:
- Index on (`symbol`, `timestamp`)
- Index on (`status`, `timestamp`)
- Index on (`signal_id`)
- Index on (`strategy_id`, `timestamp`)

#### `positions`
| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `position_id` | UUID | Unique position identifier | PRIMARY KEY |
| `symbol` | VARCHAR(16) | Ticker symbol | NOT NULL |
| `quantity` | NUMERIC(16,6) | Position quantity | NOT NULL |
| `entry_price` | NUMERIC(16,6) | Average entry price | NOT NULL |
| `current_price` | NUMERIC(16,6) | Current market price | NOT NULL |
| `entry_time` | TIMESTAMPTZ | Position entry time | NOT NULL |
| `last_update` | TIMESTAMPTZ | Last update timestamp | NOT NULL |
| `strategy_id` | UUID | Strategy managing the position | FOREIGN KEY |
| `status` | VARCHAR(16) | Position status | NOT NULL |
| `pnl` | NUMERIC(16,6) | Current P&L | NOT NULL |
| `pnl_percentage` | NUMERIC(10,6) | P&L as percentage | NOT NULL |
| `metadata` | JSONB | Additional position data | |

**Indexes**:
- Index on (`symbol`, `status`)
- Index on (`strategy_id`, `status`)

### Monitoring Tables

#### `system_metrics`
| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `timestamp` | TIMESTAMPTZ | Metric timestamp | NOT NULL |
| `metric_name` | VARCHAR(64) | Name of the metric | NOT NULL |
| `metric_value` | NUMERIC(20,8) | Value of the metric | NOT NULL |
| `component` | VARCHAR(32) | System component | NOT NULL |
| `host` | VARCHAR(64) | Host machine | NOT NULL |
| `tags` | JSONB | Additional tags | |

**Indexes**:
- Primary hypertable partition on `timestamp` with chunk interval of 1 hour
- Index on (`metric_name`, `component`, `timestamp`)
- Index on (`host`, `timestamp`)

#### `trading_metrics`
| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `timestamp` | TIMESTAMPTZ | Metric timestamp | NOT NULL |
| `metric_name` | VARCHAR(64) | Name of the metric | NOT NULL |
| `metric_value` | NUMERIC(20,8) | Value of the metric | NOT NULL |
| `symbol` | VARCHAR(16) | Ticker symbol | |
| `strategy_id` | UUID | Strategy identifier | |
| `timeframe` | VARCHAR(8) | Metric timeframe | |
| `tags` | JSONB | Additional tags | |

**Indexes**:
- Primary hypertable partition on `timestamp` with chunk interval of 1 day
- Index on (`metric_name`, `timestamp`)
- Index on (`symbol`, `metric_name`, `timestamp`) WHERE symbol IS NOT NULL
- Index on (`strategy_id`, `metric_name`, `timestamp`) WHERE strategy_id IS NOT NULL

#### `alerts`
| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `alert_id` | UUID | Unique alert identifier | PRIMARY KEY |
| `timestamp` | TIMESTAMPTZ | Alert timestamp | NOT NULL |
| `alert_type` | VARCHAR(32) | Type of alert | NOT NULL |
| `severity` | VARCHAR(16) | Alert severity | NOT NULL |
| `message` | TEXT | Alert message | NOT NULL |
| `component` | VARCHAR(32) | System component | NOT NULL |
| `status` | VARCHAR(16) | Alert status | NOT NULL |
| `resolved_at` | TIMESTAMPTZ | Resolution timestamp | |
| `metadata` | JSONB | Additional alert data | |

**Indexes**:
- Index on (`timestamp`, `severity`)
- Index on (`status`, `timestamp`)
- Index on (`component`, `timestamp`)

### Emergency Stop Tables

#### `emergency_events`
| Column | Type | Description | Constraints |
|--------|------|-------------|-------------|
| `event_id` | UUID | Unique event identifier | PRIMARY KEY |
| `timestamp` | TIMESTAMPTZ | Event timestamp | NOT NULL |
| `trigger_type` | VARCHAR(32) | What triggered the emergency | NOT NULL |
| `severity` | VARCHAR(16) | Event severity | NOT NULL |
| `description` | TEXT | Event description | NOT NULL |
| `action_taken` | VARCHAR(32) | Action taken | NOT NULL |
| `positions_closed` | INTEGER | Number of positions closed | |
| `total_value` | NUMERIC(20,8) | Total value involved | |
| `resolution_time` | TIMESTAMPTZ | When emergency was resolved | |
| `triggered_by` | VARCHAR(64) | User or system that triggered | NOT NULL |
| `metadata` | JSONB | Additional event data | |

**Indexes**:
- Index on (`timestamp`, `severity`)
- Index on (`trigger_type`, `timestamp`)

## Redis Schema

Redis is used for caching and real-time operations. The following key patterns are used:

### Feature Cache
- `feature:{symbol}:{feature_name}:{timeframe}` - Latest feature value
- `feature_history:{symbol}:{feature_name}:{timeframe}` - Recent history (sorted set)

### Model Cache
- `model:{model_id}:metadata` - Model metadata (hash)
- `model:{model_id}:predictions:{symbol}` - Recent predictions (sorted set)

### Trading State
- `active_positions` - Set of symbols with active positions
- `position:{symbol}` - Position details (hash)
- `order:{order_id}` - Order details (hash)
- `pending_orders` - Set of pending order IDs

### System State
- `system:status` - Overall system status (hash)
- `component:{component_name}:status` - Component status (hash)
- `emergency:status` - Emergency status flag (string)

### WebSocket Data Cache
- `ws:last_trade:{symbol}` - Last trade data for symbol
- `ws:last_quote:{symbol}` - Last quote data for symbol
- `ws:agg:{symbol}:{timeframe}` - Latest aggregate data

## Database Migrations

Database migrations are managed using Alembic. Migration scripts are stored in:
`/src/utils/database/migrations/versions/`

Each migration script includes:
- Upgrade operations (forward migration)
- Downgrade operations (rollback)
- Dependencies on previous migrations

## Backup and Recovery

The database backup strategy includes:
1. Daily full backups of TimescaleDB
2. Continuous WAL (Write-Ahead Log) archiving
3. Point-in-time recovery capability
4. Regular backup testing and validation

## Performance Considerations

1. **Partitioning**: Time-series tables are partitioned by time using TimescaleDB hypertables
2. **Compression**: Older chunks are compressed to reduce storage requirements
3. **Retention Policies**: Automated data retention policies based on data importance
4. **Indexing Strategy**: Carefully designed indexes to support common query patterns
5. **Query Optimization**: Prepared statements and optimized queries

## Security

1. **Access Control**: Role-based access control for database users
2. **Encryption**: Data-at-rest encryption for sensitive information
3. **Audit Logging**: Database activity monitoring and audit logging
4. **Network Security**: Database accessible only from within the application network

## API to Database Flow

The system ensures seamless flow from API endpoints to database tables:

1. **Polygon REST API → Database**:
   - Aggregates endpoint → stock_aggs/crypto_aggs/options_aggs tables
   - Trades endpoint → trades table
   - Quotes endpoint → quotes table
   - Reference data endpoints → ticker_details table
   - News endpoint → news_articles table
   - Market status endpoints → market_status and market_holidays tables

2. **Polygon WebSocket → Database**:
   - Real-time trades → trades table + Redis cache
   - Real-time quotes → quotes table + Redis cache
   - Real-time aggregates → stock_aggs/crypto_aggs tables + Redis cache

3. **Unusual Whales API → Database**:
   - Options flow endpoint → unusual_whales_options table
   - Historical flow endpoint → unusual_whales_options table (with historical flag)

## Conclusion

This database schema is designed to support the high-performance, scalable requirements of the Autonomous Trading System. It provides efficient storage and retrieval of time-series market data, features, models, and trading records while maintaining data integrity and security. The schema is specifically designed to match the exact data formats provided by API clients like Polygon and Unusual Whales, ensuring seamless integration with these data sources. The tables are structured to capture all relevant fields from the API responses, allowing for complete data flow from API endpoints to database storage.