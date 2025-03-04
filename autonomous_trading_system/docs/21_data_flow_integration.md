# Data Flow Integration

This document explains how the Autonomous Trading System handles dynamic stock tickers, position tracking, and ensures seamless data flow from API endpoints to trading decisions.

## Dynamic Stock Ticker Handling

The system is designed to dynamically handle any stock ticker through several mechanisms:

### 1. Universe Selection

- The `ticker_selector.py` component dynamically selects the universe of tradable securities based on configurable criteria:
  ```python
  # Example from src/trading_strategy/selection/ticker_selector.py
  def select_tickers(self, criteria=None, max_symbols=50):
      """Dynamically select tickers based on criteria."""
      # Criteria can include market cap, liquidity, sector, etc.
      # Returns a dynamic list of tickers that meet the criteria
  ```

- The universe is refreshed on a configurable schedule (daily by default) to adapt to changing market conditions
- New tickers can be added and inactive ones removed automatically

### 2. Dynamic Data Collection

- API clients are designed to work with any valid ticker symbol:
  ```python
  # From polygon_client.py
  async def get_aggregates(self, ticker, multiplier, timespan, from_date, to_date):
      """Get aggregate bars for any ticker symbol."""
      endpoint = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
      return await self._make_request(endpoint)
  ```

- The database schema uses the `symbol` column as a key in all market data tables, allowing for any ticker to be stored
- Indexes are created on the `symbol` column for efficient querying

### 3. Symbol Normalization

- The system normalizes ticker symbols to ensure consistency:
  ```python
  # Example normalization function
  def normalize_symbol(symbol):
      """Normalize ticker symbols to a standard format."""
      return symbol.upper().strip()
  ```

- Special handling for different asset classes:
  - Stocks: Plain symbol (e.g., "AAPL")
  - Options: OCC format (e.g., "AAPL220121C00140000")
  - Crypto: Prefixed format (e.g., "X:BTCUSD")

## Daily Position Total Tracking

The system tracks daily position totals through several components:

### 1. Position Management

- The `positions` table in the database stores current positions with quantity and value:
  ```sql
  CREATE TABLE positions (
      position_id UUID PRIMARY KEY,
      symbol VARCHAR(16) NOT NULL,
      quantity NUMERIC(16,6) NOT NULL,
      entry_price NUMERIC(16,6) NOT NULL,
      current_price NUMERIC(16,6) NOT NULL,
      -- Other fields...
  );
  ```

- Position updates are timestamped to track changes over time
- The `portfolio_risk_manager.py` component enforces position limits:
  ```python
  # From src/trading_strategy/risk/portfolio_risk_manager.py
  def check_position_limits(self, symbol, proposed_quantity):
      """Check if a proposed position exceeds configured limits."""
      # Checks against max_position_size and other risk parameters
  ```

### 2. Daily Snapshots

- The system takes daily snapshots of all positions for historical tracking:
  ```python
  # Example daily snapshot function
  async def create_daily_position_snapshot(self):
      """Create a snapshot of all positions at end of day."""
      # Stores a point-in-time record of all positions
  ```

- These snapshots are used for:
  - Performance reporting
  - Risk analysis
  - Compliance requirements

### 3. Exposure Tracking

- The system tracks exposure across different dimensions:
  - By symbol
  - By sector
  - By strategy
  - By asset class

- Exposure limits are configurable:
  ```python
  # Example from trading_config.py
  EXPOSURE_LIMITS = {
      "max_single_symbol_percentage": 0.05,  # Max 5% in any single symbol
      "max_sector_percentage": 0.25,         # Max 25% in any sector
      "max_asset_class_percentage": 0.60,    # Max 60% in any asset class
  }
  ```

## Data Endpoint Integration Factors

The system ensures seamless data flow from API endpoints to trading decisions through several factors:

### 1. Data Normalization

- All external data is normalized to a consistent internal format:
  ```python
  # Example from polygon_client.py
  def _normalize_aggregate_data(self, raw_data):
      """Convert Polygon aggregate format to internal format."""
      return {
          "timestamp": convert_to_timestamp(raw_data["t"]),
          "open": raw_data["o"],
          "high": raw_data["h"],
          "low": raw_data["l"],
          "close": raw_data["c"],
          "volume": raw_data["v"],
          # Additional fields...
      }
  ```

- This ensures that data from different sources (Polygon, Unusual Whales, etc.) can be processed uniformly

### 2. Data Pipeline Integration

- The data pipeline connects API endpoints to database storage:
  ```python
  # Example from data_pipeline.py
  async def process_market_data(self, symbol, timeframe):
      """Process market data from API to database."""
      # 1. Fetch data from API
      raw_data = await self.api_client.get_aggregates(symbol, timeframe)
      
      # 2. Normalize data
      normalized_data = self._normalize_data(raw_data)
      
      # 3. Store in database
      await self.storage_manager.store_aggregates(normalized_data)
      
      # 4. Trigger feature calculation
      await self.feature_calculator.calculate_features(symbol, timeframe)
  ```

- The pipeline handles:
  - Data fetching
  - Normalization
  - Storage
  - Feature calculation
  - Event triggering

### 3. Real-time vs. Historical Data

- The system handles both real-time and historical data:
  - WebSocket connections for real-time data
  - REST API calls for historical data
  - Seamless transition between the two

- Real-time data flow:
  ```
  WebSocket → Message Queue → Processor → Database → Feature Calculator → Trading Strategy
  ```

- Historical data flow:
  ```
  REST API → Batch Processor → Database → Feature Calculator → Backtesting Engine
  ```

### 4. Data Validation and Quality Control

- All incoming data is validated before processing:
  ```python
  # Example from data_validator.py
  def validate_ohlcv_data(self, data):
      """Validate OHLCV data meets quality standards."""
      # Check for missing values
      # Ensure high >= low
      # Verify timestamp sequence
      # Check for outliers
  ```

- Quality metrics are tracked:
  - Completeness (missing data points)
  - Timeliness (data delay)
  - Accuracy (compared to other sources)

### 5. End-to-End Data Flow Example

Here's how data flows through the system for a typical trading scenario:

1. **Data Acquisition**:
   - Polygon WebSocket streams real-time trades for selected tickers
   - Unusual Whales API provides options flow alerts

2. **Data Processing**:
   - Raw data is normalized and validated
   - OHLCV bars are constructed from trade data
   - Data is stored in appropriate database tables

3. **Feature Engineering**:
   - Technical indicators are calculated from price data
   - Options flow metrics are derived from Unusual Whales data
   - Features are stored in the feature store

4. **Model Inference**:
   - ML models consume features to generate predictions
   - Predictions are scored and filtered

5. **Trading Strategy**:
   - Signals are generated based on predictions and rules
   - Position sizing is determined based on risk parameters
   - Orders are generated and sent to execution

6. **Monitoring and Feedback**:
   - Execution quality is analyzed
   - Performance metrics are calculated
   - System adjusts based on feedback

## Configuration and Flexibility

The system is designed to be highly configurable to adapt to different trading requirements:

### 1. Dynamic Configuration

- Configuration parameters can be updated without code changes:
  ```python
  # Example from config/trading_config.py
  TRADING_CONFIG = {
      "universe_selection": {
          "method": "market_cap",
          "max_symbols": 50,
          "refresh_frequency": "daily"
      },
      "position_limits": {
          "max_position_size": 1000.0,
          "max_position_percentage": 0.05
      },
      # Other configuration...
  }
  ```

### 2. Strategy Parameterization

- Trading strategies are parameterized for flexibility:
  ```python
  # Example strategy configuration
  MOMENTUM_STRATEGY_CONFIG = {
      "lookback_period": 20,
      "threshold": 0.02,
      "timeframes": ["1d", "1h"],
      # Other parameters...
  }
  ```

### 3. Asset Class Adaptability

- The system adapts to different asset classes:
  - Stocks: Uses stock-specific endpoints and data models
  - Options: Incorporates options-specific data like implied volatility
  - Crypto: Handles 24/7 trading and exchange-specific data

## Conclusion

The Autonomous Trading System is designed to handle dynamic stock tickers and daily position tracking through a flexible, configurable architecture. The data flow from API endpoints to trading decisions is seamless, with proper normalization, validation, and integration at each step.

Key strengths of the system include:
- Dynamic universe selection
- Comprehensive position tracking
- Consistent data normalization
- Robust data validation
- Configurable parameters
- Support for multiple asset classes

These design choices ensure that the system can adapt to changing market conditions, trade any ticker symbol, and maintain accurate position tracking while enforcing risk limits.