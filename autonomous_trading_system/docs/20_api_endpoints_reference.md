# API Endpoints Reference

This document provides a comprehensive list of endpoints for the external APIs used in the Autonomous Trading System.

## Polygon.io API Endpoints

Polygon.io provides market data through both REST API and WebSocket connections.

### Polygon REST API Endpoints

#### Authentication
- Base URL: `https://api.polygon.io/`
- Authentication: API Key as query parameter `?apiKey=YOUR_API_KEY`

#### Stock Market Data

1. **Aggregates (OHLCV)**
   - Endpoint: `GET /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}`
   - Description: Get aggregate bars for a stock over a given date range in custom time window sizes
   - Parameters:
     - `ticker`: Stock symbol (e.g., AAPL)
     - `multiplier`: Size of the timespan multiplier (e.g., 1, 2, 5)
     - `timespan`: Size of the time window (minute, hour, day, week, month, quarter, year)
     - `from`: From date (YYYY-MM-DD or Unix MS)
     - `to`: To date (YYYY-MM-DD or Unix MS)
     - `adjusted`: Whether to adjust for splits (default: true)
     - `sort`: Sort order (default: asc)
     - `limit`: Limit of results (default: 5000)
   - Response: OHLCV data with volume, VWAP, and transaction count

2. **Daily Open/Close**
   - Endpoint: `GET /v1/open-close/{ticker}/{date}`
   - Description: Get the open, close, high, and low for a specific stock symbol on a certain date
   - Parameters:
     - `ticker`: Stock symbol
     - `date`: Date in format YYYY-MM-DD
     - `adjusted`: Whether to adjust for splits (default: true)
   - Response: Open, close, high, low, volume data for the specified date

3. **Previous Close**
   - Endpoint: `GET /v2/aggs/ticker/{ticker}/prev`
   - Description: Get the previous day's open, high, low, and close for a specific stock ticker
   - Parameters:
     - `ticker`: Stock symbol
     - `adjusted`: Whether to adjust for splits (default: true)
   - Response: Previous day's OHLCV data

4. **Trades**
   - Endpoint: `GET /v3/trades/{ticker}`
   - Description: Get trades for a ticker symbol in a given time range
   - Parameters:
     - `ticker`: Stock symbol
     - `timestamp`: Timestamp to filter by (RFC3339 or Unix MS)
     - `timestamp.gte`: Timestamp greater than or equal to
     - `timestamp.lte`: Timestamp less than or equal to
     - `order`: Order of results (asc, desc)
     - `limit`: Limit of results (default: 10, max: 50000)
     - `sort`: Sort field
   - Response: List of trades with price, size, exchange, conditions

5. **Quotes**
   - Endpoint: `GET /v3/quotes/{ticker}`
   - Description: Get NBBO quotes for a ticker symbol in a given time range
   - Parameters:
     - `ticker`: Stock symbol
     - `timestamp`: Timestamp to filter by (RFC3339 or Unix MS)
     - `timestamp.gte`: Timestamp greater than or equal to
     - `timestamp.lte`: Timestamp less than or equal to
     - `order`: Order of results (asc, desc)
     - `limit`: Limit of results (default: 10, max: 50000)
     - `sort`: Sort field
   - Response: List of quotes with bid price, ask price, bid size, ask size, exchange, conditions

6. **Last Trade**
   - Endpoint: `GET /v2/last/trade/{ticker}`
   - Description: Get the most recent trade for a ticker
   - Parameters:
     - `ticker`: Stock symbol
   - Response: Last trade with price, size, exchange, timestamp

7. **Last Quote**
   - Endpoint: `GET /v2/last/nbbo/{ticker}`
   - Description: Get the most recent NBBO quote for a ticker
   - Parameters:
     - `ticker`: Stock symbol
   - Response: Last quote with bid price, ask price, bid size, ask size, timestamp

#### Options Market Data

1. **Options Contracts**
   - Endpoint: `GET /v3/reference/options/contracts`
   - Description: Get available options contracts for a given underlying ticker
   - Parameters:
     - `underlying_ticker`: Underlying stock symbol
     - `expiration_date`: Expiration date (YYYY-MM-DD)
     - `contract_type`: Type of contract (call, put)
     - `strike_price`: Strike price
     - `order`: Order of results (asc, desc)
     - `limit`: Limit of results (default: 10, max: 50000)
     - `sort`: Sort field
   - Response: List of options contracts with details

2. **Options Aggregates (OHLCV)**
   - Endpoint: `GET /v2/aggs/ticker/{options_ticker}/range/{multiplier}/{timespan}/{from}/{to}`
   - Description: Get aggregate bars for an options contract over a given date range
   - Parameters: Same as stock aggregates, but with options ticker
   - Response: OHLCV data for the options contract

3. **Last Trade for Options**
   - Endpoint: `GET /v2/last/trade/{options_ticker}`
   - Description: Get the most recent trade for an options contract
   - Parameters:
     - `options_ticker`: Options contract symbol
   - Response: Last trade with price, size, exchange, timestamp

#### Reference Data

1. **Tickers**
   - Endpoint: `GET /v3/reference/tickers`
   - Description: Get a list of ticker symbols for stocks/equities, indices, forex, or crypto
   - Parameters:
     - `ticker`: Filter by ticker
     - `type`: Type of ticker (CS = Common Stock, etc.)
     - `market`: Market type (stocks, indices, forex, crypto)
     - `exchange`: Exchange ID
     - `active`: Whether the ticker is actively traded
     - `order`: Order of results (asc, desc)
     - `limit`: Limit of results (default: 10, max: 50000)
     - `sort`: Sort field
   - Response: List of tickers with details

2. **Ticker Details**
   - Endpoint: `GET /v3/reference/tickers/{ticker}`
   - Description: Get detailed information for a ticker symbol
   - Parameters:
     - `ticker`: Stock symbol
     - `date`: Date for data retrieval (YYYY-MM-DD)
   - Response: Detailed information about the ticker

3. **Ticker News**
   - Endpoint: `GET /v2/reference/news`
   - Description: Get news articles for a ticker symbol
   - Parameters:
     - `ticker`: Stock symbol
     - `published_utc.gte`: Published after datetime (RFC3339)
     - `published_utc.lte`: Published before datetime (RFC3339)
     - `order`: Order of results (asc, desc)
     - `limit`: Limit of results (default: 10, max: 1000)
     - `sort`: Sort field
   - Response: List of news articles with title, URL, source, etc.

4. **Market Holidays**
   - Endpoint: `GET /v1/marketstatus/upcoming`
   - Description: Get upcoming market holidays and special trading hours
   - Response: List of market holidays with dates and status

5. **Market Status**
   - Endpoint: `GET /v1/marketstatus/now`
   - Description: Get current market status (open/closed)
   - Response: Current market status for different markets

### Polygon WebSocket API

Polygon.io provides real-time data through WebSocket connections.

#### WebSocket Endpoints

1. **Stocks Cluster**
   - Endpoint: `wss://socket.polygon.io/stocks`
   - Authentication: Send authentication message: `{"action":"auth","params":"YOUR_API_KEY"}`
   - Channels:
     - `T.{ticker}`: Trades for a symbol
     - `Q.{ticker}`: Quotes for a symbol
     - `A.{ticker}`: Second aggregates
     - `AM.{ticker}`: Minute aggregates
     - `status`: Exchange status updates
   - Examples:
     - Subscribe to AAPL trades: `{"action":"subscribe","params":"T.AAPL"}`
     - Subscribe to multiple channels: `{"action":"subscribe","params":"T.AAPL,Q.AAPL,A.AAPL"}`
     - Unsubscribe: `{"action":"unsubscribe","params":"T.AAPL"}`

2. **Forex Cluster**
   - Endpoint: `wss://socket.polygon.io/forex`
   - Authentication: Same as stocks
   - Channels:
     - `C.{from}.{to}`: Forex currency pair aggregates
     - `CA.{from}.{to}`: Forex currency pair minute aggregates
   - Example: Subscribe to EUR/USD: `{"action":"subscribe","params":"C.EUR.USD"}`

3. **Crypto Cluster**
   - Endpoint: `wss://socket.polygon.io/crypto`
   - Authentication: Same as stocks
   - Channels:
     - `XT.{from}.{to}`: Crypto trades
     - `XQ.{from}.{to}`: Crypto quotes
     - `XA.{from}.{to}`: Crypto second aggregates
     - `XAM.{from}.{to}`: Crypto minute aggregates
   - Example: Subscribe to BTC/USD trades: `{"action":"subscribe","params":"XT.BTC.USD"}`

4. **Options Cluster**
   - Endpoint: `wss://socket.polygon.io/options`
   - Authentication: Same as stocks
   - Channels:
     - `O.{ticker}`: Options trades
     - `OQ.{ticker}`: Options quotes
   - Example: Subscribe to AAPL options trades: `{"action":"subscribe","params":"O.AAPL220121C00140000"}`

## Unusual Whales API Endpoints

Unusual Whales provides options flow data and unusual options activity.

### Authentication
- Base URL: `https://api.unusualwhales.com/`
- Authentication: API Key in HTTP header `x-api-key: YOUR_API_KEY`

### Endpoints

1. **Options Flow**
   - Endpoint: `GET /api/v1/flow/live`
   - Description: Get real-time options flow data
   - Parameters:
     - `limit`: Number of results to return (default: 100, max: 1000)
     - `page`: Page number for pagination
     - `from_date`: Start date (YYYY-MM-DD)
     - `to_date`: End date (YYYY-MM-DD)
     - `ticker`: Filter by ticker symbol
     - `min_premium`: Minimum premium amount
     - `max_premium`: Maximum premium amount
     - `min_size`: Minimum trade size
     - `max_size`: Maximum trade size
     - `min_oi`: Minimum open interest
     - `sentiment`: Filter by sentiment (bullish, bearish)
     - `contract_type`: Filter by contract type (call, put)
   - Response: List of unusual options activity with details

2. **Options Flow Historical**
   - Endpoint: `GET /api/v1/flow/historical`
   - Description: Get historical options flow data
   - Parameters: Same as live flow, but for historical data
   - Response: Historical unusual options activity

3. **Options Alert Details**
   - Endpoint: `GET /api/v1/flow/alert/{alert_id}`
   - Description: Get detailed information about a specific alert
   - Parameters:
     - `alert_id`: ID of the alert
   - Response: Detailed information about the alert

4. **Options Chain**
   - Endpoint: `GET /api/v1/options/chain/{ticker}`
   - Description: Get the options chain for a specific ticker
   - Parameters:
     - `ticker`: Stock symbol
     - `expiration`: Filter by expiration date (YYYY-MM-DD)
     - `strike`: Filter by strike price
     - `contract_type`: Filter by contract type (call, put)
   - Response: Options chain with strikes, expirations, and pricing

5. **Unusual Score**
   - Endpoint: `GET /api/v1/flow/score/{ticker}`
   - Description: Get the unusual score for a ticker
   - Parameters:
     - `ticker`: Stock symbol
   - Response: Unusual score and related metrics

6. **Top Tickers**
   - Endpoint: `GET /api/v1/flow/top_tickers`
   - Description: Get the top tickers by unusual activity
   - Parameters:
     - `limit`: Number of results to return (default: 20, max: 100)
     - `from_date`: Start date (YYYY-MM-DD)
     - `to_date`: End date (YYYY-MM-DD)
   - Response: List of top tickers with unusual activity metrics

7. **Sector Analysis**
   - Endpoint: `GET /api/v1/flow/sectors`
   - Description: Get unusual activity by sector
   - Parameters:
     - `from_date`: Start date (YYYY-MM-DD)
     - `to_date`: End date (YYYY-MM-DD)
   - Response: Unusual activity metrics by sector

## Implementation Notes

### Polygon Client Implementation

The Polygon client should be implemented with the following features:

1. **Authentication Management**
   - Securely store and manage API keys
   - Handle authentication for both REST and WebSocket connections

2. **Rate Limiting**
   - Implement rate limiting to stay within API usage limits
   - Queue and retry requests when rate limits are reached

3. **Connection Management**
   - Maintain persistent WebSocket connections
   - Handle reconnection on disconnects
   - Implement heartbeat mechanism

4. **Data Normalization**
   - Convert API responses to standardized internal formats
   - Ensure consistent timestamp formats (UTC)
   - Normalize symbol formats

5. **Error Handling**
   - Handle API errors gracefully
   - Implement exponential backoff for retries
   - Log errors for monitoring

### Unusual Whales Client Implementation

The Unusual Whales client should be implemented with the following features:

1. **Authentication Management**
   - Securely store and manage API keys
   - Handle API key rotation if needed

2. **Data Integration**
   - Map Unusual Whales data to internal data models
   - Correlate options flow data with market data from other sources

3. **Alert Processing**
   - Process and filter alerts based on strategy criteria
   - Calculate additional metrics based on alert data

4. **Historical Analysis**
   - Implement methods for historical analysis of options flow
   - Provide backtesting capabilities using historical data

5. **Error Handling**
   - Handle API errors gracefully
   - Implement retry mechanisms
   - Log errors for monitoring

## Data Storage Considerations

When storing data from these APIs, consider:

1. **Volume Management**
   - Implement data retention policies
   - Use TimescaleDB hypertables for efficient time-series storage
   - Implement data compression for historical data

2. **Real-time Processing**
   - Use Redis for caching real-time data
   - Implement a message queue for processing WebSocket data

3. **Data Integrity**
   - Implement data validation before storage
   - Handle duplicate data points
   - Ensure proper indexing for query performance

4. **Data Enrichment**
   - Combine data from multiple sources
   - Calculate derived metrics
   - Annotate data with additional context