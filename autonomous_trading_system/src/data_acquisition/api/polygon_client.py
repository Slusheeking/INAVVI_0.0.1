"""
Polygon.io API Client

This module provides a client for interacting with the Polygon.io API to fetch market data.
"""

import os
import time
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class PolygonClient:
    """Client for interacting with the Polygon.io API."""
    
    BASE_URL = "https://api.polygon.io"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Polygon.io API client.
        
        Args:
            api_key: Polygon.io API key (defaults to POLYGON_API_KEY environment variable)
        """
        self.api_key = api_key or os.getenv('POLYGON_API_KEY')
        if not self.api_key:
            raise ValueError("Polygon API key is required")
        
        self.session = requests.Session()
        self.session.headers.update({
            'Authorization': f'Bearer {self.api_key}',
            'User-Agent': 'AutonomousTradingSystem/0.1'
        })
        
        # Rate limiting parameters
        self.rate_limit_remaining = 5  # Conservative initial value
        self.rate_limit_reset = 0
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
    
    def _handle_rate_limit(self, response: requests.Response) -> None:
        """
        Handle rate limit headers from Polygon.io API.
        
        Args:
            response: API response
        """
        # Update rate limit information from headers
        if 'X-Ratelimit-Remaining' in response.headers:
            self.rate_limit_remaining = int(response.headers['X-Ratelimit-Remaining'])
        
        if 'X-Ratelimit-Reset' in response.headers:
            self.rate_limit_reset = int(response.headers['X-Ratelimit-Reset'])
        
        # If we're close to the rate limit, sleep until reset
        if self.rate_limit_remaining < 2:
            sleep_time = max(0, self.rate_limit_reset - time.time()) + 1
            logger.warning(f"Rate limit almost reached. Sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a request to the Polygon.io API with rate limit handling and retries.
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            
        Returns:
            API response as a dictionary
            
        Raises:
            requests.RequestException: If the request fails after retries
        """
        url = f"{self.BASE_URL}{endpoint}"
        params = params or {}
        
        # Add API key if not using Authorization header
        if 'apiKey' not in params:
            params['apiKey'] = self.api_key
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params)
                self._handle_rate_limit(response)
                
                # Handle HTTP errors
                response.raise_for_status()
                
                # Parse JSON response
                data = response.json()
                
                # Check for API-level errors
                if data.get('status') == 'ERROR':
                    error_msg = data.get('error', 'Unknown API error')
                    logger.error(f"API error: {error_msg}")
                    raise ValueError(f"Polygon API error: {error_msg}")
                
                return data
                
            except requests.RequestException as e:
                # Handle rate limiting (429)
                if hasattr(e, 'response') and e.response.status_code == 429:
                    retry_after = int(e.response.headers.get('Retry-After', self.retry_delay))
                    logger.warning(f"Rate limited. Retrying after {retry_after} seconds")
                    time.sleep(retry_after)
                    continue
                
                # Handle other errors
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"Request failed: {e}. Retrying in {wait_time:.2f} seconds")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Request failed after {self.max_retries} attempts: {e}")
                    raise
        
        # This should not be reached due to the raise in the loop
        raise RuntimeError("Unexpected error in request handling")
    
    def get_aggregates(
        self, 
        ticker: str, 
        multiplier: int, 
        timespan: str, 
        from_date: Union[str, datetime], 
        to_date: Union[str, datetime],
        adjusted: bool = True,
        limit: int = 50000
    ) -> pd.DataFrame:
        """
        Get aggregate bars for a ticker over a given date range.
        
        Args:
            ticker: Ticker symbol
            multiplier: Size of the timespan multiplier
            timespan: Size of the time window (minute, hour, day, week, month, quarter, year)
            from_date: Start date (YYYY-MM-DD or datetime)
            to_date: End date (YYYY-MM-DD or datetime)
            adjusted: Whether to adjust for splits
            limit: Maximum number of results
            
        Returns:
            DataFrame with aggregate bars
        """
        # Convert datetime objects to strings
        if isinstance(from_date, datetime):
            from_date = from_date.strftime('%Y-%m-%d')
        if isinstance(to_date, datetime):
            to_date = to_date.strftime('%Y-%m-%d')
        
        endpoint = f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        params = {
            'adjusted': str(adjusted).lower(),
            'sort': 'asc',
            'limit': limit
        }
        
        result = self._make_request(endpoint, params)
        
        # Check if we have results
        if not result.get('results'):
            logger.warning(f"No data found for {ticker} from {from_date} to {to_date}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(result['results'])
        
        # Rename columns to match our schema
        column_map = {
            't': 'timestamp',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'vw': 'vwap',
            'n': 'transactions'
        }
        df = df.rename(columns=column_map)
        
        # Convert timestamp from milliseconds to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # Add additional columns
        df['symbol'] = ticker
        df['timeframe'] = f"{multiplier}{timespan[0]}"  # e.g., "1m", "5m", "1d"
        df['multiplier'] = multiplier
        df['timespan_unit'] = timespan
        df['adjusted'] = adjusted
        
        return df
    
    def get_ticker_details(self, ticker: str) -> Dict[str, Any]:
        """
        Get detailed information about a ticker.
        
        Args:
            ticker: Ticker symbol
            
        Returns:
            Dictionary with ticker details
        """
        endpoint = f"/v3/reference/tickers/{ticker}"
        result = self._make_request(endpoint)
        
        return result.get('results', {})
    
    def get_quotes(
        self, 
        ticker: str, 
        date: Union[str, datetime],
        limit: int = 50000
    ) -> pd.DataFrame:
        """
        Get NBBO quotes for a ticker on a specific date.
        
        Args:
            ticker: Ticker symbol
            date: Date (YYYY-MM-DD or datetime)
            limit: Maximum number of results
            
        Returns:
            DataFrame with quotes
        """
        # Convert datetime to string
        if isinstance(date, datetime):
            date = date.strftime('%Y-%m-%d')
        
        endpoint = f"/v3/quotes/{ticker}/{date}"
        params = {
            'limit': limit
        }
        
        result = self._make_request(endpoint, params)
        
        # Check if we have results
        if not result.get('results'):
            logger.warning(f"No quotes found for {ticker} on {date}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(result['results'])
        
        # Rename columns to match our schema
        column_map = {
            't': 'timestamp',
            'p': 'bid_price',
            'P': 'ask_price',
            's': 'bid_size',
            'S': 'ask_size',
            'x': 'exchange',
            'c': 'conditions',
            'q': 'sequence_number',
            'z': 'tape'
        }
        df = df.rename(columns=column_map)
        
        # Convert timestamp from nanoseconds to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
        
        # Add additional columns
        df['symbol'] = ticker
        df['source'] = 'polygon'
        
        return df
    
    def get_trades(
        self, 
        ticker: str, 
        date: Union[str, datetime],
        limit: int = 50000
    ) -> pd.DataFrame:
        """
        Get trades for a ticker on a specific date.
        
        Args:
            ticker: Ticker symbol
            date: Date (YYYY-MM-DD or datetime)
            limit: Maximum number of results
            
        Returns:
            DataFrame with trades
        """
        # Convert datetime to string
        if isinstance(date, datetime):
            date = date.strftime('%Y-%m-%d')
        
        endpoint = f"/v3/trades/{ticker}/{date}"
        params = {
            'limit': limit
        }
        
        result = self._make_request(endpoint, params)
        
        # Check if we have results
        if not result.get('results'):
            logger.warning(f"No trades found for {ticker} on {date}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(result['results'])
        
        # Rename columns to match our schema
        column_map = {
            't': 'timestamp',
            'p': 'price',
            's': 'size',
            'x': 'exchange',
            'c': 'conditions',
            'i': 'trade_id',
            'q': 'sequence_number',
            'z': 'tape'
        }
        df = df.rename(columns=column_map)
        
        # Convert timestamp from nanoseconds to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns')
        
        # Add additional columns
        df['symbol'] = ticker
        df['source'] = 'polygon'
        
        return df
    
    def get_options_chain(
        self, 
        underlying: str,
        expiration_date: Optional[Union[str, datetime]] = None,
        strike_price: Optional[float] = None,
        contract_type: Optional[str] = None  # 'call' or 'put'
    ) -> List[Dict[str, Any]]:
        """
        Get options chain for an underlying asset.
        
        Args:
            underlying: Underlying asset symbol
            expiration_date: Optional expiration date filter (YYYY-MM-DD or datetime)
            strike_price: Optional strike price filter
            contract_type: Optional contract type filter ('call' or 'put')
            
        Returns:
            List of option contracts
        """
        # Convert datetime to string
        if isinstance(expiration_date, datetime):
            expiration_date = expiration_date.strftime('%Y-%m-%d')
        
        endpoint = f"/v3/reference/options/contracts"
        params = {
            'underlying_ticker': underlying
        }
        
        # Add optional filters
        if expiration_date:
            params['expiration_date'] = expiration_date
        if strike_price:
            params['strike_price'] = strike_price
        if contract_type:
            params['contract_type'] = contract_type.upper()
        
        result = self._make_request(endpoint, params)
        
        return result.get('results', [])
    
    def get_market_status(self) -> Dict[str, Any]:
        """
        Get current market status.
        
        Returns:
            Dictionary with market status information
        """
        endpoint = "/v1/marketstatus/now"
        result = self._make_request(endpoint)
        
        return result
    
    def get_market_holidays(self) -> List[Dict[str, Any]]:
        """
        Get market holidays.
        
        Returns:
            List of market holidays
        """
        endpoint = "/v1/marketstatus/upcoming"
        result = self._make_request(endpoint)
        
        return result
    
    def get_news(
        self, 
        ticker: Optional[str] = None,
        limit: int = 100,
        order: str = 'desc',
        sort: str = 'published_utc'
    ) -> List[Dict[str, Any]]:
        """
        Get news articles.
        
        Args:
            ticker: Optional ticker symbol to filter by
            limit: Maximum number of results
            order: Sort order ('asc' or 'desc')
            sort: Field to sort by
            
        Returns:
            List of news articles
        """
        endpoint = "/v2/reference/news"
        params = {
            'limit': limit,
            'order': order,
            'sort': sort
        }
        
        if ticker:
            params['ticker'] = ticker
        
        result = self._make_request(endpoint, params)
        
        return result.get('results', [])