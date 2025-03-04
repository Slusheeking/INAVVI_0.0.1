"""
Unusual Whales API Client

This module provides a client for interacting with the Unusual Whales API to fetch options flow data.
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

class UnusualWhalesClient:
    """Client for interacting with the Unusual Whales API."""
    
    BASE_URL = "https://api.unusualwhales.com"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the Unusual Whales API client.
        
        Args:
            api_key: Unusual Whales API key (defaults to UNUSUAL_WHALES_API_KEY environment variable)
        """
        self.api_key = api_key or os.getenv('UNUSUAL_WHALES_API_KEY')
        if not self.api_key:
            raise ValueError("Unusual Whales API key is required")
        
        self.session = requests.Session()
        self.session.headers.update({
            'x-api-key': self.api_key,
            'User-Agent': 'AutonomousTradingSystem/0.1'
        })
        
        # Rate limiting parameters
        self.rate_limit_remaining = 100  # Conservative initial value
        self.rate_limit_reset = 0
        self.max_retries = 3
        self.retry_delay = 1.0  # seconds
    
    def _handle_rate_limit(self, response: requests.Response) -> None:
        """
        Handle rate limit headers from Unusual Whales API.
        
        Args:
            response: API response
        """
        # Update rate limit information from headers if available
        if 'X-RateLimit-Remaining' in response.headers:
            self.rate_limit_remaining = int(response.headers['X-RateLimit-Remaining'])
        
        if 'X-RateLimit-Reset' in response.headers:
            self.rate_limit_reset = int(response.headers['X-RateLimit-Reset'])
        
        # If we're close to the rate limit, sleep until reset
        if self.rate_limit_remaining < 5:
            sleep_time = max(0, self.rate_limit_reset - time.time()) + 1
            logger.warning(f"Rate limit almost reached. Sleeping for {sleep_time:.2f} seconds")
            time.sleep(sleep_time)
    
    def _make_request(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Make a request to the Unusual Whales API with rate limit handling and retries.
        
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
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, params=params)
                self._handle_rate_limit(response)
                
                # Handle HTTP errors
                response.raise_for_status()
                
                # Parse JSON response
                data = response.json()
                
                # Check for API-level errors
                if data.get('status') == 'error':
                    error_msg = data.get('message', 'Unknown API error')
                    logger.error(f"API error: {error_msg}")
                    raise ValueError(f"Unusual Whales API error: {error_msg}")
                
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
    
    def get_options_flow(
        self, 
        limit: int = 100, 
        page: int = 1,
        from_date: Optional[Union[str, datetime]] = None,
        to_date: Optional[Union[str, datetime]] = None,
        ticker: Optional[str] = None,
        min_premium: Optional[float] = None,
        max_premium: Optional[float] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        min_oi: Optional[int] = None,
        sentiment: Optional[str] = None,
        contract_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get real-time options flow data.
        
        Args:
            limit: Number of results to return (default: 100, max: 1000)
            page: Page number for pagination
            from_date: Start date (YYYY-MM-DD or datetime)
            to_date: End date (YYYY-MM-DD or datetime)
            ticker: Filter by ticker symbol
            min_premium: Minimum premium amount
            max_premium: Maximum premium amount
            min_size: Minimum trade size
            max_size: Maximum trade size
            min_oi: Minimum open interest
            sentiment: Filter by sentiment (bullish, bearish)
            contract_type: Filter by contract type (call, put)
            
        Returns:
            List of unusual options activity with details
        """
        # Convert datetime objects to strings
        if isinstance(from_date, datetime):
            from_date = from_date.strftime('%Y-%m-%d')
        if isinstance(to_date, datetime):
            to_date = to_date.strftime('%Y-%m-%d')
        
        params = {
            'limit': limit,
            'page': page
        }
        
        # Add optional parameters if provided
        if from_date:
            params['from_date'] = from_date
        if to_date:
            params['to_date'] = to_date
        if ticker:
            params['ticker'] = ticker
        if min_premium is not None:
            params['min_premium'] = min_premium
        if max_premium is not None:
            params['max_premium'] = max_premium
        if min_size is not None:
            params['min_size'] = min_size
        if max_size is not None:
            params['max_size'] = max_size
        if min_oi is not None:
            params['min_oi'] = min_oi
        if sentiment:
            params['sentiment'] = sentiment
        if contract_type:
            params['contract_type'] = contract_type
        
        result = self._make_request('/api/v1/flow/live', params)
        
        return result.get('data', [])
    
    def get_historical_flow(
        self, 
        limit: int = 100, 
        page: int = 1,
        from_date: Optional[Union[str, datetime]] = None,
        to_date: Optional[Union[str, datetime]] = None,
        ticker: Optional[str] = None,
        min_premium: Optional[float] = None,
        max_premium: Optional[float] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
        min_oi: Optional[int] = None,
        sentiment: Optional[str] = None,
        contract_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get historical options flow data.
        
        Args:
            Same parameters as get_options_flow
            
        Returns:
            List of historical unusual options activity
        """
        # Convert datetime objects to strings
        if isinstance(from_date, datetime):
            from_date = from_date.strftime('%Y-%m-%d')
        if isinstance(to_date, datetime):
            to_date = to_date.strftime('%Y-%m-%d')
        
        params = {
            'limit': limit,
            'page': page
        }
        
        # Add optional parameters if provided
        if from_date:
            params['from_date'] = from_date
        if to_date:
            params['to_date'] = to_date
        if ticker:
            params['ticker'] = ticker
        if min_premium is not None:
            params['min_premium'] = min_premium
        if max_premium is not None:
            params['max_premium'] = max_premium
        if min_size is not None:
            params['min_size'] = min_size
        if max_size is not None:
            params['max_size'] = max_size
        if min_oi is not None:
            params['min_oi'] = min_oi
        if sentiment:
            params['sentiment'] = sentiment
        if contract_type:
            params['contract_type'] = contract_type
        
        result = self._make_request('/api/v1/flow/historical', params)
        
        return result.get('data', [])
    
    def get_alert_details(self, alert_id: str) -> Dict[str, Any]:
        """
        Get detailed information about a specific alert.
        
        Args:
            alert_id: ID of the alert
            
        Returns:
            Detailed information about the alert
        """
        result = self._make_request(f'/api/v1/flow/alert/{alert_id}')
        
        return result.get('data', {})
    
    def get_options_chain(
        self, 
        ticker: str,
        expiration: Optional[Union[str, datetime]] = None,
        strike: Optional[float] = None,
        contract_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get the options chain for a specific ticker.
        
        Args:
            ticker: Stock symbol
            expiration: Filter by expiration date (YYYY-MM-DD or datetime)
            strike: Filter by strike price
            contract_type: Filter by contract type (call, put)
            
        Returns:
            Options chain with strikes, expirations, and pricing
        """
        # Convert datetime to string
        if isinstance(expiration, datetime):
            expiration = expiration.strftime('%Y-%m-%d')
        
        params = {}
        
        # Add optional parameters if provided
        if expiration:
            params['expiration'] = expiration
        if strike is not None:
            params['strike'] = strike
        if contract_type:
            params['contract_type'] = contract_type
        
        result = self._make_request(f'/api/v1/options/chain/{ticker}', params)
        
        return result.get('data', [])
    
    def get_unusual_score(self, ticker: str) -> Dict[str, Any]:
        """
        Get the unusual score for a ticker.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Unusual score and related metrics
        """
        result = self._make_request(f'/api/v1/flow/score/{ticker}')
        
        return result.get('data', {})
    
    def get_top_tickers(
        self, 
        limit: int = 20,
        from_date: Optional[Union[str, datetime]] = None,
        to_date: Optional[Union[str, datetime]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get the top tickers by unusual activity.
        
        Args:
            limit: Number of results to return (default: 20, max: 100)
            from_date: Start date (YYYY-MM-DD or datetime)
            to_date: End date (YYYY-MM-DD or datetime)
            
        Returns:
            List of top tickers with unusual activity metrics
        """
        # Convert datetime objects to strings
        if isinstance(from_date, datetime):
            from_date = from_date.strftime('%Y-%m-%d')
        if isinstance(to_date, datetime):
            to_date = to_date.strftime('%Y-%m-%d')
        
        params = {
            'limit': limit
        }
        
        # Add optional parameters if provided
        if from_date:
            params['from_date'] = from_date
        if to_date:
            params['to_date'] = to_date
        
        result = self._make_request('/api/v1/flow/top_tickers', params)
        
        return result.get('data', [])
    
    def get_sector_analysis(
        self,
        from_date: Optional[Union[str, datetime]] = None,
        to_date: Optional[Union[str, datetime]] = None
    ) -> Dict[str, Any]:
        """
        Get unusual activity by sector.
        
        Args:
            from_date: Start date (YYYY-MM-DD or datetime)
            to_date: End date (YYYY-MM-DD or datetime)
            
        Returns:
            Unusual activity metrics by sector
        """
        # Convert datetime objects to strings
        if isinstance(from_date, datetime):
            from_date = from_date.strftime('%Y-%m-%d')
        if isinstance(to_date, datetime):
            to_date = to_date.strftime('%Y-%m-%d')
        
        params = {}
        
        # Add optional parameters if provided
        if from_date:
            params['from_date'] = from_date
        if to_date:
            params['to_date'] = to_date
        
        result = self._make_request('/api/v1/flow/sectors', params)
        
        return result.get('data', {})
    
    def flow_to_dataframe(self, flow_data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert options flow data to a pandas DataFrame.
        
        Args:
            flow_data: Options flow data from get_options_flow or get_historical_flow
            
        Returns:
            DataFrame with options flow data
        """
        if not flow_data:
            return pd.DataFrame()
        
        df = pd.DataFrame(flow_data)
        
        # Convert timestamp columns to datetime
        datetime_columns = ['timestamp', 'expiration_date', 'trade_time']
        for col in datetime_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        
        # Convert numeric columns
        numeric_columns = ['strike', 'premium', 'size', 'open_interest', 'implied_volatility', 'delta', 'gamma', 'theta', 'vega']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df