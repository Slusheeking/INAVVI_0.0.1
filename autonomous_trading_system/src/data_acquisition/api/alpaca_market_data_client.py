"""
Alpaca Market Data Client

This module provides a specialized client for interacting with the free tier of Alpaca Market Data API.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
import traceback
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class AlpacaMarketDataClient:
    """Specialized client for free tier of Alpaca Market Data API."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None,
        data_feed: str = 'iex'  # 'iex' or 'sip' (requires paid subscription)
    ):
        """
        Initialize the Alpaca Market Data client.
        
        Args:
            api_key: Alpaca API key (defaults to ALPACA_API_KEY environment variable)
            api_secret: Alpaca API secret (defaults to ALPACA_API_SECRET environment variable)
            base_url: Alpaca API base URL (defaults to ALPACA_API_ENDPOINT environment variable)
            data_feed: Data feed to use ('iex' is the only option for free tier)
        """
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.api_secret = api_secret or os.getenv('ALPACA_API_SECRET')
        self.base_url = base_url or os.getenv('ALPACA_API_ENDPOINT', 'https://paper-api.alpaca.markets/v2')
        self.data_feed = data_feed
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API key and secret are required")
        
        # Initialize REST API client
        self.api = REST(
            key_id=self.api_key,
            secret_key=self.api_secret,
            base_url=self.base_url
        )
        
        logger.info(f"Initialized AlpacaMarketDataClient with {self.data_feed} data feed")
    
    def get_bars(
        self,
        symbols: Union[str, List[str]],
        timeframe: str,
        start: Union[str, datetime, date],
        end: Union[str, datetime, date],
        limit: int = 1000,
        adjustment: str = 'raw'
    ) -> pd.DataFrame:
        """
        Get historical bars for one or more symbols.
        
        Args:
            symbols: Ticker symbol or list of symbols
            timeframe: Bar timeframe ('1Min', '5Min', '15Min', '1H', '1D', etc.)
            start: Start date/time
            end: End date/time
            limit: Maximum number of bars per symbol
            adjustment: Adjustment mode ('raw', 'split', 'dividend', 'all')
            
        Note: Free tier has limitations on historical data (limited lookback period and delayed data)
            
        Returns:
            DataFrame with historical bars
        """
        try:
            # Convert string timeframe to TimeFrame enum
            tf_map = {
                '1min': TimeFrame.Minute,
                '5min': TimeFrame.Minute,
                '15min': TimeFrame.Minute,
                '1h': TimeFrame.Hour,
                '1d': TimeFrame.Day,
                '1w': TimeFrame.Week,
                '1m': TimeFrame.Month
            }
            
            # Extract multiplier from timeframe string
            if timeframe.lower() in tf_map:
                tf = tf_map[timeframe.lower()]
                multiplier = 1
            else:
                # Parse timeframe like '5Min', '15Min', etc.
                multiplier = int(''.join(filter(str.isdigit, timeframe)))
                unit = ''.join(filter(str.isalpha, timeframe)).lower()
                if unit.startswith('min'):
                    tf = TimeFrame.Minute
                elif unit.startswith('h'):
                    tf = TimeFrame.Hour
                elif unit.startswith('d'):
                    tf = TimeFrame.Day
                elif unit.startswith('w'):
                    tf = TimeFrame.Week
                elif unit.startswith('m'):
                    tf = TimeFrame.Month
                else:
                    raise ValueError(f"Invalid timeframe: {timeframe}")
            
            # Convert to list if single symbol
            if isinstance(symbols, str):
                symbols = [symbols]
            
            # Get bars
            bars = self.api.get_bars(
                symbols,
                tf,
                start,
                end,
                adjustment=adjustment,
                limit=limit,
                timeframe_multiplier=multiplier
            ).df
            
            # Reset index to make symbol and timestamp columns
            if not bars.empty:
                bars = bars.reset_index()
                
                # Add timeframe column
                bars['timeframe'] = timeframe.lower()
                
                # Rename columns to match our schema
                bars = bars.rename(columns={
                    'timestamp': 'timestamp',
                    'symbol': 'symbol',
                    'open': 'open',
                    'high': 'high',
                    'low': 'low',
                    'close': 'close',
                    'volume': 'volume',
                    'vwap': 'vwap',
                    'trade_count': 'transactions'
                })
            
            return bars
        except Exception as e:
            logger.error(f"Error getting bars: {e}")
            logger.debug(f"Stack trace: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def get_clock(self) -> Dict[str, Any]:
        """
        Get market clock information.
        
        Returns:
            Dictionary with market clock information
        """
        try:
            clock = self.api.get_clock()
            return {
                'timestamp': clock.timestamp,
                'is_open': clock.is_open,
                'next_open': clock.next_open,
                'next_close': clock.next_close
            }
        except Exception as e:
            logger.error(f"Error getting market clock: {e}")
            logger.debug(f"Stack trace: {traceback.format_exc()}")
            return {}
    
    def get_calendar(
        self, 
        start: Optional[Union[str, datetime, date]] = None,
        end: Optional[Union[str, datetime, date]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get market calendar.
        
        Args:
            start: Start date
            end: End date
            
        Returns:
            List of market calendar days
        """
        try:
            calendar = self.api.get_calendar(start=start, end=end)
            return [
                {
                    'date': day.date,
                    'open': day.open,
                    'close': day.close,
                    'session_open': day.session_open,
                    'session_close': day.session_close
                }
                for day in calendar
            ]
        except Exception as e:
            logger.error(f"Error getting market calendar: {e}")
            logger.debug(f"Stack trace: {traceback.format_exc()}")
            return []