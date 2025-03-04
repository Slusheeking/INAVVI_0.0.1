"""
Alpaca API Client

This module provides a client for interacting with the Alpaca API for trading and market data.
"""

import os
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import REST, TimeFrame
from alpaca_trade_api.stream import Stream
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class AlpacaClient:
    """Client for interacting with the Alpaca API."""
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        api_secret: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        Initialize the Alpaca API client.
        
        Args:
            api_key: Alpaca API key (defaults to ALPACA_API_KEY environment variable)
            api_secret: Alpaca API secret (defaults to ALPACA_API_SECRET environment variable)
            base_url: Alpaca API base URL (defaults to ALPACA_API_ENDPOINT environment variable)
        """
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.api_secret = api_secret or os.getenv('ALPACA_API_SECRET')
        self.base_url = base_url or os.getenv('ALPACA_API_ENDPOINT', 'https://paper-api.alpaca.markets/v2')
        
        if not self.api_key or not self.api_secret:
            raise ValueError("Alpaca API key and secret are required")
        
        # Initialize REST API client
        self.api = REST(
            key_id=self.api_key,
            secret_key=self.api_secret,
            base_url=self.base_url
        )
        
        # Initialize streaming client
        self.stream = Stream(
            key_id=self.api_key,
            secret_key=self.api_secret,
            base_url=self.base_url,
            data_feed='iex'  # Use 'sip' for paid subscription
        )
    
    def get_account(self) -> Dict[str, Any]:
        """
        Get account information.
        
        Returns:
            Dictionary with account information
        """
        try:
            account = self.api.get_account()
            return {
                'id': account.id,
                'status': account.status,
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'long_market_value': float(account.long_market_value),
                'short_market_value': float(account.short_market_value),
                'initial_margin': float(account.initial_margin),
                'maintenance_margin': float(account.maintenance_margin),
                'last_equity': float(account.last_equity),
                'daytrade_count': account.daytrade_count,
                'last_maintenance_margin': float(account.last_maintenance_margin),
                'daytrading_buying_power': float(account.daytrading_buying_power),
                'regt_buying_power': float(account.regt_buying_power)
            }
        except Exception as e:
            logger.error(f"Error getting account information: {e}")
            raise
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions.
        
        Returns:
            List of positions
        """
        try:
            positions = self.api.list_positions()
            return [
                {
                    'symbol': position.symbol,
                    'quantity': float(position.qty),
                    'entry_price': float(position.avg_entry_price),
                    'current_price': float(position.current_price),
                    'market_value': float(position.market_value),
                    'cost_basis': float(position.cost_basis),
                    'unrealized_pl': float(position.unrealized_pl),
                    'unrealized_plpc': float(position.unrealized_plpc),
                    'side': 'long' if float(position.qty) > 0 else 'short'
                }
                for position in positions
            ]
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            raise
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get position for a specific symbol.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Position information or None if no position exists
        """
        try:
            position = self.api.get_position(symbol)
            return {
                'symbol': position.symbol,
                'quantity': float(position.qty),
                'entry_price': float(position.avg_entry_price),
                'current_price': float(position.current_price),
                'market_value': float(position.market_value),
                'cost_basis': float(position.cost_basis),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc),
                'side': 'long' if float(position.qty) > 0 else 'short'
            }
        except Exception as e:
            if 'position does not exist' in str(e).lower():
                return None
            logger.error(f"Error getting position for {symbol}: {e}")
            raise
    
    def get_orders(
        self, 
        status: Optional[str] = None, 
        limit: int = 100,
        after: Optional[Union[str, datetime]] = None,
        until: Optional[Union[str, datetime]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get orders.
        
        Args:
            status: Order status filter (open, closed, all)
            limit: Maximum number of orders to return
            after: Filter orders after this timestamp
            until: Filter orders until this timestamp
            
        Returns:
            List of orders
        """
        try:
            # Convert datetime objects to strings
            if isinstance(after, datetime):
                after = after.isoformat()
            if isinstance(until, datetime):
                until = until.isoformat()
            
            orders = self.api.list_orders(
                status=status,
                limit=limit,
                after=after,
                until=until
            )
            
            return [
                {
                    'id': order.id,
                    'client_order_id': order.client_order_id,
                    'symbol': order.symbol,
                    'side': order.side,
                    'type': order.type,
                    'time_in_force': order.time_in_force,
                    'limit_price': float(order.limit_price) if order.limit_price else None,
                    'stop_price': float(order.stop_price) if order.stop_price else None,
                    'quantity': float(order.qty),
                    'filled_quantity': float(order.filled_qty),
                    'status': order.status,
                    'created_at': order.created_at,
                    'updated_at': order.updated_at,
                    'submitted_at': order.submitted_at,
                    'filled_at': order.filled_at,
                    'expired_at': order.expired_at,
                    'canceled_at': order.canceled_at,
                    'failed_at': order.failed_at,
                    'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                    'extended_hours': order.extended_hours
                }
                for order in orders
            ]
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            raise
    
    def submit_order(
        self,
        symbol: str,
        quantity: float,
        side: str,
        order_type: str = 'market',
        time_in_force: str = 'day',
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        client_order_id: Optional[str] = None,
        extended_hours: bool = False
    ) -> Dict[str, Any]:
        """
        Submit an order.
        
        Args:
            symbol: Ticker symbol
            quantity: Order quantity
            side: Order side ('buy' or 'sell')
            order_type: Order type ('market', 'limit', 'stop', 'stop_limit')
            time_in_force: Time in force ('day', 'gtc', 'opg', 'cls', 'ioc', 'fok')
            limit_price: Limit price (required for 'limit' and 'stop_limit' orders)
            stop_price: Stop price (required for 'stop' and 'stop_limit' orders)
            client_order_id: Client order ID
            extended_hours: Whether to allow trading during extended hours
            
        Returns:
            Order information
        """
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                type=order_type,
                time_in_force=time_in_force,
                limit_price=limit_price,
                stop_price=stop_price,
                client_order_id=client_order_id,
                extended_hours=extended_hours
            )
            
            return {
                'id': order.id,
                'client_order_id': order.client_order_id,
                'symbol': order.symbol,
                'side': order.side,
                'type': order.type,
                'time_in_force': order.time_in_force,
                'limit_price': float(order.limit_price) if order.limit_price else None,
                'stop_price': float(order.stop_price) if order.stop_price else None,
                'quantity': float(order.qty),
                'filled_quantity': float(order.filled_qty),
                'status': order.status,
                'created_at': order.created_at,
                'updated_at': order.updated_at,
                'submitted_at': order.submitted_at,
                'filled_at': order.filled_at,
                'expired_at': order.expired_at,
                'canceled_at': order.canceled_at,
                'failed_at': order.failed_at,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                'extended_hours': order.extended_hours
            }
        except Exception as e:
            logger.error(f"Error submitting order for {symbol}: {e}")
            raise
    
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get order by ID.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order information
        """
        try:
            order = self.api.get_order(order_id)
            
            return {
                'id': order.id,
                'client_order_id': order.client_order_id,
                'symbol': order.symbol,
                'side': order.side,
                'type': order.type,
                'time_in_force': order.time_in_force,
                'limit_price': float(order.limit_price) if order.limit_price else None,
                'stop_price': float(order.stop_price) if order.stop_price else None,
                'quantity': float(order.qty),
                'filled_quantity': float(order.filled_qty),
                'status': order.status,
                'created_at': order.created_at,
                'updated_at': order.updated_at,
                'submitted_at': order.submitted_at,
                'filled_at': order.filled_at,
                'expired_at': order.expired_at,
                'canceled_at': order.canceled_at,
                'failed_at': order.failed_at,
                'filled_avg_price': float(order.filled_avg_price) if order.filled_avg_price else None,
                'extended_hours': order.extended_hours
            }
        except Exception as e:
            logger.error(f"Error getting order {order_id}: {e}")
            raise
    
    def cancel_order(self, order_id: str) -> None:
        """
        Cancel order by ID.
        
        Args:
            order_id: Order ID
        """
        try:
            self.api.cancel_order(order_id)
            logger.info(f"Order {order_id} canceled")
        except Exception as e:
            logger.error(f"Error canceling order {order_id}: {e}")
            raise
    
    def cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        try:
            self.api.cancel_all_orders()
            logger.info("All orders canceled")
        except Exception as e:
            logger.error(f"Error canceling all orders: {e}")
            raise
    
    def get_bars(
        self,
        symbols: Union[str, List[str]],
        timeframe: str,
        start: Union[str, datetime],
        end: Union[str, datetime],
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
            raise
    
    def get_latest_trades(self, symbols: Union[str, List[str]], limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get latest trades for one or more symbols.
        
        Args:
            symbols: Ticker symbol or list of symbols
            limit: Maximum number of trades per symbol
            
        Returns:
            Dictionary mapping symbols to lists of trades
        """
        try:
            # Convert to list if single symbol
            if isinstance(symbols, str):
                symbols = [symbols]
            
            result = {}
            for symbol in symbols:
                trades = self.api.get_latest_trades(symbol, limit)
                result[symbol] = [
                    {
                        'timestamp': trade.t,
                        'price': float(trade.p),
                        'size': float(trade.s),
                        'exchange': trade.x,
                        'trade_id': trade.i,
                        'tape': trade.z
                    }
                    for trade in trades
                ]
            
            return result
        except Exception as e:
            logger.error(f"Error getting latest trades: {e}")
            raise
    
    def get_latest_quotes(self, symbols: Union[str, List[str]], limit: int = 10) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get latest quotes for one or more symbols.
        
        Args:
            symbols: Ticker symbol or list of symbols
            limit: Maximum number of quotes per symbol
            
        Returns:
            Dictionary mapping symbols to lists of quotes
        """
        try:
            # Convert to list if single symbol
            if isinstance(symbols, str):
                symbols = [symbols]
            
            result = {}
            for symbol in symbols:
                quotes = self.api.get_latest_quotes(symbol, limit)
                result[symbol] = [
                    {
                        'timestamp': quote.t,
                        'bid_price': float(quote.p),
                        'ask_price': float(quote.P),
                        'bid_size': float(quote.s),
                        'ask_size': float(quote.S),
                        'exchange': quote.x,
                        'conditions': quote.c
                    }
                    for quote in quotes
                ]
            
            return result
        except Exception as e:
            logger.error(f"Error getting latest quotes: {e}")
            raise
    
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
            raise
    
    def get_calendar(
        self, 
        start: Optional[Union[str, datetime]] = None,
        end: Optional[Union[str, datetime]] = None
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
            raise
    
    def get_assets(self, status: str = 'active', asset_class: str = 'us_equity') -> List[Dict[str, Any]]:
        """
        Get assets.
        
        Args:
            status: Asset status ('active', 'inactive')
            asset_class: Asset class ('us_equity', 'crypto')
            
        Returns:
            List of assets
        """
        try:
            assets = self.api.list_assets(status=status, asset_class=asset_class)
            return [
                {
                    'id': asset.id,
                    'symbol': asset.symbol,
                    'name': asset.name,
                    'status': asset.status,
                    'tradable': asset.tradable,
                    'marginable': asset.marginable,
                    'shortable': asset.shortable,
                    'easy_to_borrow': asset.easy_to_borrow,
                    'fractionable': asset.fractionable
                }
                for asset in assets
            ]
        except Exception as e:
            logger.error(f"Error getting assets: {e}")
            raise
    
    def get_asset(self, symbol: str) -> Dict[str, Any]:
        """
        Get asset by symbol.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Asset information
        """
        try:
            asset = self.api.get_asset(symbol)
            return {
                'id': asset.id,
                'symbol': asset.symbol,
                'name': asset.name,
                'status': asset.status,
                'tradable': asset.tradable,
                'marginable': asset.marginable,
                'shortable': asset.shortable,
                'easy_to_borrow': asset.easy_to_borrow,
                'fractionable': asset.fractionable
            }
        except Exception as e:
            logger.error(f"Error getting asset {symbol}: {e}")
            raise