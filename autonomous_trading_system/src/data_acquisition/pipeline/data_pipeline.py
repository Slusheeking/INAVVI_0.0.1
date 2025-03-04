"""
Data Acquisition Pipeline

This module provides the main pipeline for acquiring market data from various sources
and storing it in the database.
"""

import os
import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from datetime import datetime, timedelta, date
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue
import traceback

from autonomous_trading_system.src.data_acquisition.api.polygon_client import PolygonClient
from autonomous_trading_system.src.data_acquisition.api.alpaca_client import AlpacaClient
from autonomous_trading_system.src.data_acquisition.api.unusual_whales_client import UnusualWhalesClient
from autonomous_trading_system.src.data_acquisition.storage.timescale_storage import TimescaleStorage
from autonomous_trading_system.src.data_acquisition.validation.data_validator import DataValidator

logger = logging.getLogger(__name__)

class DataPipeline:
    """Main pipeline for data acquisition."""
    
    def __init__(
        self,
        polygon_client: Optional[PolygonClient] = None,
        alpaca_client: Optional[AlpacaClient] = None,
        unusual_whales_client: Optional[UnusualWhalesClient] = None,
        storage: Optional[TimescaleStorage] = None,
        validator: Optional[DataValidator] = None,
        max_workers: int = 10
    ):
        """
        Initialize the data pipeline.
        
        Args:
            polygon_client: Polygon.io API client
            alpaca_client: Alpaca API client
            unusual_whales_client: Unusual Whales API client
            storage: TimescaleDB storage
            validator: Data validator
            max_workers: Maximum number of worker threads
        """
        self.polygon = polygon_client or PolygonClient()
        self.unusual_whales = unusual_whales_client or UnusualWhalesClient()
        self.alpaca = alpaca_client or AlpacaClient()
        self.storage = storage or TimescaleStorage()
        self.validator = validator or DataValidator()
        self.max_workers = max_workers
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Queue for storing results
        self.result_queue = queue.Queue()
        
        # Flags for controlling the pipeline
        self.running = False
        self.stop_requested = False

    def normalize_symbol(self, symbol: str) -> str:
        """
        Normalize ticker symbols to a standard format.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Normalized symbol
        """
        if not symbol:
            return ""
            
        # Convert to uppercase and strip whitespace
        normalized = symbol.upper().strip()
        
        # Handle special cases
        if '/' in normalized:  # Crypto pairs like BTC/USD
            normalized = normalized.replace('/', '')
            if not normalized.startswith('X:'):
                normalized = f"X:{normalized}"
        
        # Handle options symbols (convert to OCC format if needed)
        if ' ' in normalized and any(x in normalized for x in ['C', 'P', 'CALL', 'PUT']):
            parts = normalized.split()
            if len(parts) >= 3:
                underlying = parts[0]
                expiry_str = parts[1]
                strike_type = parts[2]
                
                # This is a simplified conversion - a real implementation would need more logic
                # to handle the full OCC format conversion
                normalized = f"{underlying}_{expiry_str}_{strike_type}"
        
        return normalized

    
    def collect_stock_aggs(
        self,
        symbols: List[str],
        timeframe: str,
        start_date: Union[str, datetime, date],
        end_date: Union[str, datetime, date],
        adjusted: bool = True,
        store: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect stock aggregates for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            timeframe: Bar timeframe (e.g., '1m', '5m', '1h', '1d')
            start_date: Start date
            end_date: End date
            adjusted: Whether to adjust for splits
            store: Whether to store the data
            
        Returns:
            Dictionary mapping symbols to DataFrames with aggregates
        """
        logger.info(f"Collecting stock aggregates for {len(symbols)} symbols")
        
        # Parse timeframe
        if timeframe.endswith('m'):
            multiplier = int(timeframe[:-1])
            timespan = 'minute'
        elif timeframe.endswith('h'):
            multiplier = int(timeframe[:-1])
            timespan = 'hour'
        elif timeframe.endswith('d'):
            multiplier = int(timeframe[:-1])
            timespan = 'day'
        else:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        
        # Collect data for each symbol in parallel
        futures = {}
        results = {}
        
        # Normalize symbols
        symbols = [self.normalize_symbol(symbol) for symbol in symbols]
        
        for symbol in symbols:
            future = self.executor.submit(
                self._collect_stock_aggs_for_symbol,
                symbol, multiplier, timespan, start_date, end_date, adjusted
            )
            futures[future] = symbol
        
        # Process results as they complete
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    results[symbol] = df
                    if store:
                        self.storage.store_stock_aggs(df)
            except Exception as e:
                logger.error(f"Error collecting stock aggregates for {symbol}: {e}")
                logger.debug(traceback.format_exc())
        
        logger.info(f"Collected stock aggregates for {len(results)} symbols")
        return results
    
    def _collect_stock_aggs_for_symbol(
        self,
        symbol: str,
        multiplier: int,
        timespan: str,
        start_date: Union[str, datetime, date],
        end_date: Union[str, datetime, date],
        adjusted: bool
    ) -> pd.DataFrame:
        """
        Collect stock aggregates for a single symbol.
        
        Args:
            symbol: Ticker symbol
            multiplier: Timespan multiplier
            timespan: Timespan unit
            start_date: Start date
            end_date: End date
            adjusted: Whether to adjust for splits
            
        Returns:
            DataFrame with aggregates
        """
        try:
            logger.debug(f"Collecting {timespan} aggregates for {symbol}")
            df = self.polygon.get_aggregates(
                symbol, multiplier, timespan, start_date, end_date, adjusted
            )
            
            if df.empty:
                logger.warning(f"No {timespan} aggregates found for {symbol}")
                return df
            
            # Validate data
            df = self.validator.validate_stock_aggs(df)
            
            logger.debug(f"Collected {len(df)} {timespan} aggregates for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting {timespan} aggregates for {symbol}: {e}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame()
    
    def collect_quotes(
        self,
        symbols: List[str],
        date_to_collect: Union[str, datetime, date],
        store: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect quotes for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            date_to_collect: Date to collect quotes for
            store: Whether to store the data
            
        Returns:
            Dictionary mapping symbols to DataFrames with quotes
        """
        logger.info(f"Collecting quotes for {len(symbols)} symbols on {date_to_collect}")
        
        # Ensure date is in the correct format
        if isinstance(date_to_collect, datetime):
            date_str = date_to_collect.strftime('%Y-%m-%d')
        elif isinstance(date_to_collect, date):
            date_str = date_to_collect.strftime('%Y-%m-%d')
        else:
            date_str = date_to_collect
        
        # Collect quotes for each symbol in parallel
        futures = {}
        results = {}
        
        # Normalize symbols
        symbols = [self.normalize_symbol(symbol) for symbol in symbols]
        
        for symbol in symbols:
            future = self.executor.submit(
                self._collect_quotes_for_symbol,
                symbol, date_str
            )
            futures[future] = symbol
        
        # Process results as they complete
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    results[symbol] = df
                    if store:
                        self.storage.store_quotes(df)
            except Exception as e:
                logger.error(f"Error collecting quotes for {symbol}: {e}")
                logger.debug(traceback.format_exc())
        
        logger.info(f"Collected quotes for {len(results)} symbols")
        return results
    
    def _collect_quotes_for_symbol(
        self,
        symbol: str,
        date_str: str
    ) -> pd.DataFrame:
        """
        Collect quotes for a single symbol.
        
        Args:
            symbol: Ticker symbol
            date_str: Date string (YYYY-MM-DD)
            
        Returns:
            DataFrame with quotes
        """
        try:
            logger.debug(f"Collecting quotes for {symbol} on {date_str}")
            df = self.polygon.get_quotes(symbol, date_str)
            
            if df.empty:
                logger.warning(f"No quotes found for {symbol} on {date_str}")
                return df
            
            # Validate data
            df = self.validator.validate_quotes(df)
            
            logger.debug(f"Collected {len(df)} quotes for {symbol} on {date_str}")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting quotes for {symbol} on {date_str}: {e}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame()
    
    def collect_trades(
        self,
        symbols: List[str],
        date_to_collect: Union[str, datetime, date],
        store: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Collect trades for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            date_to_collect: Date to collect trades for
            store: Whether to store the data
            
        Returns:
            Dictionary mapping symbols to DataFrames with trades
        """
        logger.info(f"Collecting trades for {len(symbols)} symbols on {date_to_collect}")
        
        # Ensure date is in the correct format
        if isinstance(date_to_collect, datetime):
            date_str = date_to_collect.strftime('%Y-%m-%d')
        elif isinstance(date_to_collect, date):
            date_str = date_to_collect.strftime('%Y-%m-%d')
        else:
            date_str = date_to_collect
        
        # Collect trades for each symbol in parallel
        futures = {}
        results = {}
        
        # Normalize symbols
        symbols = [self.normalize_symbol(symbol) for symbol in symbols]
        
        for symbol in symbols:
            future = self.executor.submit(
                self._collect_trades_for_symbol,
                symbol, date_str
            )
            futures[future] = symbol
        
        # Process results as they complete
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                df = future.result()
                if not df.empty:
                    results[symbol] = df
                    if store:
                        self.storage.store_trades(df)
            except Exception as e:
                logger.error(f"Error collecting trades for {symbol}: {e}")
                logger.debug(traceback.format_exc())
        
        logger.info(f"Collected trades for {len(results)} symbols")
        return results
    
    def _collect_trades_for_symbol(
        self,
        symbol: str,
        date_str: str
    ) -> pd.DataFrame:
        """
        Collect trades for a single symbol.
        
        Args:
            symbol: Ticker symbol
            date_str: Date string (YYYY-MM-DD)
            
        Returns:
            DataFrame with trades
        """
        try:
            logger.debug(f"Collecting trades for {symbol} on {date_str}")
            df = self.polygon.get_trades(symbol, date_str)
            
            if df.empty:
                logger.warning(f"No trades found for {symbol} on {date_str}")
                return df
            
            # Validate data
            df = self.validator.validate_trades(df)
            
            logger.debug(f"Collected {len(df)} trades for {symbol} on {date_str}")
            return df
            
        except Exception as e:
            logger.error(f"Error collecting trades for {symbol} on {date_str}: {e}")
            logger.debug(traceback.format_exc())
            return pd.DataFrame()
    
    def collect_ticker_details(
        self,
        symbols: List[str],
        store: bool = True
    ) -> Dict[str, Dict[str, Any]]:
        """
        Collect ticker details for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            store: Whether to store the data
            
        Returns:
            Dictionary mapping symbols to ticker details
        """
        logger.info(f"Collecting ticker details for {len(symbols)} symbols")
        
        # Collect ticker details for each symbol in parallel
        futures = {}
        results = {}
        
        # Normalize symbols
        symbols = [self.normalize_symbol(symbol) for symbol in symbols]
        
        for symbol in symbols:
            future = self.executor.submit(
                self._collect_ticker_details_for_symbol,
                symbol
            )
            futures[future] = symbol
        
        # Process results as they complete
        ticker_details_list = []
        
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                details = future.result()
                if details:
                    results[symbol] = details
                    
                    # Convert to DataFrame row for storage
                    df_row = pd.DataFrame([{
                        'ticker': symbol,
                        'name': details.get('name', ''),
                        'market': details.get('market', ''),
                        'locale': details.get('locale', ''),
                        'type': details.get('type', ''),
                        'currency': details.get('currency', 'USD'),
                        'active': details.get('active', True),
                        'primary_exchange': details.get('primary_exchange', ''),
                        'last_updated': datetime.now(),
                        'description': details.get('description', ''),
                        'sic_code': details.get('sic_code', ''),
                        'sic_description': details.get('sic_description', ''),
                        'ticker_root': details.get('ticker_root', ''),
                        'homepage_url': details.get('homepage_url', ''),
                        'total_employees': details.get('total_employees'),
                        'list_date': details.get('list_date'),
                        'share_class_shares_outstanding': details.get('share_class_shares_outstanding'),
                        'weighted_shares_outstanding': details.get('weighted_shares_outstanding'),
                        'market_cap': details.get('market_cap'),
                        'phone_number': details.get('phone_number', ''),
                        'address': details.get('address', {}),
                        'metadata': details
                    }])
                    
                    ticker_details_list.append(df_row)
            except Exception as e:
                logger.error(f"Error collecting ticker details for {symbol}: {e}")
                logger.debug(traceback.format_exc())
        
        # Store ticker details
        if store and ticker_details_list:
            ticker_details_df = pd.concat(ticker_details_list, ignore_index=True)
            self.storage.store_ticker_details(ticker_details_df)
        
        logger.info(f"Collected ticker details for {len(results)} symbols")
        return results
    
    def _collect_ticker_details_for_symbol(self, symbol: str) -> Dict[str, Any]:
        """
        Collect ticker details for a single symbol.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Dictionary with ticker details
        """
        try:
            logger.debug(f"Collecting ticker details for {symbol}")
            details = self.polygon.get_ticker_details(symbol)
            
            if not details:
                logger.warning(f"No ticker details found for {symbol}")
                return {}
            
            logger.debug(f"Collected ticker details for {symbol}")
            return details
            
        except Exception as e:
            logger.error(f"Error collecting ticker details for {symbol}: {e}")
            logger.debug(traceback.format_exc())
            return {}
    
    def collect_market_status(self, store: bool = True) -> Dict[str, Any]:
        """
        Collect current market status.
        
        Args:
            store: Whether to store the data
            
        Returns:
            Dictionary with market status
        """
        try:
            logger.info("Collecting market status")
            status = self.polygon.get_market_status()
            
            if not status:
                logger.warning("No market status found")
                return {}
            
            # Convert to DataFrame for storage
            if store:
                now = datetime.now()
                markets = ['us_equity', 'us_options', 'forex', 'crypto']
                rows = []
                
                for market in markets:
                    market_status = status.get(market, {})
                    rows.append({
                        'timestamp': now,
                        'market': market,
                        'status': market_status.get('status', 'unknown'),
                        'next_open': market_status.get('next_open'),
                        'next_close': market_status.get('next_close'),
                        'early_close': market_status.get('early_close', False),
                        'late_open': market_status.get('late_open', False)
                    })
                
                df = pd.DataFrame(rows)
                self.storage.store_market_status(df)
            
            logger.info("Collected market status")
            return status
            
        except Exception as e:
            logger.error(f"Error collecting market status: {e}")
            logger.debug(traceback.format_exc())
            return {}
    
    def collect_market_holidays(self, store: bool = True) -> List[Dict[str, Any]]:
        """
        Collect market holidays.
        
        Args:
            store: Whether to store the data
            
        Returns:
            List of market holidays
        """
        try:
            logger.info("Collecting market holidays")
            holidays = self.polygon.get_market_holidays()
            
            if not holidays:
                logger.warning("No market holidays found")
                return []
            
            # Convert to DataFrame for storage
            if store and holidays:
                rows = []
                
                for holiday in holidays:
                    date_str = holiday.get('date')
                    if date_str:
                        holiday_date = datetime.strptime(date_str, '%Y-%m-%d').date()
                        year = holiday_date.year
                        
                        # Add entry for each market
                        for market in ['us_equity', 'us_options', 'forex', 'crypto']:
                            status = holiday.get(market, {}).get('status', 'unknown')
                            if status != 'unknown':
                                rows.append({
                                    'date': holiday_date,
                                    'name': holiday.get('name', ''),
                                    'market': market,
                                    'status': status,
                                    'open_time': holiday.get(market, {}).get('open'),
                                    'close_time': holiday.get(market, {}).get('close'),
                                    'year': year
                                })
                
                if rows:
                    df = pd.DataFrame(rows)
                    self.storage.store_market_holidays(df)
            
            logger.info(f"Collected {len(holidays)} market holidays")
            return holidays
            
        except Exception as e:
            logger.error(f"Error collecting market holidays: {e}")
            logger.debug(traceback.format_exc())
            return []
    
    def collect_news(
        self,
        symbols: Optional[List[str]] = None,
        limit: int = 100,
        store: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Collect news articles.
        
        Args:
            symbols: Optional list of ticker symbols to filter by
            limit: Maximum number of articles to collect
            store: Whether to store the data
            
        Returns:
            List of news articles
        """
        try:
            if symbols:
                logger.info(f"Collecting news for {len(symbols)} symbols")
                
                # Collect news for each symbol in parallel
                futures = {}
                all_articles = []
                
                # Normalize symbols
                symbols = [self.normalize_symbol(symbol) for symbol in symbols]
                
                for symbol in symbols:
                    future = self.executor.submit(
                        self.polygon.get_news,
                        symbol, limit
                    )
                    futures[future] = symbol
                
                # Process results as they complete
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        articles = future.result()
                        all_articles.extend(articles)
                    except Exception as e:
                        logger.error(f"Error collecting news for {symbol}: {e}")
                        logger.debug(traceback.format_exc())
                
                # Remove duplicates
                article_ids = set()
                unique_articles = []
                
                for article in all_articles:
                    article_id = article.get('id')
                    if article_id and article_id not in article_ids:
                        article_ids.add(article_id)
                        unique_articles.append(article)
                
                articles = unique_articles
                
            else:
                logger.info("Collecting general news")
                articles = self.polygon.get_news(limit=limit)
            
            # Convert to DataFrame for storage
            if store and articles:
                rows = []
                
                for article in articles:
                    rows.append({
                        'article_id': article.get('id', ''),
                        'published_utc': article.get('published_utc'),
                        'title': article.get('title', ''),
                        'author': article.get('author', ''),
                        'article_url': article.get('article_url', ''),
                        'tickers': article.get('tickers', []),
                        'image_url': article.get('image_url', ''),
                        'description': article.get('description', ''),
                        'keywords': article.get('keywords', []),
                        'source': article.get('publisher', {}).get('name', 'unknown')
                    })
                
                if rows:
                    df = pd.DataFrame(rows)
                    self.storage.store_news_articles(df)
            
            logger.info(f"Collected {len(articles)} news articles")
            return articles
            
        except Exception as e:
            logger.error(f"Error collecting news: {e}")
            logger.debug(traceback.format_exc())
            return []

    def collect_options_flow(
        self,
        symbols: Optional[List[str]] = None,
        days_back: int = 1,
        limit: int = 1000,
        store: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Collect options flow data from Unusual Whales.
        
        Args:
            symbols: Optional list of ticker symbols to filter by
            days_back: Number of days to look back
            limit: Maximum number of results to return
            store: Whether to store the data
            
        Returns:
            List of options flow data
        """
        try:
            logger.info(f"Collecting options flow data for the past {days_back} days")
            
            # Calculate date range
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=days_back)
            
            # Normalize symbols if provided
            normalized_symbols = None
            if symbols:
                normalized_symbols = [self.normalize_symbol(symbol) for symbol in symbols]
                logger.info(f"Filtering options flow for {len(normalized_symbols)} symbols")
            
            # Collect options flow data
            flow_data = []
            
            # Get historical flow data
            for symbol in (normalized_symbols or [None]):
                try:
                    historical_flow = self.unusual_whales.get_historical_flow(
                        limit=limit,
                        from_date=start_date,
                        to_date=end_date,
                        ticker=symbol
                    )
                    
                    if historical_flow:
                        flow_data.extend(historical_flow)
                        
                except Exception as e:
                    logger.error(f"Error collecting historical options flow for {symbol}: {e}")
                    logger.debug(traceback.format_exc())
            
            # Get live flow data
            if normalized_symbols:
                for symbol in normalized_symbols:
                    try:
                        live_flow = self.unusual_whales.get_options_flow(
                            limit=limit,
                            ticker=symbol
                        )
                        
                        if live_flow:
                            flow_data.extend(live_flow)
                            
                    except Exception as e:
                        logger.error(f"Error collecting live options flow for {symbol}: {e}")
                        logger.debug(traceback.format_exc())
            else:
                try:
                    live_flow = self.unusual_whales.get_options_flow(limit=limit)
                    if live_flow:
                        flow_data.extend(live_flow)
                except Exception as e:
                    logger.error(f"Error collecting live options flow: {e}")
                    logger.debug(traceback.format_exc())
            
            # Remove duplicates
            seen_ids = set()
            unique_flow_data = []
            
            for item in flow_data:
                item_id = item.get('id')
                if item_id and item_id not in seen_ids:
                    seen_ids.add(item_id)
                    unique_flow_data.append(item)
            
            flow_data = unique_flow_data
            
            # Store data if requested
            if store and flow_data:
                # Convert to DataFrame
                df = self.unusual_whales.flow_to_dataframe(flow_data)
                
                if not df.empty:
                    # Store in options_flow table
                    # Note: This assumes a table structure that matches the DataFrame
                    self.storage.execute_statement("""
                        CREATE TABLE IF NOT EXISTS options_flow (
                            id TEXT PRIMARY KEY,
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
                            source TEXT
                        )
                    """)
                    
                    # Use to_sql with if_exists='append' to store the data
                    df.to_sql('options_flow', self.storage.engine, if_exists='append', index=False, 
                              method='multi', chunksize=10000)
                    
                    logger.info(f"Stored {len(df)} options flow records")
            
            logger.info(f"Collected {len(flow_data)} options flow records")
            return flow_data
            
        except Exception as e:
            logger.error(f"Error collecting options flow: {e}")
            logger.debug(traceback.format_exc())
            return []
    
    def run_daily_collection(
        self,
        symbols: List[str],
        timeframes: List[str] = ['1m', '5m', '15m', '1h', '1d'],
        include_quotes: bool = True,
        include_trades: bool = False,
        include_options_flow: bool = True,
        days_back: int = 1
    ) -> Dict[str, Any]:
        """
        Run a daily collection of market data.
        
        Args:
            symbols: List of ticker symbols
            timeframes: List of timeframes to collect
            include_quotes: Whether to collect quotes
            include_trades: Whether to collect trades
            include_options_flow: Whether to collect options flow data
            days_back: Number of days to look back
            
        Returns:
            Dictionary with collection statistics
        """
        logger.info(f"Starting daily collection for {len(symbols)} symbols")
        
        # Calculate date range
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days_back)
        
        stats = {
            'symbols_processed': len(symbols),
            'timeframes_processed': len(timeframes),
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'aggregates_collected': {},
            'quotes_collected': 0,
            'trades_collected': 0,
            'ticker_details_collected': 0,
            'news_articles_collected': 0,
            'errors': []
        }

        # Normalize symbols
        normalized_symbols = [self.normalize_symbol(symbol) for symbol in symbols]
        
        try:
            # Collect ticker details
            ticker_details = self.collect_ticker_details(normalized_symbols)
            stats['ticker_details_collected'] = len(ticker_details)
            
            # Collect aggregates for each timeframe
            for timeframe in timeframes:
                try:
                    results = self.collect_stock_aggs(
                        normalized_symbols, timeframe, start_date, end_date
                    )
                    
                    total_bars = sum(len(df) for df in results.values())
                    stats['aggregates_collected'][timeframe] = total_bars
                    
                    logger.info(f"Collected {total_bars} {timeframe} bars for {len(results)} symbols")
                    
                except Exception as e:
                    error_msg = f"Error collecting {timeframe} aggregates: {e}"
                    logger.error(error_msg)
                    logger.debug(traceback.format_exc())
                    stats['errors'].append(error_msg)
            
            # Collect quotes
            if include_quotes:
                try:
                    for day_offset in range(days_back):
                        collection_date = end_date - timedelta(days=day_offset)
                        quotes_results = self.collect_quotes(normalized_symbols, collection_date)
                        
                        total_quotes = sum(len(df) for df in quotes_results.values())
                        stats['quotes_collected'] += total_quotes
                        
                        logger.info(f"Collected {total_quotes} quotes for {len(quotes_results)} symbols on {collection_date}")
                        
                except Exception as e:
                    error_msg = f"Error collecting quotes: {e}"
                    logger.error(error_msg)
                    logger.debug(traceback.format_exc())
                    stats['errors'].append(error_msg)
            
            # Collect trades
            if include_trades:
                try:
                    for day_offset in range(days_back):
                        collection_date = end_date - timedelta(days=day_offset)
                        trades_results = self.collect_trades(normalized_symbols, collection_date)
                        
                        total_trades = sum(len(df) for df in trades_results.values())
                        stats['trades_collected'] += total_trades
                        
                        logger.info(f"Collected {total_trades} trades for {len(trades_results)} symbols on {collection_date}")
                        
                except Exception as e:
                    error_msg = f"Error collecting trades: {e}"
                    logger.error(error_msg)
                    logger.debug(traceback.format_exc())
                    stats['errors'].append(error_msg)
            
            # Collect options flow data
            if include_options_flow:
                try:
                    options_flow = self.collect_options_flow(normalized_symbols, days_back)
                    stats['options_flow_collected'] = len(options_flow)
                    
                except Exception as e:
                    error_msg = f"Error collecting options flow: {e}"
                    logger.error(error_msg)
                    logger.debug(traceback.format_exc())
                    stats['errors'].append(error_msg)
            
            # Collect news
            try:
                news_articles = self.collect_news(normalized_symbols)
                stats['news_articles_collected'] = len(news_articles)
                
            except Exception as e:
                error_msg = f"Error collecting news: {e}"
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                stats['errors'].append(error_msg)
            
            # Collect market status and holidays
            try:
                self.collect_market_status()
                self.collect_market_holidays()
                
            except Exception as e:
                error_msg = f"Error collecting market status/holidays: {e}"
                logger.error(error_msg)
                logger.debug(traceback.format_exc())
                stats['errors'].append(error_msg)
            
            logger.info("Daily collection completed")
            return stats
            
        except Exception as e:
            error_msg = f"Error in daily collection: {e}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            stats['errors'].append(error_msg)
            return stats
    
    def shutdown(self) -> None:
        """Shutdown the pipeline and release resources."""
        logger.info("Shutting down data pipeline")
        self.stop_requested = True
        self.executor.shutdown(wait=True)
        logger.info("Data pipeline shutdown complete")