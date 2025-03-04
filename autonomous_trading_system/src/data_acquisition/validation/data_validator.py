"""
Data Validator

This module provides functionality for validating market data to ensure quality and consistency.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta, date
import json

from autonomous_trading_system.src.data_acquisition.validation.validation_rules import (
    STOCK_AGGS_RULES,
    QUOTE_RULES,
    TRADE_RULES,
    OPTIONS_RULES,
    NEWS_SENTIMENT_RULES,
    FEATURE_RULES,
    MODEL_PREDICTION_RULES,
    TRADING_SIGNAL_RULES,
    SYSTEM_METRICS_RULES,
    HFT_VALIDATION_RULES,
    DAY_TRADING_VALIDATION_RULES,
    SWING_TRADING_VALIDATION_RULES,
    MARKET_MAKING_VALIDATION_RULES,
    get_adaptive_price_threshold,
    get_adaptive_volume_threshold,
    create_stock_validation_rules,
    create_crypto_validation_rules
)

from autonomous_trading_system.src.data_acquisition.storage.data_schema import SystemMetricsSchema

logger = logging.getLogger(__name__)

class DataValidator:
    """Class for validating market data."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data validator.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}

        self._load_config_from_rules()
        
        self.volatility_cache = {}  # Cache for storing recent volatility calculations
        self.volatility_regimes = {}  # Store volatility regimes by symbol
        self.market_hours_status = {}  # Store market hours status by symbol
        self.warned_symbols = set()  # Track symbols that have generated warnings
        
        
        # Redis client for caching validation results (optional)
        self.redis_client = None
        
        # Validation statistics
        self.validation_stats = {
            'total_validated': 0,
            'total_errors': 0,
            'error_types': {},
            'modified_records': 0
        }

    def _load_config_from_rules(self) -> None:
        """Load configuration from validation rules."""
        # Start with default stock aggregates rules
        default_rules = STOCK_AGGS_RULES.copy()
        
        # Override with any provided config
        for key, value in default_rules.items():
            if key not in self.config:
                self.config[key] = value
        
        # Set instance variables from config
        self.strict_mode = self.config.get('strict_mode', False)
        self.use_adaptive_thresholds = self.config.get('use_adaptive_thresholds', False)
        self.volatility_lookback_periods = self.config.get('volatility_lookback_periods', 20)
        self.volatility_multiplier = self.config.get('volatility_multiplier', 3.0)
        self.max_price_change_pct = self.config.get('max_price_change_pct', 20.0)
        self.min_price = self.config.get('min_price', 0.01)
        self.max_volume_change_pct = self.config.get('max_volume_change_pct', 1000.0)
        self.max_gap_seconds = self.config.get('max_gap_seconds', {
            '1m': 120,    # 2 minutes
            '5m': 600,    # 10 minutes
            '15m': 1800,  # 30 minutes
            '1h': 7200,   # 2 hours
            '1d': 172800  # 2 days
        })
        self.interpolate_gaps = self.config.get('interpolate_gaps', False)
    
    def set_redis_client(self, redis_client) -> None:
        """
        Set Redis client for caching validation results.
        
        Args:
            redis_client: Redis client instance
        """
        self.redis_client = redis_client
        logger.info("Redis client set for validation result caching")
    
    def load_rules_for_data_type(self, data_type: str) -> None:
        """
        Load validation rules for a specific data type.
        
        Args:
            data_type: Type of data ('stock_aggs', 'quotes', 'trades', 'options', 'news_sentiment')
        """
        if data_type == 'stock_aggs':
            self.config.update(STOCK_AGGS_RULES)
        elif data_type == 'quotes':
            self.config.update(QUOTE_RULES)
        elif data_type == 'trades':
            self.config.update(TRADE_RULES)
        elif data_type == 'options':
            self.config.update(OPTIONS_RULES)
        elif data_type == 'news_sentiment':
            self.config.update(NEWS_SENTIMENT_RULES)
        elif data_type == 'features':
            self.config.update(FEATURE_RULES)
        elif data_type == 'model_predictions':
            self.config.update(MODEL_PREDICTION_RULES)
        elif data_type == 'trading_signals':
            self.config.update(TRADING_SIGNAL_RULES)
        self._load_config_from_rules()
    
    def validate_multi_timeframe_data(self, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Validate multi-timeframe data.
        
        Args:
            data: Nested dictionary mapping symbols to timeframes to DataFrames
            
        Returns:
            Validated multi-timeframe data
        """
        logger.info("Validating multi-timeframe data")
        validated_data = {}
        
        for symbol, timeframe_data in data.items():
            validated_data[symbol] = {}
            
            for timeframe, df in timeframe_data.items():
                if df.empty:
                    continue
                    
                # Add timeframe column if not present
                if 'timeframe' not in df.columns:
                    df['timeframe'] = timeframe
                
                # Apply standard validation for each timeframe
                validated_df = self.validate_stock_aggs(df)
                
                if not validated_df.empty:
                    # Handle data gaps with interpolation if enabled
                    if self.interpolate_gaps:
                        validated_df = self._handle_data_gaps(validated_df, timeframe)
                    validated_data[symbol][timeframe] = validated_df
            
            # Apply cross-timeframe validation if multiple timeframes exist
            if len(validated_data[symbol]) > 1:
                validated_data[symbol] = self._validate_cross_timeframe_consistency(validated_data[symbol])
        
        return validated_data
    
    def validate_from_schema(self, schema_obj: Any, schema_type: str) -> Any:
        """
        Validate data from a schema object.
        
        Args:
            schema_obj: Schema object to validate
            schema_type: Type of schema
            
        Returns:
            Validated schema object
        """
        logger.debug(f"Validating {schema_type} schema object")
        
        # Load appropriate rules for this schema type
        self.load_rules_for_data_type(schema_type)
        
        # Check if we have a cached validation result
        if self.redis_client and hasattr(schema_obj, 'id'):
            cache_key = f"validation:{schema_type}:{schema_obj.id}"
            cached_result = self.redis_client.get(cache_key)
            if cached_result:
                logger.debug(f"Using cached validation result for {schema_type} {schema_obj.id}")
                return schema_obj
        
        # Apply schema-specific validation
        if schema_type == 'stock_aggs':
            # Convert schema to DataFrame for validation
            df = schema_obj.to_dataframe()
            validated_df = self.validate_stock_aggs(df)
            # Update schema object with validated data
            schema_obj.from_dataframe(validated_df)
        elif schema_type == 'system_metrics':
            # System metrics don't need validation
            pass
        
        # Cache validation result
        if self.redis_client and hasattr(schema_obj, 'id'):
            self.redis_client.set(f"validation:{schema_type}:{schema_obj.id}", "validated", ex=3600)  # 1 hour expiration
        return schema_obj
    
    def validate_stock_aggs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate stock aggregate (OHLCV) data.
        
        Args:
            df: DataFrame with stock aggregates
            
        Returns:
            Validated DataFrame
        """
        if df.empty:
            return df
        
        original_count = len(df)
        self.validation_stats['total_validated'] += original_count
        
        # Make a copy to avoid modifying the original
        validated_df = df.copy()
        
        # Ensure required columns are present
        required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in validated_df.columns]
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            logger.error(error_msg)
            self._record_error('missing_columns', error_msg)
            if self.strict_mode:
                raise ValueError(error_msg)
            return pd.DataFrame()
        
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(validated_df['timestamp']):
            validated_df['timestamp'] = pd.to_datetime(validated_df['timestamp'], errors='coerce')
            # Drop rows with invalid timestamps
            invalid_timestamps = validated_df['timestamp'].isna()
            if invalid_timestamps.any():
                error_msg = f"Found {invalid_timestamps.sum()} rows with invalid timestamps"
                logger.warning(error_msg)
                self._record_error('invalid_timestamps', error_msg)
                validated_df = validated_df[~invalid_timestamps].copy()
        
        # Sort by timestamp
        validated_df = validated_df.sort_values('timestamp')
        
        # Check for duplicates
        duplicates = validated_df.duplicated(subset=['timestamp', 'symbol'], keep='first')
        if duplicates.any():
            error_msg = f"Found {duplicates.sum()} duplicate rows"
            logger.warning(error_msg)
            self._record_error('duplicates', error_msg)
            validated_df = validated_df[~duplicates].copy()
        
        # Apply statistical anomaly detection
        validated_df = self._apply_statistical_anomaly_detection(validated_df)
        
        # Check for price anomalies
        validated_df = self._check_price_anomalies(validated_df)
        
        # Check for volume anomalies
        validated_df = self._check_volume_anomalies(validated_df)
        
        # Check for data gaps (but don't interpolate here - that's done in validate_multi_timeframe_data)
        # This just identifies and logs gaps
        if 'timeframe' in validated_df.columns:
            self._identify_data_gaps(validated_df)
        validated_df = self._check_data_gaps(validated_df)
        
        # Check for OHLC consistency
        validated_df = self._check_ohlc_consistency(validated_df)
        
        # Log validation results
        removed_count = original_count - len(validated_df)
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} rows during validation ({removed_count/original_count:.2%})")
        
        return validated_df
    
    def validate_quotes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate quote data.
        
        Args:
            df: DataFrame with quotes
            
        Returns:
            Validated DataFrame
        """
        # Load quote validation rules
        self.load_rules_for_data_type('quotes')
        
        if df.empty:
            return df
        
        original_count = len(df)
        self.validation_stats['total_validated'] += original_count
        
        # Make a copy to avoid modifying the original
        validated_df = df.copy()
        
        # Ensure required columns are present
        required_columns = ['timestamp', 'symbol', 'bid_price', 'ask_price', 'bid_size', 'ask_size']
        missing_columns = [col for col in required_columns if col not in validated_df.columns]
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            logger.error(error_msg)
            self._record_error('missing_columns', error_msg)
            if self.strict_mode:
                raise ValueError(error_msg)
            return pd.DataFrame()
        
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(validated_df['timestamp']):
            validated_df['timestamp'] = pd.to_datetime(validated_df['timestamp'], errors='coerce')
            # Drop rows with invalid timestamps
            invalid_timestamps = validated_df['timestamp'].isna()
            if invalid_timestamps.any():
                error_msg = f"Found {invalid_timestamps.sum()} rows with invalid timestamps"
                logger.warning(error_msg)
                self._record_error('invalid_timestamps', error_msg)
                validated_df = validated_df[~invalid_timestamps].copy()
        
        # Sort by timestamp
        validated_df = validated_df.sort_values('timestamp')
        
        # Check for duplicates
        duplicates = validated_df.duplicated(subset=['timestamp', 'symbol'], keep='first')
        if duplicates.any():
            error_msg = f"Found {duplicates.sum()} duplicate rows"
            logger.warning(error_msg)
            self._record_error('duplicates', error_msg)
            validated_df = validated_df[~duplicates].copy()
        
        # Check for price anomalies
        validated_df = self._check_quote_price_anomalies(validated_df)
        
        # Check for size anomalies
        validated_df = self._check_quote_size_anomalies(validated_df)
        
        # Check for bid-ask consistency
        validated_df = self._check_bid_ask_consistency(validated_df)
        
        # Log validation results
        removed_count = original_count - len(validated_df)
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} rows during validation ({removed_count/original_count:.2%})")
        
        return validated_df
    
    def validate_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate trade data.
        
        Args:
            df: DataFrame with trades
            
        Returns:
            Validated DataFrame
        """
        if df.empty:
            return df
        
        original_count = len(df)
        self.validation_stats['total_validated'] += original_count
        
        # Make a copy to avoid modifying the original
        validated_df = df.copy()
        
        # Ensure required columns are present
        required_columns = ['timestamp', 'symbol', 'price', 'size', 'exchange']
        missing_columns = [col for col in required_columns if col not in validated_df.columns]
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            logger.error(error_msg)
            self._record_error('missing_columns', error_msg)
            if self.strict_mode:
                raise ValueError(error_msg)
            return pd.DataFrame()
        
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(validated_df['timestamp']):
            validated_df['timestamp'] = pd.to_datetime(validated_df['timestamp'], errors='coerce')
            # Drop rows with invalid timestamps
            invalid_timestamps = validated_df['timestamp'].isna()
            if invalid_timestamps.any():
                error_msg = f"Found {invalid_timestamps.sum()} rows with invalid timestamps"
                logger.warning(error_msg)
                self._record_error('invalid_timestamps', error_msg)
                validated_df = validated_df[~invalid_timestamps].copy()
        
        # Sort by timestamp
        validated_df = validated_df.sort_values('timestamp')
        
        # Check for duplicates
        duplicates = validated_df.duplicated(subset=['timestamp', 'symbol', 'price', 'size', 'exchange'], keep='first')
        if duplicates.any():
            error_msg = f"Found {duplicates.sum()} duplicate rows"
            logger.warning(error_msg)
            self._record_error('duplicates', error_msg)
            validated_df = validated_df[~duplicates].copy()
        
        # Check for price anomalies
        validated_df = self._check_trade_price_anomalies(validated_df)
        
        # Check for size anomalies
        validated_df = self._check_trade_size_anomalies(validated_df)
        
        # Log validation results
        removed_count = original_count - len(validated_df)
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} rows during validation ({removed_count/original_count:.2%})")
        
        return validated_df
    
    def validate_options_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate options data.
        
        Args:
            df: DataFrame with options data
            
        Returns:
            Validated DataFrame
        """
        if df.empty:
            return df
        
        original_count = len(df)
        self.validation_stats['total_validated'] += original_count
        
        # Make a copy to avoid modifying the original
        validated_df = df.copy()
        
        # Ensure required columns are present
        required_columns = ['timestamp', 'symbol', 'underlying', 'expiration', 'strike', 'option_type']
        missing_columns = [col for col in required_columns if col not in validated_df.columns]
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            logger.error(error_msg)
            self._record_error('missing_columns', error_msg)
            if self.strict_mode:
                raise ValueError(error_msg)
            return pd.DataFrame()
        
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(validated_df['timestamp']):
            validated_df['timestamp'] = pd.to_datetime(validated_df['timestamp'], errors='coerce')
            # Drop rows with invalid timestamps
            invalid_timestamps = validated_df['timestamp'].isna()
            if invalid_timestamps.any():
                error_msg = f"Found {invalid_timestamps.sum()} rows with invalid timestamps"
                logger.warning(error_msg)
                self._record_error('invalid_timestamps', error_msg)
                validated_df = validated_df[~invalid_timestamps].copy()
        
        # Ensure expiration is in date format
        if 'expiration' in validated_df.columns and not pd.api.types.is_datetime64_any_dtype(validated_df['expiration']):
            validated_df['expiration'] = pd.to_datetime(validated_df['expiration'], errors='coerce')
            # Drop rows with invalid expiration dates
            invalid_expirations = validated_df['expiration'].isna()
            if invalid_expirations.any():
                error_msg = f"Found {invalid_expirations.sum()} rows with invalid expiration dates"
                logger.warning(error_msg)
                self._record_error('invalid_expirations', error_msg)
                validated_df = validated_df[~invalid_expirations].copy()
        
        # Sort by timestamp
        validated_df = validated_df.sort_values('timestamp')
        
        # Check for duplicates
        duplicates = validated_df.duplicated(subset=['timestamp', 'symbol'], keep='first')
        if duplicates.any():
            error_msg = f"Found {duplicates.sum()} duplicate rows"
            logger.warning(error_msg)
            self._record_error('duplicates', error_msg)
            validated_df = validated_df[~duplicates].copy()
        
        # Check for options-specific anomalies
        validated_df = self._check_options_anomalies(validated_df)
        
        # Log validation results
        removed_count = original_count - len(validated_df)
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} rows during validation ({removed_count/original_count:.2%})")
        
        return validated_df
    
    def validate_news_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate news sentiment data.
        
        Args:
            df: DataFrame with news sentiment data
            
        Returns:
            Validated DataFrame
        """
        if df.empty:
            return df
        
        original_count = len(df)
        self.validation_stats['total_validated'] += original_count
        
        # Make a copy to avoid modifying the original
        validated_df = df.copy()
        
        # Ensure required columns are present
        required_columns = ['timestamp', 'article_id', 'sentiment_score', 'sentiment_label']
        missing_columns = [col for col in required_columns if col not in validated_df.columns]
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            logger.error(error_msg)
            self._record_error('missing_columns', error_msg)
            if self.strict_mode:
                raise ValueError(error_msg)
            return pd.DataFrame()
        
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(validated_df['timestamp']):
            validated_df['timestamp'] = pd.to_datetime(validated_df['timestamp'], errors='coerce')
            # Drop rows with invalid timestamps
            invalid_timestamps = validated_df['timestamp'].isna()
            if invalid_timestamps.any():
                error_msg = f"Found {invalid_timestamps.sum()} rows with invalid timestamps"
                logger.warning(error_msg)
                self._record_error('invalid_timestamps', error_msg)
                validated_df = validated_df[~invalid_timestamps].copy()
        
        # Sort by timestamp
        validated_df = validated_df.sort_values('timestamp')
        
        # Check for duplicates
        duplicates = validated_df.duplicated(subset=['article_id'], keep='first')
        if duplicates.any():
            error_msg = f"Found {duplicates.sum()} duplicate rows"
            logger.warning(error_msg)
            self._record_error('duplicates', error_msg)
            validated_df = validated_df[~duplicates].copy()
        
        # Check for sentiment score range
        if 'sentiment_score' in validated_df.columns:
            invalid_scores = (validated_df['sentiment_score'] < -1.0) | (validated_df['sentiment_score'] > 1.0)
            if invalid_scores.any():
                error_msg = f"Found {invalid_scores.sum()} rows with sentiment scores outside [-1, 1] range"
                logger.warning(error_msg)
                self._record_error('invalid_sentiment_scores', error_msg)
                
                # Clip scores to valid range
                validated_df.loc[invalid_scores, 'sentiment_score'] = validated_df.loc[invalid_scores, 'sentiment_score'].clip(-1.0, 1.0)
                self.validation_stats['modified_records'] += invalid_scores.sum()
        
        # Log validation results
        removed_count = original_count - len(validated_df)
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} rows during validation ({removed_count/original_count:.2%})")
        
        return validated_df
    
    def validate_volume_weighted_price_pressure(self, symbol: str, timestamp: datetime, vwpp: float, reference_price: float) -> bool:
        """
        Validate volume-weighted price pressure.
        
        Args:
            symbol: Ticker symbol
            timestamp: Timestamp of the data point
            vwpp: Volume-weighted price pressure value
            reference_price: Reference price for comparison
            
        Returns:
            True if valid, False otherwise
        """
        # Get volatility regime for this symbol
        volatility_regime = self.volatility_regimes.get(symbol, 'normal')
        
        # Set threshold based on volatility regime
        if volatility_regime == 'low':
            threshold = 0.05  # 5% for low volatility
        elif volatility_regime == 'high':
            threshold = 0.20  # 20% for high volatility
        else:  # normal
            threshold = 0.10  # 10% for normal volatility
        
        # Calculate VWPP as percentage of reference price
        vwpp_pct = abs(vwpp / reference_price)
        
        # Check if VWPP exceeds threshold
        if vwpp_pct > threshold:
            error_msg = f"VWPP for {symbol} at {timestamp} exceeds threshold: {vwpp_pct:.2%} > {threshold:.2%}"
            logger.warning(error_msg)
            self._record_error('excessive_vwpp', error_msg)
            return False
        
        return True
    
    def validate_relative_strength(self, symbol: str, timestamp: datetime, relative_strength: float) -> bool:
        """
        Validate relative strength value.
        
        Args:
            symbol: Ticker symbol
            timestamp: Timestamp of the data point
            relative_strength: Relative strength value
            
        Returns:
            True if valid, False otherwise
        """
        # Check if relative strength is within reasonable bounds
        if relative_strength < -10.0 or relative_strength > 10.0:
            error_msg = f"Relative strength for {symbol} at {timestamp} outside reasonable bounds: {relative_strength}"
            logger.warning(error_msg)
            self._record_error('invalid_relative_strength', error_msg)
            return False
        
        return True
    
    def validate_order_book_imbalance(self, symbol: str, timestamp: datetime, bid_sizes: List[float], 
                                     ask_sizes: List[float], bid_prices: List[float], ask_prices: List[float]) -> bool:
        """
        Validate order book imbalance.
        
        Args:
            symbol: Ticker symbol
            timestamp: Timestamp of the data point
            bid_sizes: List of bid sizes
            ask_sizes: List of ask sizes
            bid_prices: List of bid prices
            ask_prices: List of ask prices
            
        Returns:
            True if valid, False otherwise
        """
        if not bid_sizes or not ask_sizes or not bid_prices or not ask_prices:
            error_msg = f"Empty order book data for {symbol} at {timestamp}"
            logger.warning(error_msg)
            self._record_error('empty_order_book', error_msg)
            return False
        
        # Check for negative sizes
        if any(size < 0 for size in bid_sizes) or any(size < 0 for size in ask_sizes):
            error_msg = f"Negative sizes in order book for {symbol} at {timestamp}"
            logger.warning(error_msg)
            self._record_error('negative_order_book_sizes', error_msg)
            return False
        
        # Check for negative prices
        if any(price < 0 for price in bid_prices) or any(price < 0 for price in ask_prices):
            error_msg = f"Negative prices in order book for {symbol} at {timestamp}"
            logger.warning(error_msg)
            self._record_error('negative_order_book_prices', error_msg)
            return False
        
        return True
    
    def validate_trade_flow_imbalance(self, symbol: str, timestamp: datetime, buy_volume: float, sell_volume: float) -> bool:
        """
        Validate trade flow imbalance.
        
        Args:
            symbol: Ticker symbol
            timestamp: Timestamp of the data point
            buy_volume: Buy volume
            sell_volume: Sell volume
            
        Returns:
            True if valid, False otherwise
        """
        if buy_volume < 0 or sell_volume < 0:
            error_msg = f"Negative volume in trade flow for {symbol} at {timestamp}"
            logger.warning(error_msg)
            self._record_error('negative_trade_flow_volume', error_msg)
            return False
        
        return True
    
    def validate_price_change(self, symbol: str, previous_price: float, current_price: float) -> bool:
        """
        Validate price change between two consecutive prices.
        
        Args:
            symbol: Ticker symbol
            previous_price: Previous price
            current_price: Current price
            
        Returns:
            True if valid, False otherwise
        """
        # Get volatility regime for this symbol
        volatility_regime = self.volatility_regimes.get(symbol, 'normal')
        
        # Set threshold based on volatility regime
        if volatility_regime == 'low':
            threshold = self.max_price_change_pct * 0.5  # 50% of normal threshold for low volatility
        elif volatility_regime == 'high':
            threshold = self.max_price_change_pct * 2.0  # 200% of normal threshold for high volatility
        else:  # normal
            threshold = self.max_price_change_pct
        
        # Calculate price change percentage
        if previous_price == 0:
            return True  # Can't calculate percentage change from zero
        
        price_change_pct = abs((current_price - previous_price) / previous_price) * 100
        
        # Check if price change exceeds threshold
        if price_change_pct > threshold:
            error_msg = f"Price change for {symbol} exceeds threshold: {price_change_pct:.2f}% > {threshold:.2f}%"
            logger.warning(error_msg)
            self._record_error('excessive_price_change', error_msg)
            return False
        
        return True
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """
        Get validation statistics.
        
        Returns:
            Dictionary with validation statistics
        """
        return self.validation_stats
    
    def log_validation_stats(self, component: str = 'data_validator') -> None:
        """
        Log validation statistics to system metrics.
        
        Args:
            component: Component name
        """
        if self.redis_client:
            self.redis_client.store_component_status(component, self.validation_stats)
            logger.info(f"Logged validation stats to Redis: {json.dumps(self.validation_stats)}")
    
    def reset_validation_stats(self) -> None:
        """Reset validation statistics."""
        self.validation_stats = {
            'total_validated': 0,
            'total_errors': 0,
            'error_types': {},
            'modified_records': 0
        }
    
    def set_volatility_regime(self, symbol: str, volatility_regime: str) -> None:
        """
        Set volatility regime for a symbol.
        
        Args:
            symbol: Ticker symbol
            volatility_regime: Volatility regime ('low', 'normal', 'high')
        """
        if volatility_regime not in ['low', 'normal', 'high']:
            logger.warning(f"Invalid volatility regime: {volatility_regime}. Using 'normal'.")
            volatility_regime = 'normal'
        
        self.volatility_regimes[symbol] = volatility_regime
        logger.debug(f"Set volatility regime for {symbol} to {volatility_regime}")
    
    def set_market_hours_status(self, symbol: str, is_market_hours: bool) -> None:
        """
        Set market hours status for a symbol.
        
        Args:
            symbol: Ticker symbol
            is_market_hours: Whether current time is during market hours
        """
        self.market_hours_status[symbol] = is_market_hours
        logger.debug(f"Set market hours status for {symbol} to {is_market_hours}")
        
        # Adjust validation thresholds based on market hours
        if not is_market_hours and symbol not in self.warned_symbols:
            logger.info(f"Using relaxed validation thresholds for {symbol} (outside market hours)")
            self.warned_symbols.add(symbol)
    
    def _record_error(self, error_type: str, error_msg: str) -> None:
        """
        Record an error in the validation statistics.
        
        Args:
            error_type: Type of error
            error_msg: Error message
        """
        self.validation_stats['total_errors'] += 1
        if error_type not in self.validation_stats['error_types']:
            self.validation_stats['error_types'][error_type] = 0
        self.validation_stats['error_types'][error_type] += 1
    
    def _calculate_adaptive_threshold(self, df: pd.DataFrame, base_threshold: float) -> float:
        """
        Calculate adaptive threshold based on recent market volatility.
        
        Args:
            df: DataFrame with price data
            base_threshold: Base threshold value
            
        Returns:
            Adjusted threshold value
        """
        if not self.use_adaptive_thresholds or len(df) < self.volatility_lookback_periods:
            return get_adaptive_price_threshold(df, base_threshold, self.volatility_lookback_periods, self.volatility_multiplier)
        
        # Calculate recent volatility (standard deviation of returns)
        returns = df['close'].pct_change().dropna()
        recent_volatility = returns.tail(self.volatility_lookback_periods).std() * 100
        
        # Adjust threshold based on volatility
        adjusted_threshold = base_threshold * (1 + recent_volatility * self.volatility_multiplier)
        
        return adjusted_threshold
    
    def _check_price_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check for price anomalies in OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Validated DataFrame
        """
        # Check for negative prices
        negative_prices = (
            (df['open'] < 0) | 
            (df['high'] < 0) | 
            (df['low'] < 0) | 
            (df['close'] < 0)
        )
        if negative_prices.any():
            error_msg = f"Found {negative_prices.sum()} rows with negative prices"
            logger.warning(error_msg)
            self._record_error('negative_prices', error_msg)
            df = df[~negative_prices].copy()
        
        # Check for prices below minimum threshold
        low_prices = (
            (df['open'] < self.min_price) | 
            (df['high'] < self.min_price) | 
            (df['low'] < self.min_price) | 
            (df['close'] < self.min_price)
        )
        if low_prices.any():
            error_msg = f"Found {low_prices.sum()} rows with prices below {self.min_price}"
            logger.warning(error_msg)
            self._record_error('low_prices', error_msg)
            if self.strict_mode:
                df = df[~low_prices].copy()
        
        # Check for extreme price changes
        if len(df) > 1:
            # Group by symbol and calculate price changes
            for symbol in df['symbol'].unique():
                symbol_df = df[df['symbol'] == symbol].copy()
                if len(symbol_df) <= 1:
                    continue
                
                # Calculate percentage change in close price
                symbol_df['close_pct_change'] = symbol_df['close'].pct_change().abs() * 100
                
                # Get threshold (adaptive or fixed)
                threshold = self.max_price_change_pct
                if self.use_adaptive_thresholds:
                    threshold = self._calculate_adaptive_threshold(
                        symbol_df, self.max_price_change_pct
                    )
                
                # Identify extreme changes
                extreme_changes = symbol_df['close_pct_change'] > threshold
                if extreme_changes.any():
                    error_msg = f"Found {extreme_changes.sum()} rows with extreme price changes for {symbol}"
                    logger.warning(error_msg)
                    self._record_error('extreme_price_changes', error_msg)
                    
                    # Remove rows with extreme changes if in strict mode
                    if self.strict_mode:
                        extreme_indices = symbol_df[extreme_changes].index
                        df = df[~df.index.isin(extreme_indices)].copy()
        
        return df
    
    def _check_volume_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check for volume anomalies in OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Validated DataFrame
        """
        # Check for negative volume
        negative_volume = df['volume'] < 0
        if negative_volume.any():
            error_msg = f"Found {negative_volume.sum()} rows with negative volume"
            logger.warning(error_msg)
            self._record_error('negative_volume', error_msg)
            df = df[~negative_volume].copy()
        
        # Check for extreme volume changes
        if len(df) > 1:
            # Group by symbol and calculate volume changes
            for symbol in df['symbol'].unique():
                symbol_df = df[df['symbol'] == symbol].copy()
                if len(symbol_df) <= 1:
                    continue
                
                # Calculate percentage change in volume
                symbol_df['volume_pct_change'] = symbol_df['volume'].pct_change().abs() * 100
                
                # Get threshold (adaptive or fixed)
                threshold = self.max_volume_change_pct
                if self.use_adaptive_thresholds:
                    threshold = self._calculate_adaptive_threshold(
                        symbol_df, self.max_volume_change_pct
                    )
                
                # Identify extreme changes
                extreme_changes = symbol_df['volume_pct_change'] > threshold
                if extreme_changes.any():
                    error_msg = f"Found {extreme_changes.sum()} rows with extreme volume changes for {symbol}"
                    logger.warning(error_msg)
                    self._record_error('extreme_volume_changes', error_msg)
                    
                    # Remove rows with extreme changes if in strict mode
                    if self.strict_mode:
                        extreme_indices = symbol_df[extreme_changes].index
                        df = df[~df.index.isin(extreme_indices)].copy()
        
        return df
    
    def _check_data_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check for data gaps in time series.
        
        Args:
            df: DataFrame with time series data
            
        Returns:
            Validated DataFrame
        """
        if 'timeframe' not in df.columns:
            return df
        
        # Group by symbol and timeframe
        for symbol in df['symbol'].unique():
            for timeframe in df[df['symbol'] == symbol]['timeframe'].unique():
                # Get max gap seconds for this timeframe
                max_gap_seconds = self.max_gap_seconds.get(timeframe, 3600)  # Default to 1 hour
                
                # Filter data for this symbol and timeframe
                symbol_tf_df = df[(df['symbol'] == symbol) & (df['timeframe'] == timeframe)].copy()
                if len(symbol_tf_df) <= 1:
                    continue
                
                # Sort by timestamp
                symbol_tf_df = symbol_tf_df.sort_values('timestamp')
                
                # Calculate time difference between consecutive rows
                symbol_tf_df['time_diff'] = symbol_tf_df['timestamp'].diff().dt.total_seconds()
                
                # Identify gaps
                gaps = symbol_tf_df['time_diff'] > max_gap_seconds
                if gaps.any():
                    error_msg = f"Found {gaps.sum()} data gaps for {symbol} ({timeframe})"
                    logger.warning(error_msg)
                    self._record_error('data_gaps', error_msg)
                    
                    # Log details of gaps
                    gap_rows = symbol_tf_df[gaps]
                    for _, row in gap_rows.iterrows():
                        gap_seconds = row['time_diff']
                        gap_minutes = gap_seconds / 60
                        prev_timestamp = row['timestamp'] - timedelta(seconds=gap_seconds)
                        logger.debug(f"Gap of {gap_minutes:.2f} minutes between {prev_timestamp} and {row['timestamp']} for {symbol} ({timeframe})")
        
        return df
    
    def _identify_data_gaps(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Identify data gaps in time series without modifying the data.
        
        Args:
            df: DataFrame with time series data
            
        Returns:
            List of identified gaps with details
        """
        if df.empty or 'timeframe' not in df.columns or 'timestamp' not in df.columns:
            return []
        
        gaps = []
        
        # Group by symbol and timeframe
        for symbol in df['symbol'].unique():
            for timeframe in df[df['symbol'] == symbol]['timeframe'].unique():
                # Get max gap seconds for this timeframe
                max_gap_seconds = self.max_gap_seconds.get(timeframe, 3600)  # Default to 1 hour
                
                # Filter data for this symbol and timeframe
                symbol_tf_df = df[(df['symbol'] == symbol) & (df['timeframe'] == timeframe)].copy()
                if len(symbol_tf_df) <= 1:
                    continue
                
                # Sort by timestamp
                symbol_tf_df = symbol_tf_df.sort_values('timestamp')
                
                # Calculate time difference between consecutive rows
                symbol_tf_df['time_diff'] = symbol_tf_df['timestamp'].diff().dt.total_seconds()
                
                # Identify gaps
                gap_rows = symbol_tf_df[symbol_tf_df['time_diff'] > max_gap_seconds]
                
                for _, row in gap_rows.iterrows():
                    gap_seconds = row['time_diff']
                    gap_minutes = gap_seconds / 60
                    prev_timestamp = row['timestamp'] - timedelta(seconds=gap_seconds)
                    
                    gap_info = {
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'start_time': prev_timestamp,
                        'end_time': row['timestamp'],
                        'gap_seconds': gap_seconds,
                        'gap_minutes': gap_minutes
                    }
                    
                    gaps.append(gap_info)
                    
                    logger.debug(f"Gap of {gap_minutes:.2f} minutes between {prev_timestamp} and {row['timestamp']} for {symbol} ({timeframe})")
        
        return gaps
    
    def _handle_data_gaps(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Handle data gaps in time series with interpolation.
        
        Args:
            df: DataFrame with time series data
            timeframe: Data timeframe
            
        Returns:
            DataFrame with gaps handled
        """
        if df.empty or 'timestamp' not in df.columns or not self.interpolate_gaps:
            return df
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        # Determine expected frequency based on timeframe
        freq = None
        if timeframe.endswith('m'):
            freq = f"{timeframe[:-1]}min"
        elif timeframe.endswith('h'):
            freq = f"{timeframe[:-1]}H"
        elif timeframe.endswith('d'):
            freq = f"{timeframe[:-1]}D"
        
        if not freq:
            return df
        
        # Group by symbol
        symbols = df['symbol'].unique()
        result_dfs = []
        
        for symbol in symbols:
            symbol_df = df[df['symbol'] == symbol].copy()
            
            # Create expected timestamp range
            start_time = symbol_df['timestamp'].min()
            end_time = symbol_df['timestamp'].max()
            expected_timestamps = pd.date_range(start=start_time, end=end_time, freq=freq)
            
            # Check for missing timestamps
            actual_timestamps = set(symbol_df['timestamp'])
            missing_timestamps = [ts for ts in expected_timestamps if ts not in actual_timestamps]
            
            if not missing_timestamps:
                result_dfs.append(symbol_df)
                continue
            
            # Log gap information
            error_msg = f"Found {len(missing_timestamps)} missing timestamps in {timeframe} data for {symbol}"
            logger.warning(error_msg)
            self._record_error('missing_timestamps', error_msg)
            
            # Set timestamp as index for easier interpolation
            symbol_df.set_index('timestamp', inplace=True)
            
            # Create a new DataFrame with all expected timestamps
            full_idx_df = pd.DataFrame(index=expected_timestamps)
            
            # Join with actual data
            merged_df = full_idx_df.join(symbol_df, how='left')
            
            # Fill symbol column
            merged_df['symbol'] = symbol
            
            # Fill timeframe column
            merged_df['timeframe'] = timeframe
            
            # Interpolate numeric columns
            numeric_cols = ['open', 'high', 'low', 'close']
            merged_df[numeric_cols] = merged_df[numeric_cols].interpolate(method='linear')
            
            # Forward fill non-numeric columns
            non_numeric_cols = [col for col in merged_df.columns if col not in numeric_cols and col != 'volume']
            merged_df[non_numeric_cols] = merged_df[non_numeric_cols].fillna(method='ffill')
            
            # Handle volume separately - use 0 or average
            if 'volume' in merged_df.columns:
                # Use average volume for missing values
                avg_volume = symbol_df['volume'].mean()
                merged_df['volume'].fillna(avg_volume, inplace=True)
            
            # Reset index to get timestamp as column again
            merged_df.reset_index(inplace=True)
            merged_df.rename(columns={'index': 'timestamp'}, inplace=True)
            
            # Add to results
            result_dfs.append(merged_df)
            
            # Update stats
            self.validation_stats['modified_records'] += len(missing_timestamps)
        
        # Combine all symbol DataFrames
        if result_dfs:
            return pd.concat(result_dfs, ignore_index=True)
        else:
            return df
    
    def _validate_cross_timeframe_consistency(self, timeframe_data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Validate consistency across timeframes.
        
        Args:
            timeframe_data: Dictionary mapping timeframes to DataFrames
            
        Returns:
            Validated timeframe data
        """
        # Sort timeframes by granularity (smallest to largest)
        timeframe_minutes = {}
        for tf in timeframe_data.keys():
            if tf.endswith('m'):
                timeframe_minutes[tf] = int(tf[:-1])
            elif tf.endswith('h'):
                timeframe_minutes[tf] = int(tf[:-1]) * 60
            elif tf.endswith('d'):
                timeframe_minutes[tf] = int(tf[:-1]) * 60 * 24
        
        sorted_timeframes = sorted(timeframe_data.keys(), key=lambda tf: timeframe_minutes.get(tf, 0))
        
        # Skip if only one timeframe
        if len(sorted_timeframes) <= 1:
            return timeframe_data
        
        # Compare adjacent timeframes for consistency
        for i in range(len(sorted_timeframes) - 1):
            smaller_tf = sorted_timeframes[i]
            larger_tf = sorted_timeframes[i + 1]
            
            smaller_df = timeframe_data[smaller_tf]
            larger_df = timeframe_data[larger_tf]
            
            # Skip if either DataFrame is empty
            if smaller_df.empty or larger_df.empty:
                continue
            
            # Resample smaller timeframe to match larger timeframe
            resampled_df = self._resample_timeframe(smaller_df, smaller_tf, larger_tf)
            
            # Compare resampled data with larger timeframe data
            inconsistencies = self._detect_timeframe_inconsistencies(resampled_df, larger_df)
            
            if inconsistencies:
                error_msg = f"Found {len(inconsistencies)} cross-timeframe inconsistencies between {smaller_tf} and {larger_tf}"
                logger.warning(error_msg)
                self._record_error('cross_timeframe_inconsistencies', error_msg)
                
                # Log details of inconsistencies
                for ts, details in inconsistencies.items():
                    logger.debug(f"Inconsistency at {ts}: {details}")
        
        return timeframe_data
    
    def _resample_timeframe(self, df: pd.DataFrame, source_tf: str, target_tf: str) -> pd.DataFrame:
        """
        Resample data from a smaller timeframe to a larger timeframe.
        
        Args:
            df: DataFrame with smaller timeframe data
            source_tf: Source timeframe
            target_tf: Target timeframe
            
        Returns:
            Resampled DataFrame
        """
        if df.empty or 'timestamp' not in df.columns:
            return pd.DataFrame()
        
        # Determine resampling frequency
        freq = None
        if target_tf.endswith('m'):
            freq = f"{target_tf[:-1]}min"
        elif target_tf.endswith('h'):
            freq = f"{target_tf[:-1]}H"
        elif target_tf.endswith('d'):
            freq = f"{target_tf[:-1]}D"
        
        if not freq:
            return pd.DataFrame()
        
        # Set timestamp as index
        df_copy = df.copy()
        df_copy.set_index('timestamp', inplace=True)
        
        # Resample OHLCV data
        resampled = pd.DataFrame()
        resampled['open'] = df_copy['open'].resample(freq).first()
        resampled['high'] = df_copy['high'].resample(freq).max()
        resampled['low'] = df_copy['low'].resample(freq).min()
        resampled['close'] = df_copy['close'].resample(freq).last()
        resampled['volume'] = df_copy['volume'].resample(freq).sum()
        
        # Reset index
        resampled.reset_index(inplace=True)
        
        # Add symbol and timeframe columns
        if 'symbol' in df.columns:
            symbol = df['symbol'].iloc[0]
            resampled['symbol'] = symbol
        
        resampled['timeframe'] = target_tf
        
        return resampled
    
    def _detect_timeframe_inconsistencies(self, resampled_df: pd.DataFrame, target_df: pd.DataFrame) -> Dict[datetime, Dict[str, Any]]:
        """
        Detect inconsistencies between resampled data and target timeframe data.
        
        Args:
            resampled_df: Resampled DataFrame from smaller timeframe
            target_df: DataFrame with target timeframe data
            
        Returns:
            Dictionary mapping timestamps to inconsistency details
        """
        if resampled_df.empty or target_df.empty:
            return {}
        
        # Ensure both DataFrames have timestamp as column
        if 'timestamp' not in resampled_df.columns or 'timestamp' not in target_df.columns:
            return {}
        
        # Find common timestamps
        resampled_times = set(resampled_df['timestamp'])
        target_times = set(target_df['timestamp'])
        common_times = resampled_times.intersection(target_times)
        
        inconsistencies = {}
        
        # Check each common timestamp
        for ts in common_times:
            resampled_row = resampled_df[resampled_df['timestamp'] == ts].iloc[0]
            target_row = target_df[target_df['timestamp'] == ts].iloc[0]
            
            # Check OHLC consistency
            issues = {}
            
            # High should match
            high_diff_pct = abs(resampled_row['high'] - target_row['high']) / target_row['high'] * 100
            if high_diff_pct > 1.0:  # Allow 1% difference
                issues['high'] = {
                    'resampled': resampled_row['high'],
                    'target': target_row['high'],
                    'diff_pct': high_diff_pct
                }
            
            # Low should match
            low_diff_pct = abs(resampled_row['low'] - target_row['low']) / target_row['low'] * 100
            if low_diff_pct > 1.0:  # Allow 1% difference
                issues['low'] = {
                    'resampled': resampled_row['low'],
                    'target': target_row['low'],
                    'diff_pct': low_diff_pct
                }
            
            # Open and close might differ slightly due to timing differences
            open_diff_pct = abs(resampled_row['open'] - target_row['open']) / target_row['open'] * 100
            if open_diff_pct > 2.0:  # Allow 2% difference
                issues['open'] = {
                    'resampled': resampled_row['open'],
                    'target': target_row['open'],
                    'diff_pct': open_diff_pct
                }
            
            close_diff_pct = abs(resampled_row['close'] - target_row['close']) / target_row['close'] * 100
            if close_diff_pct > 2.0:  # Allow 2% difference
                issues['close'] = {
                    'resampled': resampled_row['close'],
                    'target': target_row['close'],
                    'diff_pct': close_diff_pct
                }
            
            # If any issues found, add to inconsistencies
            if issues:
                inconsistencies[ts] = issues
        
        return inconsistencies
    
    def _apply_statistical_anomaly_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply statistical anomaly detection to identify outliers.
        
        Args:
            df: DataFrame with time series data
            
        Returns:
            DataFrame with anomalies handled
        """
        if df.empty:
            return df
        
        # Make a copy to avoid modifying the original
        result_df = df.copy()
        
        # Group by symbol
        for symbol in result_df['symbol'].unique():
            symbol_df = result_df[result_df['symbol'] == symbol].copy()
            
            # Skip if not enough data
            if len(symbol_df) < 10:
                continue
            
            # Detect price anomalies using Z-score method
            price_anomalies = self._detect_statistical_anomalies(symbol_df, 'close')
            volume_anomalies = self._detect_statistical_anomalies(symbol_df, 'volume')
            
            # Combine anomalies
            combined_anomalies = price_anomalies | volume_anomalies
            
            if combined_anomalies.any():
                anomaly_count = combined_anomalies.sum()
                error_msg = f"Found {anomaly_count} statistical anomalies for {symbol}"
                logger.warning(error_msg)
                self._record_error('statistical_anomalies', error_msg)
                
                # Remove anomalies if in strict mode
                if self.strict_mode:
                    anomaly_indices = symbol_df[combined_anomalies].index
                    result_df = result_df[~result_df.index.isin(anomaly_indices)].copy()
        
        return result_df
    
    def _detect_statistical_anomalies(self, df: pd.DataFrame, column: str, z_threshold: float = 3.0) -> pd.Series:
        """
        Detect statistical anomalies using Z-score method.
        
        Args:
            df: DataFrame with data
            column: Column to check for anomalies
            z_threshold: Z-score threshold for anomaly detection
            
        Returns:
            Boolean series indicating anomalies
        """
        if df.empty or column not in df.columns:
            return pd.Series([], dtype=bool)
        
        # Calculate rolling mean and standard deviation
        rolling_mean = df[column].rolling(window=20, min_periods=5).mean()
        rolling_std = df[column].rolling(window=20, min_periods=5).std()
        
        # Calculate Z-scores
        z_scores = (df[column] - rolling_mean) / rolling_std
        
        # Identify anomalies
        anomalies = z_scores.abs() > z_threshold
        
        return anomalies
    
    def _check_ohlc_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check for OHLC consistency in price data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Validated DataFrame
        """
        # Check high >= low
        high_low_inconsistent = df['high'] < df['low']
        if high_low_inconsistent.any():
            error_msg = f"Found {high_low_inconsistent.sum()} rows with high < low"
            logger.warning(error_msg)
            self._record_error('high_low_inconsistent', error_msg)
            
            if self.strict_mode:
                df = df[~high_low_inconsistent].copy()
            else:
                # Fix inconsistencies by swapping high and low
                inconsistent_indices = df[high_low_inconsistent].index
                df.loc[inconsistent_indices, ['high', 'low']] = df.loc[inconsistent_indices, ['low', 'high']].values
                self.validation_stats['modified_records'] += len(inconsistent_indices)
        
        # Check high >= open and high >= close
        high_inconsistent = (df['high'] < df['open']) | (df['high'] < df['close'])
        if high_inconsistent.any():
            error_msg = f"Found {high_inconsistent.sum()} rows with high < open/close"
            logger.warning(error_msg)
            self._record_error('high_inconsistent', error_msg)
            
            if self.strict_mode:
                df = df[~high_inconsistent].copy()
            else:
                # Fix inconsistencies by setting high to max(open, close, high)
                inconsistent_indices = df[high_inconsistent].index
                df.loc[inconsistent_indices, 'high'] = df.loc[inconsistent_indices, ['open', 'close', 'high']].max(axis=1)
                self.validation_stats['modified_records'] += len(inconsistent_indices)
        
        # Check low <= open and low <= close
        low_inconsistent = (df['low'] > df['open']) | (df['low'] > df['close'])
        if low_inconsistent.any():
            error_msg = f"Found {low_inconsistent.sum()} rows with low > open/close"
            logger.warning(error_msg)
            self._record_error('low_inconsistent', error_msg)
            
            if self.strict_mode:
                df = df[~low_inconsistent].copy()
            else:
                # Fix inconsistencies by setting low to min(open, close, low)
                inconsistent_indices = df[low_inconsistent].index
                df.loc[inconsistent_indices, 'low'] = df.loc[inconsistent_indices, ['open', 'close', 'low']].min(axis=1)
                self.validation_stats['modified_records'] += len(inconsistent_indices)
        
        return df
    
    def _check_quote_price_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check for price anomalies in quote data.
        
        Args:
            df: DataFrame with quote data
            
        Returns:
            Validated DataFrame
        """
        # Check for negative prices
        negative_prices = (df['bid_price'] < 0) | (df['ask_price'] < 0)
        if negative_prices.any():
            error_msg = f"Found {negative_prices.sum()} rows with negative prices"
            logger.warning(error_msg)
            self._record_error('negative_prices', error_msg)
            df = df[~negative_prices].copy()
        
        # Check for prices below minimum threshold
        low_prices = (df['bid_price'] < self.min_price) | (df['ask_price'] < self.min_price)
        if low_prices.any():
            error_msg = f"Found {low_prices.sum()} rows with prices below {self.min_price}"
            logger.warning(error_msg)
            self._record_error('low_prices', error_msg)
            if self.strict_mode:
                df = df[~low_prices].copy()
        
        return df
    
    def _check_quote_size_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check for size anomalies in quote data.
        
        Args:
            df: DataFrame with quote data
            
        Returns:
            Validated DataFrame
        """
        # Check for negative sizes
        negative_sizes = (df['bid_size'] < 0) | (df['ask_size'] < 0)
        if negative_sizes.any():
            error_msg = f"Found {negative_sizes.sum()} rows with negative sizes"
            logger.warning(error_msg)
            self._record_error('negative_sizes', error_msg)
            df = df[~negative_sizes].copy()
        
        return df
    
    def _check_bid_ask_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check for bid-ask consistency in quote data.
        
        Args:
            df: DataFrame with quote data
            
        Returns:
            Validated DataFrame
        """
        # Check for crossed quotes (bid > ask)
        crossed_quotes = df['bid_price'] > df['ask_price']
        if crossed_quotes.any():
            error_msg = f"Found {crossed_quotes.sum()} rows with crossed quotes (bid > ask)"
            logger.warning(error_msg)
            self._record_error('crossed_quotes', error_msg)
            
            if self.strict_mode:
                df = df[~crossed_quotes].copy()
        
        # Check for zero spread (bid == ask)
        zero_spread = df['bid_price'] == df['ask_price']
        if zero_spread.any():
            error_msg = f"Found {zero_spread.sum()} rows with zero spread (bid == ask)"
            logger.warning(error_msg)
            self._record_error('zero_spread', error_msg)
            
            # This is unusual but not necessarily invalid, so we don't remove these rows
        
        return df
    
    def _check_trade_price_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check for price anomalies in trade data.
        
        Args:
            df: DataFrame with trade data
            
        Returns:
            Validated DataFrame
        """
        # Check for negative prices
        negative_prices = df['price'] < 0
        if negative_prices.any():
            error_msg = f"Found {negative_prices.sum()} rows with negative prices"
            logger.warning(error_msg)
            self._record_error('negative_prices', error_msg)
            df = df[~negative_prices].copy()
        
        # Check for prices below minimum threshold
        low_prices = df['price'] < self.min_price
        if low_prices.any():
            error_msg = f"Found {low_prices.sum()} rows with prices below {self.min_price}"
            logger.warning(error_msg)
            self._record_error('low_prices', error_msg)
            if self.strict_mode:
                df = df[~low_prices].copy()
        
        # Check for extreme price changes
        if len(df) > 1:
            # Group by symbol and calculate price changes
            for symbol in df['symbol'].unique():
                symbol_df = df[df['symbol'] == symbol].copy()
                if len(symbol_df) <= 1:
                    continue
                
                # Calculate percentage change in price
                symbol_df['price_pct_change'] = symbol_df['price'].pct_change().abs() * 100
                
                # Identify extreme changes
                extreme_changes = symbol_df['price_pct_change'] > self.max_price_change_pct
                if extreme_changes.any():
                    error_msg = f"Found {extreme_changes.sum()} rows with extreme price changes for {symbol}"
                    logger.warning(error_msg)
                    self._record_error('extreme_price_changes', error_msg)
                    
                    # Remove rows with extreme changes if in strict mode
                    if self.strict_mode:
                        extreme_indices = symbol_df[extreme_changes].index
                        df = df[~df.index.isin(extreme_indices)].copy()
        
        return df
    
    def _check_trade_size_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check for size anomalies in trade data.
        
        Args:
            df: DataFrame with trade data
            
        Returns:
            Validated DataFrame
        """
        # Check for negative sizes
        negative_sizes = df['size'] < 0
        if negative_sizes.any():
            error_msg = f"Found {negative_sizes.sum()} rows with negative sizes"
            logger.warning(error_msg)
            self._record_error('negative_sizes', error_msg)
            df = df[~negative_sizes].copy()
        
        # Check for zero sizes
        zero_sizes = df['size'] == 0
        if zero_sizes.any():
            error_msg = f"Found {zero_sizes.sum()} rows with zero sizes"
            logger.warning(error_msg)
            self._record_error('zero_sizes', error_msg)
            df = df[~zero_sizes].copy()
        
        return df
    
    def _check_options_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check for anomalies in options data.
        
        Args:
            df: DataFrame with options data
            
        Returns:
            Validated DataFrame
        """
        # Check for invalid option types
        if 'option_type' in df.columns:
            invalid_types = ~df['option_type'].isin(['call', 'put', 'C', 'P'])
            if invalid_types.any():
                error_msg = f"Found {invalid_types.sum()} rows with invalid option types"
                logger.warning(error_msg)
                self._record_error('invalid_option_types', error_msg)
                df = df[~invalid_types].copy()
        
        # Check for negative strikes
        if 'strike' in df.columns:
            negative_strikes = df['strike'] < 0
            if negative_strikes.any():
                error_msg = f"Found {negative_strikes.sum()} rows with negative strikes"
                logger.warning(error_msg)
                self._record_error('negative_strikes', error_msg)
                df = df[~negative_strikes].copy()
        
        # Check for expired options
        if 'expiration' in df.columns:
            now = pd.Timestamp.now().normalize()
            expired = df['expiration'] < now
            if expired.any():
                error_msg = f"Found {expired.sum()} rows with expired options"
                logger.warning(error_msg)
                self._record_error('expired_options', error_msg)
                # We don't remove expired options as they might be needed for historical analysis
        
        return df
