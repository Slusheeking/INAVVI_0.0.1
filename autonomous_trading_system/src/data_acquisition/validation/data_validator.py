"""
Data Validator

This module provides functionality for validating market data to ensure quality and consistency.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta

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
        
        # Default configuration
        self.strict_mode = self.config.get('strict_mode', False)
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
        
        # Validation statistics
        self.validation_stats = {
            'total_validated': 0,
            'total_errors': 0,
            'error_types': {},
            'modified_records': 0
        }
    
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
        
        # Check for price anomalies
        validated_df = self._check_price_anomalies(validated_df)
        
        # Check for volume anomalies
        validated_df = self._check_volume_anomalies(validated_df)
        
        # Check for data gaps
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
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """
        Get validation statistics.
        
        Returns:
            Dictionary with validation statistics
        """
        return self.validation_stats
    
    def reset_validation_stats(self) -> None:
        """Reset validation statistics."""
        self.validation_stats = {
            'total_validated': 0,
            'total_errors': 0,
            'error_types': {},
            'modified_records': 0
        }
    
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
                
                # Identify extreme changes
                extreme_changes = symbol_df['close_pct_change'] > self.max_price_change_pct
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
                
                # Identify extreme changes
                extreme_changes = symbol_df['volume_pct_change'] > self.max_volume_change_pct
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