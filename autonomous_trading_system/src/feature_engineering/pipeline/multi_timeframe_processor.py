"""
Multi-Timeframe Processor

This module provides functionality for processing validated multi-timeframe data
and generating features across multiple timeframes with consistency checks.
"""

import logging
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
import pandas as pd
import concurrent.futures
from datetime import datetime, timedelta
import redis
import json
import pickle

from autonomous_trading_system.src.data_acquisition.validation.data_validator import DataValidator
from autonomous_trading_system.src.feature_engineering.calculators.price_features import PriceFeatureCalculator
from autonomous_trading_system.src.feature_engineering.calculators.volume_features import VolumeFeatureCalculator
from autonomous_trading_system.src.feature_engineering.calculators.volatility_features import VolatilityFeatureCalculator
from autonomous_trading_system.src.feature_engineering.calculators.momentum_features import MomentumFeatureCalculator
from autonomous_trading_system.src.feature_engineering.calculators.trend_features import TrendFeatureCalculator
from autonomous_trading_system.src.feature_engineering.calculators.pattern_features import PatternFeatureCalculator
from autonomous_trading_system.src.feature_engineering.calculators.microstructure_features import MicrostructureFeatureCalculator
from autonomous_trading_system.src.feature_engineering.store.feature_cache import FeatureStoreCache

logger = logging.getLogger(__name__)

class MultiTimeframeProcessor:
    """
    Processes validated multi-timeframe data to generate features across multiple timeframes.
    
    This class serves as a bridge between the data acquisition and feature engineering subsystems,
    taking validated multi-timeframe data and transforming it into features that can be used
    for model training and trading decisions.
    """
    
    def __init__(
        self,
        feature_store_cache: Optional[FeatureStoreCache] = None,
        use_redis_cache: bool = True,
        redis_cache_ttl: int = 3600,  # 1 hour
        parallel_processing: bool = True,
        max_workers: int = 10
    ):
        """
        Initialize the MultiTimeframeProcessor.
        
        Args:
            feature_store_cache: Cache for storing and retrieving features
            use_redis_cache: Whether to use Redis for caching
            redis_cache_ttl: Time-to-live for Redis cache entries (seconds)
            parallel_processing: Whether to use parallel processing
            max_workers: Maximum number of worker threads for parallel processing
        """
        self.feature_store_cache = feature_store_cache or FeatureStoreCache(use_redis=use_redis_cache)
        self.use_redis_cache = use_redis_cache
        self.redis_cache_ttl = redis_cache_ttl
        self.parallel_processing = parallel_processing
        self.max_workers = max_workers
        
        # Initialize feature calculators
        self.price_calculator = PriceFeatureCalculator()
        self.volume_calculator = VolumeFeatureCalculator()
        self.volatility_calculator = VolatilityFeatureCalculator()
        self.momentum_calculator = MomentumFeatureCalculator()
        self.trend_calculator = TrendFeatureCalculator()
        self.pattern_calculator = PatternFeatureCalculator()
        self.microstructure_calculator = MicrostructureFeatureCalculator()
        
        # Initialize data validator for cross-timeframe consistency checks
        self.validator = DataValidator()
        
        logger.info("MultiTimeframeProcessor initialized")
    
    def process_multi_timeframe_data(
        self,
        data: Dict[str, Dict[str, pd.DataFrame]],
        feature_groups: Optional[List[str]] = None,
        include_target: bool = True,
        target_horizon: int = 1,
        force_recalculate: bool = False
    ) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Process multi-timeframe data to generate features.
        
        Args:
            data: Nested dictionary mapping symbols to timeframes to DataFrames
            feature_groups: List of feature groups to calculate (price, volume, volatility, etc.)
            include_target: Whether to include target variables
            target_horizon: Horizon for target variable calculation (in bars)
            force_recalculate: Whether to force recalculation of features (ignore cache)
            
        Returns:
            Nested dictionary mapping symbols to timeframes to DataFrames with features
        """
        logger.info(f"Processing multi-timeframe data for {len(data)} symbols")
        
        if feature_groups is None:
            feature_groups = [
                'price', 'volume', 'volatility', 'momentum', 
                'trend', 'pattern', 'microstructure'
            ]
        
        results = {}
        
        # Process each symbol
        for symbol, timeframe_data in data.items():
            try:
                # Check cache first if not forcing recalculation
                if not force_recalculate and self.use_redis_cache:
                    cache_key = f"multi_timeframe_features:{symbol}"
                    cache_params = {
                        "symbol": symbol,
                        "feature_groups": feature_groups,
                        "include_target": include_target,
                        "target_horizon": target_horizon
                    }
                    
                    cached_data = self.feature_store_cache.get_feature(cache_key, cache_params)
                    if cached_data is not None:
                        logger.info(f"Cache hit for {symbol} multi-timeframe features")
                        results[symbol] = cached_data
                        continue
                
                # Process timeframes in parallel or sequentially
                if self.parallel_processing and len(timeframe_data) > 1:
                    symbol_results = self._process_timeframes_parallel(
                        symbol, timeframe_data, feature_groups, include_target, target_horizon
                    )
                else:
                    symbol_results = self._process_timeframes_sequential(
                        symbol, timeframe_data, feature_groups, include_target, target_horizon
                    )
                
                # Perform cross-timeframe consistency checks
                symbol_results = self._ensure_cross_timeframe_consistency(symbol, symbol_results)
                
                # Cache results
                if self.use_redis_cache:
                    cache_key = f"multi_timeframe_features:{symbol}"
                    cache_params = {
                        "symbol": symbol,
                        "feature_groups": feature_groups,
                        "include_target": include_target,
                        "target_horizon": target_horizon
                    }
                    
                    self.feature_store_cache.set_feature(
                        cache_key, cache_params, symbol_results, self.redis_cache_ttl
                    )
                    logger.debug(f"Cached multi-timeframe features for {symbol}")
                
                results[symbol] = symbol_results
                
            except Exception as e:
                logger.error(f"Error processing multi-timeframe data for {symbol}: {e}")
                logger.debug(traceback.format_exc())
        
        logger.info(f"Processed multi-timeframe data for {len(results)} symbols")
        return results
    
    def _process_timeframes_parallel(
        self,
        symbol: str,
        timeframe_data: Dict[str, pd.DataFrame],
        feature_groups: List[str],
        include_target: bool,
        target_horizon: int
    ) -> Dict[str, pd.DataFrame]:
        """
        Process timeframes in parallel.
        
        Args:
            symbol: Ticker symbol
            timeframe_data: Dictionary mapping timeframes to DataFrames
            feature_groups: List of feature groups to calculate
            include_target: Whether to include target variables
            target_horizon: Horizon for target variable calculation
            
        Returns:
            Dictionary mapping timeframes to DataFrames with features
        """
        results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit tasks
            future_to_timeframe = {
                executor.submit(
                    self._process_single_timeframe,
                    symbol, timeframe, df, feature_groups, include_target, target_horizon
                ): timeframe 
                for timeframe, df in timeframe_data.items()
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_timeframe):
                timeframe = future_to_timeframe[future]
                try:
                    processed_df = future.result()
                    results[timeframe] = processed_df
                except Exception as e:
                    logger.error(f"Error processing {symbol} {timeframe}: {e}")
                    logger.debug(traceback.format_exc())
        
        return results
    
    def _process_timeframes_sequential(
        self,
        symbol: str,
        timeframe_data: Dict[str, pd.DataFrame],
        feature_groups: List[str],
        include_target: bool,
        target_horizon: int
    ) -> Dict[str, pd.DataFrame]:
        """
        Process timeframes sequentially.
        
        Args:
            symbol: Ticker symbol
            timeframe_data: Dictionary mapping timeframes to DataFrames
            feature_groups: List of feature groups to calculate
            include_target: Whether to include target variables
            target_horizon: Horizon for target variable calculation
            
        Returns:
            Dictionary mapping timeframes to DataFrames with features
        """
        results = {}
        
        for timeframe, df in timeframe_data.items():
            try:
                processed_df = self._process_single_timeframe(
                    symbol, timeframe, df, feature_groups, include_target, target_horizon
                )
                results[timeframe] = processed_df
            except Exception as e:
                logger.error(f"Error processing {symbol} {timeframe}: {e}")
                logger.debug(traceback.format_exc())
        
        return results
    
    def _process_single_timeframe(
        self,
        symbol: str,
        timeframe: str,
        df: pd.DataFrame,
        feature_groups: List[str],
        include_target: bool,
        target_horizon: int
    ) -> pd.DataFrame:
        """
        Process a single timeframe to generate features.
        
        Args:
            symbol: Ticker symbol
            timeframe: Timeframe (e.g., '1m', '5m', '1h', '1d')
            df: DataFrame with OHLCV data
            feature_groups: List of feature groups to calculate
            include_target: Whether to include target variables
            target_horizon: Horizon for target variable calculation
            
        Returns:
            DataFrame with features
        """
        logger.debug(f"Processing {symbol} {timeframe}")
        
        # Make a copy to avoid modifying the original
        df = df.copy()
        
        # Calculate features for each group
        for group in feature_groups:
            try:
                if group == 'price':
                    df = self.price_calculator.calculate_features(df)
                elif group == 'volume':
                    df = self.volume_calculator.calculate_features(df)
                elif group == 'volatility':
                    df = self.volatility_calculator.calculate_features(df)
                elif group == 'momentum':
                    df = self.momentum_calculator.calculate_features(df)
                elif group == 'trend':
                    df = self.trend_calculator.calculate_features(df)
                elif group == 'pattern':
                    df = self.pattern_calculator.calculate_features(df)
                elif group == 'microstructure':
                    df = self.microstructure_calculator.calculate_features(df, symbol=symbol)
            except Exception as e:
                logger.error(f"Error calculating {group} features for {symbol} {timeframe}: {e}")
                logger.debug(traceback.format_exc())
        
        # Calculate target variables if requested
        if include_target:
            try:
                df = self._calculate_target_variables(df, target_horizon)
            except Exception as e:
                logger.error(f"Error calculating target variables for {symbol} {timeframe}: {e}")
                logger.debug(traceback.format_exc())
        
        # Apply feature transformations
        try:
            df = self._apply_feature_transformations(df)
        except Exception as e:
            logger.error(f"Error applying feature transformations for {symbol} {timeframe}: {e}")
            logger.debug(traceback.format_exc())
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        logger.debug(f"Processed {symbol} {timeframe} with {len(df.columns)} features")
        return df
    
    def _calculate_target_variables(self, df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """
        Calculate target variables for model training.
        
        Args:
            df: DataFrame with OHLCV data
            horizon: Horizon for target variable calculation (in bars)
            
        Returns:
            DataFrame with target variables
        """
        # Calculate future returns
        df['future_return'] = df['close'].pct_change(horizon).shift(-horizon)
        
        # Calculate binary target (1 for positive return, 0 for negative return)
        df['target_binary'] = np.where(df['future_return'] > 0, 1, 0)
        
        # Calculate multi-class target (2 for strong positive, 1 for weak positive, 
        # 0 for neutral, -1 for weak negative, -2 for strong negative)
        df['target_multiclass'] = pd.cut(
            df['future_return'],
            bins=[-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf],
            labels=[-2, -1, 0, 1, 2]
        ).astype(int)
        
        # Calculate regression target (future return)
        df['target_regression'] = df['future_return']
        
        return df
    
    def _apply_feature_transformations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply transformations to features.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with transformed features
        """
        # Get numeric columns (excluding target variables)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        target_cols = [col for col in numeric_cols if col.startswith('target_') or col == 'future_return']
        numeric_cols = [col for col in numeric_cols if col not in target_cols]
        
        # Apply log transformation to highly skewed features
        for col in numeric_cols:
            # Skip columns with non-positive values
            if (df[col] <= 0).any():
                continue
            
            # Calculate skewness
            skewness = df[col].skew()
            
            # Apply log transformation to highly skewed features
            if abs(skewness) > 2:
                df[f"{col}_log"] = np.log(df[col])
        
        # Apply normalization to numeric features
        for col in numeric_cols:
            # Calculate z-score
            mean = df[col].mean()
            std = df[col].std()
            
            # Skip if standard deviation is zero
            if std == 0:
                continue
            
            df[f"{col}_zscore"] = (df[col] - mean) / std
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in features.
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with handled missing values
        """
        # Forward fill missing values
        df = df.fillna(method='ffill')
        
        # Backward fill any remaining missing values
        df = df.fillna(method='bfill')
        
        # Fill any still-missing values with zeros
        df = df.fillna(0)
        
        return df
    
    def _ensure_cross_timeframe_consistency(
        self,
        symbol: str,
        timeframe_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        """
        Ensure consistency between features across different timeframes.
        
        Args:
            symbol: Ticker symbol
            timeframe_data: Dictionary mapping timeframes to DataFrames with features
            
        Returns:
            Dictionary mapping timeframes to DataFrames with consistent features
        """
        logger.debug(f"Ensuring cross-timeframe consistency for {symbol}")
        
        # Sort timeframes by granularity (smallest to largest)
        timeframes = sorted(
            timeframe_data.keys(),
            key=lambda tf: self._timeframe_to_minutes(tf)
        )
        
        # If only one timeframe, no consistency checks needed
        if len(timeframes) <= 1:
            return timeframe_data
        
        # Get the smallest timeframe as reference
        smallest_tf = timeframes[0]
        smallest_df = timeframe_data[smallest_tf]
        
        # Check consistency for each larger timeframe
        for tf in timeframes[1:]:
            try:
                # Get the current timeframe data
                df = timeframe_data[tf]
                
                # Resample the smallest timeframe to match the current timeframe
                resampled_df = self._resample_dataframe(smallest_df, smallest_tf, tf)
                
                # Check consistency between resampled and actual data
                consistency_issues = self._check_consistency(resampled_df, df)
                
                if consistency_issues:
                    logger.warning(f"Cross-timeframe consistency issues for {symbol} {tf}: {consistency_issues}")
                    
                    # Adjust the current timeframe data based on resampled data
                    timeframe_data[tf] = self._adjust_for_consistency(df, resampled_df)
                    
            except Exception as e:
                logger.error(f"Error ensuring cross-timeframe consistency for {symbol} {tf}: {e}")
                logger.debug(traceback.format_exc())
        
        return timeframe_data
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """
        Convert timeframe string to minutes.
        
        Args:
            timeframe: Timeframe string (e.g., '1m', '5m', '1h', '1d')
            
        Returns:
            Number of minutes in the timeframe
        """
        if timeframe.endswith('m'):
            return int(timeframe[:-1])
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]) * 60
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]) * 60 * 24
        else:
            raise ValueError(f"Invalid timeframe: {timeframe}")
    
    def _resample_dataframe(
        self,
        df: pd.DataFrame,
        source_tf: str,
        target_tf: str
    ) -> pd.DataFrame:
        """
        Resample a DataFrame from one timeframe to another.
        
        Args:
            df: DataFrame to resample
            source_tf: Source timeframe
            target_tf: Target timeframe
            
        Returns:
            Resampled DataFrame
        """
        # Ensure the index is a datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            else:
                raise ValueError("DataFrame must have a timestamp column or datetime index")
        
        # Convert timeframes to pandas frequency strings
        source_freq = self._timeframe_to_freq(source_tf)
        target_freq = self._timeframe_to_freq(target_tf)
        
        # Resample OHLCV data
        resampled = df.resample(target_freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        return resampled
    
    def _timeframe_to_freq(self, timeframe: str) -> str:
        """
        Convert timeframe string to pandas frequency string.
        
        Args:
            timeframe: Timeframe string (e.g., '1m', '5m', '1h', '1d')
            
        Returns:
            Pandas frequency string
        """
        if timeframe.endswith('m'):
            return f"{timeframe[:-1]}T"
        elif timeframe.endswith('h'):
            return f"{timeframe[:-1]}H"
        elif timeframe.endswith('d'):
            return f"{timeframe[:-1]}D"
        else:
            raise ValueError(f"Invalid timeframe: {timeframe}")
    
    def _check_consistency(self, resampled_df: pd.DataFrame, actual_df: pd.DataFrame) -> List[str]:
        """
        Check consistency between resampled and actual data.
        
        Args:
            resampled_df: Resampled DataFrame
            actual_df: Actual DataFrame
            
        Returns:
            List of consistency issues
        """
        issues = []
        
        # Align indices
        resampled_df, actual_df = resampled_df.align(actual_df, join='inner')
        
        # Check OHLCV consistency
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in resampled_df.columns or col not in actual_df.columns:
                continue
                
            # Calculate relative difference
            rel_diff = np.abs(resampled_df[col] - actual_df[col]) / np.maximum(np.abs(resampled_df[col]), 1e-10)
            
            # Check for large differences
            large_diff = rel_diff > 0.05  # 5% threshold
            if large_diff.any():
                count = large_diff.sum()
                if count > 0:
                    issues.append(f"{col}: {count} large differences")
        
        return issues
    
    def _adjust_for_consistency(self, df: pd.DataFrame, resampled_df: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust DataFrame for consistency with resampled data.
        
        Args:
            df: DataFrame to adjust
            resampled_df: Resampled DataFrame to use as reference
            
        Returns:
            Adjusted DataFrame
        """
        # Align indices
        resampled_df, df = resampled_df.align(df, join='inner')
        
        # Create a copy to avoid modifying the original
        adjusted_df = df.copy()
        
        # Adjust OHLCV values
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in resampled_df.columns or col not in df.columns:
                continue
                
            # Calculate relative difference
            rel_diff = np.abs(resampled_df[col] - df[col]) / np.maximum(np.abs(resampled_df[col]), 1e-10)
            
            # Adjust values with large differences
            large_diff = rel_diff > 0.05  # 5% threshold
            if large_diff.any():
                # Use a weighted average of resampled and actual values
                weight = 0.7  # 70% weight to resampled values
                adjusted_df.loc[large_diff, col] = (
                    weight * resampled_df.loc[large_diff, col] + 
                    (1 - weight) * df.loc[large_diff, col]
                )
        
        return adjusted_df