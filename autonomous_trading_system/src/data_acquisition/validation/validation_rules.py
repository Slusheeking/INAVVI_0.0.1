"""
Validation Rules

This module provides predefined validation rule sets for different types of financial data.
These rules can be imported and used by the DataValidator class.
"""

from typing import Dict, Any, List, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Default validation rule sets for different data types
# These can be imported and used directly or customized as needed

# OHLCV (Stock Aggregates) Validation Rules
STOCK_AGGS_RULES = {
    'strict_mode': False,
    'use_adaptive_thresholds': True,
    'volatility_lookback_periods': 20,
    'volatility_multiplier': 3.0,
    'max_price_change_pct': 20.0,
    'min_price': 0.01,
    'max_volume_change_pct': 1000.0,
    'max_gap_seconds': {
        '1m': 120,    # 2 minutes
        '5m': 600,    # 10 minutes
        '15m': 1800,  # 30 minutes
        '1h': 7200,   # 2 hours
        '1d': 172800  # 2 days
    },
    'interpolate_gaps': True,
    'required_columns': ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
}

# Quote Data Validation Rules
QUOTE_RULES = {
    'strict_mode': False,
    'min_price': 0.01,
    'max_spread_pct': 10.0,  # Maximum bid-ask spread as percentage of mid price
    'min_size': 1,
    'required_columns': ['timestamp', 'symbol', 'bid_price', 'ask_price', 'bid_size', 'ask_size']
}

# Trade Data Validation Rules
TRADE_RULES = {
    'strict_mode': False,
    'min_price': 0.01,
    'max_price_change_pct': 20.0,
    'min_size': 1,
    'required_columns': ['timestamp', 'symbol', 'price', 'size', 'exchange']
}

# Options Data Validation Rules
OPTIONS_RULES = {
    'strict_mode': False,
    'min_price': 0.01,
    'max_price_change_pct': 100.0,  # Options can be more volatile
    'min_size': 1,
    'valid_option_types': ['call', 'put', 'C', 'P'],
    'required_columns': ['timestamp', 'symbol', 'underlying', 'expiration', 'strike', 'option_type']
}

# News Sentiment Validation Rules
NEWS_SENTIMENT_RULES = {
    'strict_mode': False,
    'sentiment_score_range': (-1.0, 1.0),
    'required_columns': ['timestamp', 'article_id', 'sentiment_score', 'sentiment_label']
}

# Feature Data Validation Rules
FEATURE_RULES = {
    'strict_mode': False,
    'max_value_change_pct': 100.0,
    'required_columns': ['timestamp', 'symbol', 'feature_name', 'feature_value']
}

# Model Prediction Validation Rules
MODEL_PREDICTION_RULES = {
    'strict_mode': False,
    'prediction_range': (-1.0, 1.0),  # For normalized predictions
    'confidence_range': (0.0, 1.0),
    'required_columns': ['timestamp', 'symbol', 'model_id', 'prediction', 'confidence']
}

# Trading Signal Validation Rules
TRADING_SIGNAL_RULES = {
    'strict_mode': False,
    'valid_signal_types': ['buy', 'sell', 'hold', 'strong_buy', 'strong_sell'],
    'confidence_range': (0.0, 1.0),
    'required_columns': ['timestamp', 'symbol', 'signal_type', 'confidence', 'model_id']
}

# System Metrics Validation Rules
SYSTEM_METRICS_RULES = {
    'strict_mode': False,
    'required_columns': ['timestamp', 'metric_name', 'metric_value', 'component']
}

# Adaptive Validation Rule Functions

def get_adaptive_price_threshold(df: pd.DataFrame, base_threshold: float = 20.0, 
                               lookback_periods: int = 20, multiplier: float = 3.0) -> float:
    """
    Calculate adaptive price threshold based on recent volatility.
    
    Args:
        df: DataFrame with price data
        base_threshold: Base threshold percentage
        lookback_periods: Number of periods to look back for volatility calculation
        multiplier: Volatility multiplier
        
    Returns:
        Adjusted threshold value
    """
    if len(df) < lookback_periods or 'close' not in df.columns:
        return base_threshold
    
    # Calculate recent volatility (standard deviation of returns)
    returns = df['close'].pct_change().dropna()
    if len(returns) < lookback_periods:
        return base_threshold
        
    recent_volatility = returns.tail(lookback_periods).std() * 100
    
    # Adjust threshold based on volatility
    adjusted_threshold = base_threshold * (1 + recent_volatility * multiplier)
    
    return adjusted_threshold

def get_adaptive_volume_threshold(df: pd.DataFrame, base_threshold: float = 1000.0,
                                lookback_periods: int = 20, multiplier: float = 2.0) -> float:
    """
    Calculate adaptive volume threshold based on recent volume volatility.
    
    Args:
        df: DataFrame with volume data
        base_threshold: Base threshold percentage
        lookback_periods: Number of periods to look back for volatility calculation
        multiplier: Volatility multiplier
        
    Returns:
        Adjusted threshold value
    """
    if len(df) < lookback_periods or 'volume' not in df.columns:
        return base_threshold
    
    # Calculate recent volume volatility (standard deviation of volume changes)
    volume_changes = df['volume'].pct_change().abs().dropna() * 100
    if len(volume_changes) < lookback_periods:
        return base_threshold
        
    recent_volatility = volume_changes.tail(lookback_periods).std()
    
    # Adjust threshold based on volatility
    adjusted_threshold = base_threshold * (1 + recent_volatility * multiplier)
    
    return adjusted_threshold

# Market-Specific Validation Rules

# Equity Market Rules
EQUITY_MARKET_RULES = {
    'market_hours': {
        'pre_market_start': '04:00',  # Eastern Time
        'market_open': '09:30',
        'market_close': '16:00',
        'post_market_end': '20:00'
    },
    'price_limits': {
        'circuit_breaker_level_1': 7.0,  # 7% decline from previous close
        'circuit_breaker_level_2': 13.0,  # 13% decline
        'circuit_breaker_level_3': 20.0   # 20% decline
    },
    'tick_sizes': {
        'below_1': 0.0001,  # Sub-penny for stocks under $1
        'above_1': 0.01     # Penny for stocks above $1
    }
}

# Crypto Market Rules
CRYPTO_MARKET_RULES = {
    'market_hours': {
        'is_24h': True
    },
    'price_precision': {
        'BTC': 2,    # 2 decimal places
        'ETH': 2,
        'default': 6  # 6 decimal places for other coins
    },
    'min_order_size': {
        'BTC': 0.0001,
        'ETH': 0.001,
        'default': 1.0
    }
}

# Options Market Rules
OPTIONS_MARKET_RULES = {
    'market_hours': {
        'market_open': '09:30',  # Eastern Time
        'market_close': '16:00'
    },
    'price_limits': {
        'max_daily_change': 100.0  # Options can move significantly
    },
    'tick_sizes': {
        'below_3': 0.01,   # $0.01 for options under $3
        'above_3': 0.05    # $0.05 for options above $3
    }
}

# Forex Market Rules
FOREX_MARKET_RULES = {
    'market_hours': {
        'sunday_open': '17:00',  # Eastern Time, Sunday
        'friday_close': '17:00'  # Eastern Time, Friday
    },
    'price_precision': {
        'major_pairs': 4,    # 4 decimal places (e.g., EUR/USD)
        'jpy_pairs': 2,      # 2 decimal places for JPY pairs
        'exotic_pairs': 5    # 5 decimal places for exotic pairs
    }
}

# Validation Rule Factories

def create_stock_validation_rules(market_cap_category: str = 'mid', 
                                volatility_category: str = 'medium') -> Dict[str, Any]:
    """
    Create validation rules tailored to stock characteristics.
    
    Args:
        market_cap_category: 'large', 'mid', or 'small'
        volatility_category: 'low', 'medium', or 'high'
        
    Returns:
        Dictionary with validation rules
    """
    rules = STOCK_AGGS_RULES.copy()
    
    # Adjust thresholds based on market cap
    if market_cap_category == 'large':
        rules['max_price_change_pct'] = 10.0
        rules['max_volume_change_pct'] = 500.0
    elif market_cap_category == 'mid':
        rules['max_price_change_pct'] = 15.0
        rules['max_volume_change_pct'] = 750.0
    elif market_cap_category == 'small':
        rules['max_price_change_pct'] = 25.0
        rules['max_volume_change_pct'] = 1500.0
    
    # Adjust thresholds based on volatility
    if volatility_category == 'low':
        rules['volatility_multiplier'] = 2.0
    elif volatility_category == 'medium':
        rules['volatility_multiplier'] = 3.0
    elif volatility_category == 'high':
        rules['volatility_multiplier'] = 4.0
    
    return rules

def create_crypto_validation_rules(liquidity_category: str = 'medium',
                                 volatility_category: str = 'high') -> Dict[str, Any]:
    """
    Create validation rules tailored to cryptocurrency characteristics.
    
    Args:
        liquidity_category: 'high', 'medium', or 'low'
        volatility_category: 'low', 'medium', or 'high'
        
    Returns:
        Dictionary with validation rules
    """
    rules = STOCK_AGGS_RULES.copy()
    
    # Crypto is more volatile by default
    rules['max_price_change_pct'] = 30.0
    rules['max_volume_change_pct'] = 2000.0
    
    # Adjust thresholds based on liquidity
    if liquidity_category == 'high':
        rules['max_gap_seconds'] = {
            '1m': 60,      # 1 minute
            '5m': 300,     # 5 minutes
            '15m': 900,    # 15 minutes
            '1h': 3600,    # 1 hour
            '1d': 86400    # 1 day
        }
    elif liquidity_category == 'medium':
        rules['max_gap_seconds'] = {
            '1m': 120,     # 2 minutes
            '5m': 600,     # 10 minutes
            '15m': 1800,   # 30 minutes
            '1h': 7200,    # 2 hours
            '1d': 86400    # 1 day
        }
    elif liquidity_category == 'low':
        rules['max_gap_seconds'] = {
            '1m': 300,     # 5 minutes
            '5m': 1200,    # 20 minutes
            '15m': 3600,   # 1 hour
            '1h': 14400,   # 4 hours
            '1d': 86400    # 1 day
        }
    
    # Adjust thresholds based on volatility
    if volatility_category == 'low':
        rules['max_price_change_pct'] = 20.0
        rules['volatility_multiplier'] = 2.5
    elif volatility_category == 'medium':
        rules['max_price_change_pct'] = 30.0
        rules['volatility_multiplier'] = 3.5
    elif volatility_category == 'high':
        rules['max_price_change_pct'] = 50.0
        rules['volatility_multiplier'] = 5.0
    
    return rules

def create_options_validation_rules(days_to_expiration: int = 30,
                                  moneyness: str = 'atm') -> Dict[str, Any]:
    """
    Create validation rules tailored to options characteristics.
    
    Args:
        days_to_expiration: Number of days to expiration
        moneyness: 'itm' (in-the-money), 'atm' (at-the-money), or 'otm' (out-of-the-money)
        
    Returns:
        Dictionary with validation rules
    """
    rules = OPTIONS_RULES.copy()
    
    # Adjust thresholds based on days to expiration
    if days_to_expiration <= 7:  # Weekly options
        rules['max_price_change_pct'] = 200.0  # Very volatile
    elif days_to_expiration <= 30:  # Monthly options
        rules['max_price_change_pct'] = 100.0
    elif days_to_expiration <= 90:  # Quarterly options
        rules['max_price_change_pct'] = 75.0
    else:  # LEAPS or long-dated options
        rules['max_price_change_pct'] = 50.0
    
    # Adjust thresholds based on moneyness
    if moneyness == 'itm':
        rules['min_price'] = 0.5  # ITM options typically have higher prices
    elif moneyness == 'atm':
        rules['min_price'] = 0.1
    elif moneyness == 'otm':
        rules['min_price'] = 0.01  # OTM options can be very cheap
    
    return rules

# Validation Rule Combiners

def combine_validation_rules(*rule_sets: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combine multiple validation rule sets, with later rules overriding earlier ones.
    
    Args:
        *rule_sets: Variable number of rule set dictionaries
        
    Returns:
        Combined rule set
    """
    combined_rules = {}
    
    for rules in rule_sets:
        for key, value in rules.items():
            if isinstance(value, dict) and key in combined_rules and isinstance(combined_rules[key], dict):
                # Recursively combine nested dictionaries
                combined_rules[key] = {**combined_rules[key], **value}
            else:
                # Override or add the value
                combined_rules[key] = value
    
    return combined_rules

# Validation Rule Exporters

def export_validation_rules_to_json(rules: Dict[str, Any], filepath: str) -> None:
    """
    Export validation rules to a JSON file.
    
    Args:
        rules: Validation rules dictionary
        filepath: Path to save the JSON file
    """
    import json
    
    # Convert any non-serializable objects to strings
    def serialize(obj):
        if isinstance(obj, (datetime, np.datetime64)):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return str(obj)
    
    # Serialize the rules
    serialized_rules = {}
    for key, value in rules.items():
        if isinstance(value, dict):
            serialized_rules[key] = {k: serialize(v) for k, v in value.items()}
        else:
            serialized_rules[key] = serialize(value)
    
    # Write to file
    with open(filepath, 'w') as f:
        json.dump(serialized_rules, f, indent=4)

def import_validation_rules_from_json(filepath: str) -> Dict[str, Any]:
    """
    Import validation rules from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        Dictionary with validation rules
    """
    import json
    
    with open(filepath, 'r') as f:
        rules = json.load(f)
    
    # Parse any special values
    for key, value in rules.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if k.endswith('_seconds') and isinstance(v, dict):
                    # Convert string keys back to integers for max_gap_seconds
                    rules[key][k] = {int(sk) if sk.isdigit() else sk: sv for sk, sv in v.items()}
    
    return rules

# Predefined Rule Sets for Common Use Cases

# High-frequency trading validation rules
HFT_VALIDATION_RULES = {
    'strict_mode': True,  # Strict validation for HFT
    'use_adaptive_thresholds': True,
    'volatility_lookback_periods': 10,  # Shorter lookback for faster adaptation
    'volatility_multiplier': 4.0,  # More sensitive to volatility
    'max_price_change_pct': 5.0,  # Smaller price changes expected in short timeframes
    'min_price': 0.01,
    'max_volume_change_pct': 500.0,
    'max_gap_seconds': {
        '1m': 60,     # 1 minute
        '5m': 300,    # 5 minutes
        '15m': 900,   # 15 minutes
        '1h': 3600,   # 1 hour
        '1d': 86400   # 1 day
    },
    'interpolate_gaps': False  # Don't interpolate for HFT
}

# Day trading validation rules
DAY_TRADING_VALIDATION_RULES = {
    'strict_mode': False,
    'use_adaptive_thresholds': True,
    'volatility_lookback_periods': 20,
    'volatility_multiplier': 3.0,
    'max_price_change_pct': 10.0,
    'min_price': 0.01,
    'max_volume_change_pct': 1000.0,
    'max_gap_seconds': {
        '1m': 120,    # 2 minutes
        '5m': 600,    # 10 minutes
        '15m': 1800,  # 30 minutes
        '1h': 7200,   # 2 hours
        '1d': 86400   # 1 day
    },
    'interpolate_gaps': True
}

# Swing trading validation rules
SWING_TRADING_VALIDATION_RULES = {
    'strict_mode': False,
    'use_adaptive_thresholds': True,
    'volatility_lookback_periods': 50,  # Longer lookback for swing trading
    'volatility_multiplier': 2.5,
    'max_price_change_pct': 15.0,
    'min_price': 0.01,
    'max_volume_change_pct': 1500.0,
    'max_gap_seconds': {
        '1m': 300,     # 5 minutes
        '5m': 1200,    # 20 minutes
        '15m': 3600,   # 1 hour
        '1h': 14400,   # 4 hours
        '1d': 172800   # 2 days
    },
    'interpolate_gaps': True
}

# Long-term investing validation rules
LONG_TERM_VALIDATION_RULES = {
    'strict_mode': False,
    'use_adaptive_thresholds': True,
    'volatility_lookback_periods': 100,  # Very long lookback
    'volatility_multiplier': 2.0,
    'max_price_change_pct': 20.0,
    'min_price': 0.01,
    'max_volume_change_pct': 2000.0,
    'max_gap_seconds': {
        '1m': 600,     # 10 minutes
        '5m': 1800,    # 30 minutes
        '15m': 7200,   # 2 hours
        '1h': 28800,   # 8 hours
        '1d': 259200   # 3 days
    },
    'interpolate_gaps': True
}

# Market making validation rules
MARKET_MAKING_VALIDATION_RULES = {
    'strict_mode': True,  # Strict validation for market making
    'use_adaptive_thresholds': True,
    'volatility_lookback_periods': 5,  # Very short lookback
    'volatility_multiplier': 5.0,  # Very sensitive to volatility
    'max_price_change_pct': 3.0,  # Small price changes expected
    'min_price': 0.01,
    'max_volume_change_pct': 300.0,
    'max_gap_seconds': {
        '1m': 30,      # 30 seconds
        '5m': 150,     # 2.5 minutes
        '15m': 450,    # 7.5 minutes
        '1h': 1800,    # 30 minutes
        '1d': 43200    # 12 hours
    },
    'interpolate_gaps': False  # Don't interpolate for market making
}

# Microstructure Validation Functions

def validate_order_book_imbalance(bid_sizes: List[float], ask_sizes: List[float], 
                                 bid_prices: List[float], ask_prices: List[float],
                                 volatility_regime: str = 'normal') -> Dict[str, Any]:
    """
    Validate order book imbalance.
    
    Args:
        bid_sizes: List of bid sizes
        ask_sizes: List of ask sizes
        bid_prices: List of bid prices
        ask_prices: List of ask prices
        volatility_regime: Current volatility regime ('low', 'normal', 'high')
        
    Returns:
        Dictionary with validation results
    """
    if not bid_sizes or not ask_sizes or not bid_prices or not ask_prices:
        return {
            'valid': False,
            'error': 'Empty order book data'
        }
    
    # Check for negative sizes or prices
    if any(size < 0 for size in bid_sizes) or any(size < 0 for size in ask_sizes):
        return {
            'valid': False,
            'error': 'Negative sizes in order book'
        }
    
    if any(price < 0 for price in bid_prices) or any(price < 0 for price in ask_prices):
        return {
            'valid': False,
            'error': 'Negative prices in order book'
        }
    
    # Calculate order book imbalance
    total_bid_value = sum(bid_sizes[i] * bid_prices[i] for i in range(min(len(bid_sizes), len(bid_prices))))
    total_ask_value = sum(ask_sizes[i] * ask_prices[i] for i in range(min(len(ask_sizes), len(ask_prices))))
    
    if total_bid_value + total_ask_value == 0:
        return {
            'valid': False,
            'error': 'Zero total order book value'
        }
    
    imbalance = (total_bid_value - total_ask_value) / (total_bid_value + total_ask_value)
    
    # Set threshold based on volatility regime
    if volatility_regime == 'low':
        threshold = 0.7  # 70% imbalance for low volatility
    elif volatility_regime == 'high':
        threshold = 0.9  # 90% imbalance for high volatility
    else:  # normal
        threshold = 0.8  # 80% imbalance for normal volatility
    
    return {
        'valid': abs(imbalance) <= threshold,
        'imbalance': imbalance,
        'threshold': threshold,
        'error': f'Order book imbalance exceeds threshold: {abs(imbalance):.2f} > {threshold:.2f}' if abs(imbalance) > threshold else None
    }

def validate_trade_flow_imbalance(buy_volume: float, sell_volume: float,
                                volatility_regime: str = 'normal') -> Dict[str, Any]:
    """
    Validate trade flow imbalance.
    
    Args:
        buy_volume: Buy volume
        sell_volume: Sell volume
        volatility_regime: Current volatility regime ('low', 'normal', 'high')
        
    Returns:
        Dictionary with validation results
    """
    if buy_volume < 0 or sell_volume < 0:
        return {
            'valid': False,
            'error': 'Negative volume in trade flow'
        }
    
    if buy_volume + sell_volume == 0:
        return {
            'valid': False,
            'error': 'Zero total trade volume'
        }
    
    imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume)
    
    # Set threshold based on volatility regime
    if volatility_regime == 'low':
        threshold = 0.6  # 60% imbalance for low volatility
    elif volatility_regime == 'high':
        threshold = 0.8  # 80% imbalance for high volatility
    else:  # normal
        threshold = 0.7  # 70% imbalance for normal volatility
    
    return {
        'valid': abs(imbalance) <= threshold,
        'imbalance': imbalance,
        'threshold': threshold,
        'error': f'Trade flow imbalance exceeds threshold: {abs(imbalance):.2f} > {threshold:.2f}' if abs(imbalance) > threshold else None
    }

def calculate_volume_weighted_price_pressure(bid_sizes: List[float], ask_sizes: List[float], 
                                           bid_prices: List[float], ask_prices: List[float]) -> float:
    """
    Calculate volume-weighted price pressure.
    
    Args:
        bid_sizes: List of bid sizes
        ask_sizes: List of ask sizes
        bid_prices: List of bid prices
        ask_prices: List of ask prices
        
    Returns:
        Volume-weighted price pressure value
    """
    if not bid_sizes or not ask_sizes or not bid_prices or not ask_prices:
        return 0.0
    
    # Calculate weighted average bid and ask prices
    total_bid_size = sum(bid_sizes)
    total_ask_size = sum(ask_sizes)
    
    if total_bid_size == 0 or total_ask_size == 0:
        return 0.0
    
    weighted_bid_price = sum(bid_sizes[i] * bid_prices[i] for i in range(min(len(bid_sizes), len(bid_prices)))) / total_bid_size
    weighted_ask_price = sum(ask_sizes[i] * ask_prices[i] for i in range(min(len(ask_sizes), len(ask_prices)))) / total_ask_size
    
    # Calculate mid price
    mid_price = (weighted_bid_price + weighted_ask_price) / 2
    
    # Calculate volume-weighted price pressure
    vwpp = ((weighted_bid_price * total_bid_size) - (weighted_ask_price * total_ask_size)) / (total_bid_size + total_ask_size)
    
    # Normalize by mid price
    if mid_price > 0:
        vwpp = vwpp / mid_price
    
    return vwpp

def calculate_relative_strength(current_price: float, reference_prices: List[float], 
                              reference_volumes: List[float] = None) -> float:
    """
    Calculate relative strength of a symbol compared to reference symbols.
    
    Args:
        current_price: Current price of the symbol
        reference_prices: List of prices for reference symbols (e.g., sector ETF, market index)
        reference_volumes: Optional list of volumes for reference symbols for volume-weighting
        
    Returns:
        Relative strength value
    """
    if not reference_prices or len(reference_prices) == 0:
        return 0.0
    
    # If no reference volumes provided, use equal weighting
    if reference_volumes is None or len(reference_volumes) != len(reference_prices):
        reference_volumes = [1.0] * len(reference_prices)
    
    # Calculate weighted average of reference prices
    total_volume = sum(reference_volumes)
    if total_volume == 0:
        return 0.0
    
    weighted_reference_price = sum(reference_prices[i] * reference_volumes[i] for i in range(len(reference_prices))) / total_volume
    
    # Calculate relative strength
    if weighted_reference_price == 0:
        return 0.0
    
    relative_strength = (current_price / weighted_reference_price) - 1.0
    
    return relative_strength