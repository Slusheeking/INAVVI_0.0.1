"""
Data Schema Module

This module defines the schema for market data in the Autonomous Trading System.
It provides data classes and validation functions for different types of market data.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Set
from datetime import datetime, date
import pandas as pd

# Stock/Crypto Aggregate (OHLCV) Schema
@dataclass
class AggregateSchema:
    """Schema for stock/crypto aggregate (OHLCV) data."""
    
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: Optional[float] = None
    transactions: Optional[int] = None
    timeframe: str = "1m"  # e.g., "1m", "5m", "1h", "1d"
    source: str = "polygon"
    multiplier: int = 1
    timespan_unit: str = "minute"
    adjusted: bool = False
    
    @classmethod
    def from_polygon(cls, data: Dict[str, Any], symbol: str, timeframe: str) -> "AggregateSchema":
        """
        Create an AggregateSchema instance from Polygon API data.
        
        Args:
            data: Dictionary with Polygon API data
            symbol: Ticker symbol
            timeframe: Data timeframe
            
        Returns:
            AggregateSchema instance
        """
        multiplier, timespan_unit = cls._parse_timeframe(timeframe)
        
        return cls(
            timestamp=datetime.fromtimestamp(data['t'] / 1000),
            symbol=symbol,
            open=data['o'],
            high=data['h'],
            low=data['l'],
            close=data['c'],
            volume=data['v'],
            vwap=data.get('vw'),
            transactions=data.get('n'),
            timeframe=timeframe,
            source="polygon",
            multiplier=multiplier,
            timespan_unit=timespan_unit
        )
    
    @classmethod
    def from_alpaca(cls, data: Dict[str, Any], symbol: str, timeframe: str) -> "AggregateSchema":
        """
        Create an AggregateSchema instance from Alpaca API data.
        Note: Alpaca free tier has limitations on historical data.
        
        Args:
            data: Dictionary with Alpaca API data
            symbol: Ticker symbol
            timeframe: Data timeframe
            
        Returns:
            AggregateSchema instance
        """
        multiplier, timespan_unit = cls._parse_timeframe(timeframe)
        
        return cls(
            timestamp=pd.to_datetime(data['t']),
            symbol=symbol,
            open=data['o'],
            high=data['h'],
            low=data['l'],
            close=data['c'],
            volume=data['v'],
            vwap=data.get('vw'),
            timeframe=timeframe,
            source="alpaca",
            multiplier=multiplier,
            timespan_unit=timespan_unit
        )
    
    @staticmethod
    def _parse_timeframe(timeframe: str) -> tuple:
        """
        Parse timeframe string into multiplier and timespan unit.
        
        Args:
            timeframe: Timeframe string (e.g., "1m", "5m", "1h", "1d")
            
        Returns:
            Tuple of (multiplier, timespan_unit)
        """
        if timeframe.endswith('m'):
            return int(timeframe[:-1]), "minute"
        elif timeframe.endswith('h'):
            return int(timeframe[:-1]), "hour"
        elif timeframe.endswith('d'):
            return int(timeframe[:-1]), "day"
        elif timeframe.endswith('w'):
            return int(timeframe[:-1]), "week"
        else:
            raise ValueError(f"Invalid timeframe format: {timeframe}")

# Quote Schema
@dataclass
class QuoteSchema:
    """Schema for quote data."""
    
    timestamp: datetime
    symbol: str
    bid_price: float
    ask_price: float
    bid_size: int
    ask_size: int
    exchange: Optional[str] = None
    conditions: Optional[List[str]] = None
    sequence_number: Optional[int] = None
    tape: Optional[str] = None
    source: str = "polygon"
    
    @classmethod
    def from_polygon(cls, data: Dict[str, Any], symbol: str) -> "QuoteSchema":
        """
        Create a QuoteSchema instance from Polygon API data.
        
        Args:
            data: Dictionary with Polygon API data
            symbol: Ticker symbol
            
        Returns:
            QuoteSchema instance
        """
        return cls(
            timestamp=datetime.fromtimestamp(data['t'] / 1000000000),
            symbol=symbol,
            bid_price=data['p'],
            ask_price=data['P'],
            bid_size=data['s'],
            ask_size=data['S'],
            exchange=data.get('x'),
            conditions=data.get('c'),
            sequence_number=data.get('q'),
            tape=data.get('z'),
            source="polygon"
        )

# Trade Schema
@dataclass
class TradeSchema:
    """Schema for trade data."""
    
    timestamp: datetime
    symbol: str
    price: float
    size: int
    exchange: str
    conditions: Optional[List[str]] = None
    tape: Optional[str] = None
    sequence_number: Optional[int] = None
    trade_id: Optional[str] = None
    source: str = "polygon"
    
    @classmethod
    def from_polygon(cls, data: Dict[str, Any], symbol: str) -> "TradeSchema":
        """
        Create a TradeSchema instance from Polygon API data.
        
        Args:
            data: Dictionary with Polygon API data
            symbol: Ticker symbol
            
        Returns:
            TradeSchema instance
        """
        return cls(
            timestamp=datetime.fromtimestamp(data['t'] / 1000000000),
            symbol=symbol,
            price=data['p'],
            size=data['s'],
            exchange=data['x'],
            conditions=data.get('c'),
            tape=data.get('z'),
            sequence_number=data.get('q'),
            trade_id=data.get('i'),
            source="polygon"
        )

# Options Schema
@dataclass
class OptionsSchema:
    """Schema for options data."""
    
    timestamp: datetime
    symbol: str
    underlying: str
    expiration: date
    strike: float
    option_type: str  # 'call' or 'put'
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    timeframe: str = "1d"
    multiplier: int = 1
    timespan_unit: str = "day"
    source: str = "polygon"
    
    @classmethod
    def from_polygon(cls, data: Dict[str, Any], symbol: str, underlying: str, 
                    expiration: date, strike: float, option_type: str) -> "OptionsSchema":
        """
        Create an OptionsSchema instance from Polygon API data.
        
        Args:
            data: Dictionary with Polygon API data
            symbol: Option symbol
            underlying: Underlying asset symbol
            expiration: Expiration date
            strike: Strike price
            option_type: Option type ('call' or 'put')
            
        Returns:
            OptionsSchema instance
        """
        return cls(
            timestamp=datetime.fromtimestamp(data['t'] / 1000),
            symbol=symbol,
            underlying=underlying,
            expiration=expiration,
            strike=strike,
            option_type=option_type.lower(),
            open=data.get('o'),
            high=data.get('h'),
            low=data.get('l'),
            close=data.get('c'),
            volume=data.get('v'),
            open_interest=data.get('oi'),
            source="polygon"
        )

# Options Flow Schema
@dataclass
class OptionsFlowSchema:
    """Schema for options flow data."""
    
    id: str
    timestamp: datetime
    symbol: str
    contract_type: str  # 'call' or 'put'
    strike: float
    expiration_date: datetime
    premium: Optional[float] = None
    size: Optional[float] = None
    open_interest: Optional[float] = None
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    sentiment: Optional[str] = None
    trade_type: Optional[str] = None
    source: str = "unusual_whales"
    
    @classmethod
    def from_unusual_whales(cls, data: Dict[str, Any]) -> "OptionsFlowSchema":
        """
        Create an OptionsFlowSchema instance from Unusual Whales API data.
        
        Args:
            data: Dictionary with Unusual Whales API data
            
        Returns:
            OptionsFlowSchema instance
        """
        return cls(
            id=data['id'],
            timestamp=pd.to_datetime(data['timestamp']),
            symbol=data['symbol'],
            contract_type=data['contract_type'].lower(),
            strike=data['strike'],
            expiration_date=pd.to_datetime(data['expiration_date']),
            premium=data.get('premium'),
            size=data.get('size'),
            open_interest=data.get('open_interest'),
            implied_volatility=data.get('implied_volatility'),
            delta=data.get('delta'),
            gamma=data.get('gamma'),
            theta=data.get('theta'),
            vega=data.get('vega'),
            sentiment=data.get('sentiment'),
            trade_type=data.get('trade_type'),
            source="unusual_whales"
        )

# Market Status Schema
@dataclass
class MarketStatusSchema:
    """Schema for market status data."""
    
    timestamp: datetime
    market: str
    status: str
    next_open: Optional[datetime] = None
    next_close: Optional[datetime] = None
    early_close: bool = False
    late_open: bool = False
    
    @classmethod
    def from_polygon(cls, data: Dict[str, Any]) -> "MarketStatusSchema":
        """
        Create a MarketStatusSchema instance from Polygon API data.
        
        Args:
            data: Dictionary with Polygon API data
            
        Returns:
            MarketStatusSchema instance
        """
        return cls(
            timestamp=datetime.now(),
            market=data['market'],
            status=data['status'],
            next_open=pd.to_datetime(data.get('next_open')),
            next_close=pd.to_datetime(data.get('next_close')),
            early_close=data.get('early_close', False),
            late_open=data.get('late_open', False)
        )
    
    @classmethod
    def from_alpaca(cls, data: Dict[str, Any]) -> "MarketStatusSchema":
        """
        Create a MarketStatusSchema instance from Alpaca API data.
        
        Args:
            data: Dictionary with Alpaca API data
            
        Returns:
            MarketStatusSchema instance
        """
        # Map Alpaca market status to our format
        status_map = {
            'open': 'open',
            'closed': 'closed',
            'pre-market': 'extended_hours',
            'after-hours': 'extended_hours'
        }
        
        return cls(
            timestamp=datetime.now(),
            market='us_equity',
            status=status_map.get(data['status'], data['status']),
            next_open=pd.to_datetime(data.get('next_open')),
            next_close=pd.to_datetime(data.get('next_close')),
            early_close=data.get('is_early_close', False)
        )

# News Article Schema
@dataclass
class NewsArticleSchema:
    """Schema for news article data."""
    
    article_id: str
    published_utc: datetime
    title: str
    article_url: str
    source: str
    author: Optional[str] = None
    tickers: Optional[List[str]] = None
    image_url: Optional[str] = None
    description: Optional[str] = None
    keywords: Optional[List[str]] = None
    
    @classmethod
    def from_polygon(cls, data: Dict[str, Any]) -> "NewsArticleSchema":
        """
        Create a NewsArticleSchema instance from Polygon API data.
        
        Args:
            data: Dictionary with Polygon API data
            
        Returns:
            NewsArticleSchema instance
        """
        return cls(
            article_id=data['id'],
            published_utc=pd.to_datetime(data['published_utc']),
            title=data['title'],
            article_url=data['article_url'],
            source=data['publisher']['name'],
            author=data.get('author'),
            tickers=data.get('tickers'),
            image_url=data.get('image_url'),
            description=data.get('description'),
            keywords=data.get('keywords')
        )

# Ticker Details Schema
@dataclass
class TickerDetailsSchema:
    """Schema for ticker details data."""
    
    ticker: str
    name: str
    market: str
    locale: str
    type: str
    currency: str
    active: bool = True
    primary_exchange: Optional[str] = None
    last_updated: datetime = field(default_factory=datetime.now)
    description: Optional[str] = None
    sic_code: Optional[str] = None
    sic_description: Optional[str] = None
    ticker_root: Optional[str] = None
    homepage_url: Optional[str] = None
    total_employees: Optional[int] = None
    list_date: Optional[date] = None
    share_class_shares_outstanding: Optional[int] = None
    weighted_shares_outstanding: Optional[int] = None
    market_cap: Optional[int] = None
    phone_number: Optional[str] = None
    address: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_polygon(cls, data: Dict[str, Any]) -> "TickerDetailsSchema":
        """
        Create a TickerDetailsSchema instance from Polygon API data.
        
        Args:
            data: Dictionary with Polygon API data
            
        Returns:
            TickerDetailsSchema instance
        """
        return cls(
            ticker=data['ticker'],
            name=data['name'],
            market=data['market'],
            locale=data['locale'],
            type=data['type'],
            currency=data['currency_name'],
            active=data.get('active', True),
            primary_exchange=data.get('primary_exchange'),
            last_updated=datetime.now(),
            description=data.get('description'),
            sic_code=data.get('sic_code'),
            sic_description=data.get('sic_description'),
            ticker_root=data.get('ticker_root'),
            homepage_url=data.get('homepage_url'),
            total_employees=data.get('total_employees'),
            list_date=pd.to_datetime(data.get('list_date')).date() if data.get('list_date') else None,
            share_class_shares_outstanding=data.get('share_class_shares_outstanding'),
            weighted_shares_outstanding=data.get('weighted_shares_outstanding'),
            market_cap=data.get('market_cap'),
            phone_number=data.get('phone_number'),
            address=data.get('address'),
        )
# News Sentiment Schema
@dataclass
class NewsSentimentSchema:
    """Schema for news sentiment analysis."""
    
    article_id: str
    timestamp: datetime
    symbol: str
    sentiment_score: float
    sentiment_label: str
    confidence: float
    entity_mentions: Optional[Dict[str, Any]] = None
    keywords: Optional[Dict[str, Any]] = None
    model_version: str = "finbert-v1"
    
    @classmethod
    def from_finbert(cls, article_id: str, symbol: str, sentiment_data: Dict[str, Any]) -> "NewsSentimentSchema":
        """
        Create a NewsSentimentSchema instance from FinBERT sentiment analysis.
        
        Args:
            article_id: News article ID
            symbol: Ticker symbol
            sentiment_data: Dictionary with sentiment analysis data
            
        Returns:
            NewsSentimentSchema instance
        """
        return cls(
            article_id=article_id,
            timestamp=datetime.now(),
            symbol=symbol,
            sentiment_score=sentiment_data['sentiment_score'],
            sentiment_label=sentiment_data['sentiment_label'],
            confidence=sentiment_data['confidence'],
            entity_mentions=sentiment_data.get('entity_mentions'),
            keywords=sentiment_data.get('keywords'),
            model_version=sentiment_data.get('model_version', 'finbert-v1')
        )

# Feature Schema
@dataclass
class FeatureSchema:
    """Schema for feature data."""
    
    timestamp: datetime
    symbol: str
    feature_name: str
    feature_value: float
    timeframe: str
    feature_group: str
    
    @classmethod
    def from_calculator(cls, symbol: str, feature_name: str, feature_value: float, 
                       timestamp: datetime, timeframe: str, feature_group: str) -> "FeatureSchema":
        """
        Create a FeatureSchema instance from feature calculator output.
        
        Args:
            symbol: Ticker symbol
            feature_name: Name of the feature
            feature_value: Value of the feature
            timestamp: Timestamp of the feature
            timeframe: Data timeframe
            feature_group: Group the feature belongs to
            
        Returns:
            FeatureSchema instance
        """
        return cls(
            timestamp=timestamp,
            symbol=symbol,
            feature_name=feature_name,
            feature_value=feature_value,
            timeframe=timeframe,
            feature_group=feature_group
        )

# Feature Metadata Schema
@dataclass
class FeatureMetadataSchema:
    """Schema for feature metadata."""
    
    feature_name: str
    description: str
    formula: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"
    is_active: bool = True

# Model Schema
@dataclass
class ModelSchema:
    """Schema for machine learning model data."""
    
    model_id: str
    model_name: str
    model_type: str
    target: str
    features: List[str]
    parameters: Dict[str, Any]
    metrics: Dict[str, Any]
    created_at: datetime
    trained_at: datetime
    version: str
    status: str
    file_path: str

# Model Training Run Schema
@dataclass
class ModelTrainingRunSchema:
    """Schema for model training run data."""
    
    run_id: str
    model_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    status: str = "running"
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Optional[Dict[str, Any]] = None
    logs: Optional[str] = None

# Trading Signal Schema
@dataclass
class TradingSignalSchema:
    """Schema for trading signal data."""
    
    signal_id: str
    timestamp: datetime
    symbol: str
    signal_type: str
    confidence: float
    model_id: Optional[str] = None
    timeframe: str = "1d"
    parameters: Optional[Dict[str, Any]] = None
    features_snapshot: Optional[Dict[str, Any]] = None

# Order Schema
@dataclass
class OrderSchema:
    """Schema for order data."""
    
    order_id: str
    timestamp: datetime
    symbol: str
    order_type: str
    side: str
    quantity: float
    price: Optional[float] = None
    status: str = "pending"
    signal_id: Optional[str] = None
    strategy_id: Optional[str] = None
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    commission: float = 0.0
    updated_at: datetime = field(default_factory=datetime.now)
    external_order_id: Optional[str] = None

# Position Schema
@dataclass
class PositionSchema:
    """Schema for position data."""
    
    position_id: str
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    entry_time: datetime
    last_update: datetime = field(default_factory=datetime.now)
    strategy_id: Optional[str] = None
    status: str = "open"
    pnl: float = 0.0
    pnl_percentage: float = 0.0
    metadata: Optional[Dict[str, Any]] = None

# System Metrics Schema
@dataclass
class SystemMetricsSchema:
    """Schema for system metrics data."""
    
    timestamp: datetime
    metric_name: str
    metric_value: float
    component: str
    host: str
    tags: Optional[Dict[str, Any]] = None