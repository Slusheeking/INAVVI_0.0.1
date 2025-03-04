"""
Storage Package

This package provides storage functionality for the Autonomous Trading System.
It includes classes for storing and retrieving data from TimescaleDB and Redis.
"""

from autonomous_trading_system.src.data_acquisition.storage.timescale_storage import TimescaleStorage
from autonomous_trading_system.src.data_acquisition.storage.timescale_manager import TimescaleManager
from autonomous_trading_system.src.data_acquisition.storage.redis_storage import RedisStorage
from autonomous_trading_system.src.data_acquisition.storage.data_schema import (
    AggregateSchema,
    QuoteSchema,
    TradeSchema,
    OptionsSchema,
    OptionsFlowSchema,
    MarketStatusSchema,
    NewsArticleSchema,
    TickerDetailsSchema,
    NewsSentimentSchema,
    FeatureSchema,
    FeatureMetadataSchema,
    ModelSchema,
    ModelTrainingRunSchema,
    TradingSignalSchema,
    OrderSchema,
    PositionSchema,
    SystemMetricsSchema
)

__all__ = [
    'TimescaleStorage',
    'TimescaleManager',
    'RedisStorage',
    'AggregateSchema',
    'QuoteSchema',
    'TradeSchema',
    'OptionsSchema',
    'OptionsFlowSchema',
    'MarketStatusSchema',
    'NewsArticleSchema',
    'TickerDetailsSchema',
    'NewsSentimentSchema',
    'FeatureSchema',
    'FeatureMetadataSchema',
    'ModelSchema',
    'ModelTrainingRunSchema',
    'TradingSignalSchema',
    'OrderSchema',
    'PositionSchema',
    'SystemMetricsSchema'
]