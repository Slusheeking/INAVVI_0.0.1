"""
TimescaleDB Storage Module

This module provides functionality for storing market data in TimescaleDB.
It uses the data schemas defined in data_schema.py for data validation and transformation.
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import psycopg2
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from autonomous_trading_system.src.config.database_config import get_db_connection_string
from autonomous_trading_system.src.data_acquisition.storage.data_schema import (
    AggregateSchema,
    QuoteSchema,
    TradeSchema,
    OptionsSchema,
    OptionsFlowSchema,
    MarketStatusSchema,
    NewsArticleSchema,
    TickerDetailsSchema
)

logger = logging.getLogger(__name__)

class TimescaleStorage:
    """Class for storing market data in TimescaleDB."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize the TimescaleDB storage.
        
        Args:
            connection_string: Database connection string (defaults to config)
        """
        self.connection_string = connection_string or get_db_connection_string()
        self.engine = create_engine(self.connection_string)
    
    # OHLCV Data Storage Methods (Available in both Polygon and Alpaca)
    def store_stock_aggs(self, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """
        Store stock aggregates in TimescaleDB.
        
        Args:
            df: DataFrame with stock aggregates
            if_exists: How to behave if the table exists ('fail', 'replace', 'append')
            
        Returns:
            Number of rows stored
        """
        if df.empty:
            logger.warning("No stock aggregates to store")
            return 0
        
        # Ensure required columns are present
        required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        try:
            # Store data
            df.to_sql('stock_aggs', self.engine, if_exists=if_exists, index=False, 
                      method='multi', chunksize=10000)
            
            logger.info(f"Stored {len(df)} stock aggregates")
            return len(df)
            
        except Exception as e:
            logger.error(f"Error storing stock aggregates: {e}")
            raise
    
    def store_crypto_aggs(self, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """
        Store crypto aggregates in TimescaleDB.
        
        Args:
            df: DataFrame with crypto aggregates
            if_exists: How to behave if the table exists ('fail', 'replace', 'append')
            
        Returns:
            Number of rows stored
        """
        if df.empty:
            logger.warning("No crypto aggregates to store")
            return 0
        
        # Ensure required columns are present
        required_columns = ['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        try:
            # Store data
            df.to_sql('crypto_aggs', self.engine, if_exists=if_exists, index=False, 
                      method='multi', chunksize=10000)
            
            logger.info(f"Stored {len(df)} crypto aggregates")
            return len(df)
            
        except Exception as e:
            logger.error(f"Error storing crypto aggregates: {e}")
            raise
    
    # Quote Data Storage Methods (Polygon only - not available in Alpaca free tier)
    def store_quotes(self, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """
        Store quotes in TimescaleDB.
        Note: Quote data is only available from Polygon, not in Alpaca's free tier.
        
        Args:
            df: DataFrame with quotes
            if_exists: How to behave if the table exists ('fail', 'replace', 'append')
            
        Returns:
            Number of rows stored
        """
        if df.empty:
            logger.warning("No quotes to store")
            return 0
        
        # Ensure required columns are present
        required_columns = ['timestamp', 'symbol', 'bid_price', 'ask_price', 'bid_size', 'ask_size']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        try:
            # Store data
            df.to_sql('quotes', self.engine, if_exists=if_exists, index=False, 
                      method='multi', chunksize=10000)
            
            logger.info(f"Stored {len(df)} quotes")
            return len(df)
            
        except Exception as e:
            logger.error(f"Error storing quotes: {e}")
            raise
    
    # Trade Data Storage Methods (Polygon only - not available in Alpaca free tier)
    def store_trades(self, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """
        Store trades in TimescaleDB.
        Note: Trade data is only available from Polygon, not in Alpaca's free tier.
        
        Args:
            df: DataFrame with trades
            if_exists: How to behave if the table exists ('fail', 'replace', 'append')
            
        Returns:
            Number of rows stored
        """
        if df.empty:
            logger.warning("No trades to store")
            return 0
        
        # Ensure required columns are present
        required_columns = ['timestamp', 'symbol', 'price', 'size', 'exchange']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        try:
            # Store data
            df.to_sql('trades', self.engine, if_exists=if_exists, index=False, 
                      method='multi', chunksize=10000)
            
            logger.info(f"Stored {len(df)} trades")
            return len(df)
            
        except Exception as e:
            logger.error(f"Error storing trades: {e}")
            raise
    
    # Options Data Storage Methods (Polygon only - not available in Alpaca free tier)
    def store_options_aggs(self, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """
        Store options aggregates in TimescaleDB.
        Note: Options data is only available from Polygon, not in Alpaca's free tier.
        
        Args:
            df: DataFrame with options aggregates
            if_exists: How to behave if the table exists ('fail', 'replace', 'append')
            
        Returns:
            Number of rows stored
        """
        if df.empty:
            logger.warning("No options aggregates to store")
            return 0
        
        # Ensure required columns are present
        required_columns = ['timestamp', 'symbol', 'underlying', 'expiration', 'strike', 'option_type']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Ensure expiration is in date format
        if not pd.api.types.is_datetime64_any_dtype(df['expiration']):
            df['expiration'] = pd.to_datetime(df['expiration']).dt.date
        
        try:
            # Store data
            df.to_sql('options_aggs', self.engine, if_exists=if_exists, index=False, 
                      method='multi', chunksize=10000)
            
            logger.info(f"Stored {len(df)} options aggregates")
            return len(df)
            
        except Exception as e:
            logger.error(f"Error storing options aggregates: {e}")
            raise
    
    # Options Flow Data Storage Methods (Unusual Whales data)
    def store_options_flow(self, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """
        Store options flow data in TimescaleDB.
        
        Args:
            df: DataFrame with options flow data
            if_exists: How to behave if the table exists ('fail', 'replace', 'append')
            
        Returns:
            Number of rows stored
        """
        if df.empty:
            logger.warning("No options flow data to store")
            return 0
        
        # Ensure required columns are present
        required_columns = ['id', 'timestamp', 'symbol']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Ensure expiration_date is in datetime format if present
        if 'expiration_date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['expiration_date']):
            df['expiration_date'] = pd.to_datetime(df['expiration_date'])
        
        try:
            # Store data using upsert (ON CONFLICT DO UPDATE)
            with self.engine.connect() as conn:
                # Create a temporary table
                temp_table_name = 'temp_options_flow'
                df.to_sql(temp_table_name, conn, if_exists='replace', index=False)
                
                # Perform upsert
                conn.execute(text(f"""
                    INSERT INTO options_flow
                    SELECT * FROM {temp_table_name}
                    ON CONFLICT (id)
                    DO UPDATE SET
                        timestamp = EXCLUDED.timestamp,
                        symbol = EXCLUDED.symbol,
                        contract_type = EXCLUDED.contract_type,
                        strike = EXCLUDED.strike,
                        expiration_date = EXCLUDED.expiration_date,
                        premium = EXCLUDED.premium,
                        size = EXCLUDED.size,
                        open_interest = EXCLUDED.open_interest,
                        implied_volatility = EXCLUDED.implied_volatility,
                        delta = EXCLUDED.delta,
                        gamma = EXCLUDED.gamma,
                        theta = EXCLUDED.theta,
                        vega = EXCLUDED.vega,
                        sentiment = EXCLUDED.sentiment,
                        trade_type = EXCLUDED.trade_type,
                        source = EXCLUDED.source
                """))
                
                # Drop the temporary table
                conn.execute(text(f"DROP TABLE {temp_table_name}"))
            
            logger.info(f"Stored {len(df)} options flow records")
            return len(df)
            
        except Exception as e:
            logger.error(f"Error storing options flow data: {e}")
            raise
    
    # Reference Data Storage Methods
    def store_ticker_details(self, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """
        Store ticker details in TimescaleDB.
        
        Args:
            df: DataFrame with ticker details
            if_exists: How to behave if the table exists ('fail', 'replace', 'append')
            
        Returns:
            Number of rows stored
        """
        if df.empty:
            logger.warning("No ticker details to store")
            return 0
        
        # Ensure required columns are present
        required_columns = ['ticker', 'name', 'market', 'locale', 'type', 'currency', 'last_updated']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure last_updated is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['last_updated']):
            df['last_updated'] = pd.to_datetime(df['last_updated'])
        
        try:
            # Store data using upsert (ON CONFLICT DO UPDATE)
            with self.engine.connect() as conn:
                # Create a temporary table
                temp_table_name = 'temp_ticker_details'
                df.to_sql(temp_table_name, conn, if_exists='replace', index=False)
                
                # Perform upsert
                conn.execute(text(f"""
                    INSERT INTO ticker_details
                    SELECT * FROM {temp_table_name}
                    ON CONFLICT (ticker)
                    DO UPDATE SET
                        name = EXCLUDED.name,
                        market = EXCLUDED.market,
                        locale = EXCLUDED.locale,
                        type = EXCLUDED.type,
                        currency = EXCLUDED.currency,
                        active = EXCLUDED.active,
                        primary_exchange = EXCLUDED.primary_exchange,
                        last_updated = EXCLUDED.last_updated,
                        description = EXCLUDED.description,
                        sic_code = EXCLUDED.sic_code,
                        sic_description = EXCLUDED.sic_description,
                        ticker_root = EXCLUDED.ticker_root,
                        homepage_url = EXCLUDED.homepage_url,
                        total_employees = EXCLUDED.total_employees,
                        list_date = EXCLUDED.list_date,
                        share_class_shares_outstanding = EXCLUDED.share_class_shares_outstanding,
                        weighted_shares_outstanding = EXCLUDED.weighted_shares_outstanding,
                        market_cap = EXCLUDED.market_cap,
                        phone_number = EXCLUDED.phone_number,
                        address = EXCLUDED.address,
                        metadata = EXCLUDED.metadata
                """))
                
                # Drop the temporary table
                conn.execute(text(f"DROP TABLE {temp_table_name}"))
            
            logger.info(f"Stored {len(df)} ticker details")
            return len(df)
            
        except Exception as e:
            logger.error(f"Error storing ticker details: {e}")
            raise
    
    # Market Status Storage Methods (Available in both Polygon and Alpaca)
    def store_market_status(self, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """
        Store market status in TimescaleDB.
        
        Args:
            df: DataFrame with market status
            if_exists: How to behave if the table exists ('fail', 'replace', 'append')
            
        Returns:
            Number of rows stored
        """
        if df.empty:
            logger.warning("No market status to store")
            return 0
        
        # Ensure required columns are present
        required_columns = ['timestamp', 'market', 'status']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        try:
            # Store data
            df.to_sql('market_status', self.engine, if_exists=if_exists, index=False, 
                      method='multi', chunksize=10000)
            
            logger.info(f"Stored {len(df)} market status records")
            return len(df)
            
        except Exception as e:
            logger.error(f"Error storing market status: {e}")
            raise
    
    # Market Holidays Storage Methods (Available in both Polygon and Alpaca)
    def store_market_holidays(self, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """
        Store market holidays in TimescaleDB.
        
        Args:
            df: DataFrame with market holidays
            if_exists: How to behave if the table exists ('fail', 'replace', 'append')
            
        Returns:
            Number of rows stored
        """
        if df.empty:
            logger.warning("No market holidays to store")
            return 0
        
        # Ensure required columns are present
        required_columns = ['date', 'name', 'market', 'status', 'year']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure date is in date format
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date']).dt.date
        
        try:
            # Store data using upsert (ON CONFLICT DO UPDATE)
            with self.engine.connect() as conn:
                # Create a temporary table
                temp_table_name = 'temp_market_holidays'
                df.to_sql(temp_table_name, conn, if_exists='replace', index=False)
                
                # Perform upsert
                conn.execute(text(f"""
                    INSERT INTO market_holidays
                    SELECT * FROM {temp_table_name}
                    ON CONFLICT (date, market)
                    DO UPDATE SET
                        name = EXCLUDED.name,
                        status = EXCLUDED.status,
                        open_time = EXCLUDED.open_time,
                        close_time = EXCLUDED.close_time,
                        year = EXCLUDED.year
                """))
                
                # Drop the temporary table
                conn.execute(text(f"DROP TABLE {temp_table_name}"))
            
            logger.info(f"Stored {len(df)} market holidays")
            return len(df)
            
        except Exception as e:
            logger.error(f"Error storing market holidays: {e}")
            raise
    
    # News and Sentiment Storage Methods
    def store_news_articles(self, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """
        Store news articles in TimescaleDB.
        
        Args:
            df: DataFrame with news articles
            if_exists: How to behave if the table exists ('fail', 'replace', 'append')
            
        Returns:
            Number of rows stored
        """
        if df.empty:
            logger.warning("No news articles to store")
            return 0
        
        # Ensure required columns are present
        required_columns = ['article_id', 'published_utc', 'title', 'article_url', 'source']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure published_utc is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['published_utc']):
            df['published_utc'] = pd.to_datetime(df['published_utc'])
        
        try:
            # Store data using upsert (ON CONFLICT DO UPDATE)
            with self.engine.connect() as conn:
                # Create a temporary table
                temp_table_name = 'temp_news_articles'
                df.to_sql(temp_table_name, conn, if_exists='replace', index=False)
                
                # Perform upsert
                conn.execute(text(f"""
                    INSERT INTO news_articles
                    SELECT * FROM {temp_table_name}
                    ON CONFLICT (article_id)
                    DO UPDATE SET
                        published_utc = EXCLUDED.published_utc,
                        title = EXCLUDED.title,
                        author = EXCLUDED.author,
                        article_url = EXCLUDED.article_url,
                        tickers = EXCLUDED.tickers,
                        image_url = EXCLUDED.image_url,
                        description = EXCLUDED.description,
                        keywords = EXCLUDED.keywords,
                        source = EXCLUDED.source
                """))
                
                # Drop the temporary table
                conn.execute(text(f"DROP TABLE {temp_table_name}"))
            
            logger.info(f"Stored {len(df)} news articles")
            return len(df)
            
        except Exception as e:
            logger.error(f"Error storing news articles: {e}")
            raise
    
    def store_news_sentiment(self, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """
        Store news sentiment analysis in TimescaleDB.
        
        Args:
            df: DataFrame with news sentiment
            if_exists: How to behave if the table exists ('fail', 'replace', 'append')
            
        Returns:
            Number of rows stored
        """
        if df.empty:
            logger.warning("No news sentiment to store")
            return 0
        
        # Ensure required columns are present
        required_columns = ['article_id', 'timestamp', 'symbol', 'sentiment_score', 'sentiment_label', 'confidence']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        try:
            # Store data using upsert (ON CONFLICT DO UPDATE)
            with self.engine.connect() as conn:
                # Create a temporary table
                temp_table_name = 'temp_news_sentiment'
                df.to_sql(temp_table_name, conn, if_exists='replace', index=False)
                
                # Perform upsert
                conn.execute(text(f"""
                    INSERT INTO news_sentiment
                    SELECT * FROM {temp_table_name}
                    ON CONFLICT (article_id, symbol, timestamp)
                    DO UPDATE SET
                        sentiment_score = EXCLUDED.sentiment_score,
                        sentiment_label = EXCLUDED.sentiment_label,
                        confidence = EXCLUDED.confidence,
                        entity_mentions = EXCLUDED.entity_mentions,
                        keywords = EXCLUDED.keywords,
                        model_version = EXCLUDED.model_version
                """))
                
                # Drop the temporary table
                conn.execute(text(f"DROP TABLE {temp_table_name}"))
            
            logger.info(f"Stored {len(df)} news sentiment records")
            return len(df)
            
        except Exception as e:
            logger.error(f"Error storing news sentiment: {e}")
            raise
    
    # Feature Engineering Storage Methods
    def store_features(self, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """
        Store feature data in TimescaleDB.
        
        Args:
            df: DataFrame with feature data
            if_exists: How to behave if the table exists ('fail', 'replace', 'append')
            
        Returns:
            Number of rows stored
        """
        if df.empty:
            logger.warning("No feature data to store")
            return 0
        
        # Ensure required columns are present
        required_columns = ['timestamp', 'symbol', 'feature_name', 'feature_value', 'timeframe', 'feature_group']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        try:
            # Store data
            df.to_sql('features', self.engine, if_exists=if_exists, index=False, 
                      method='multi', chunksize=10000)
            
            logger.info(f"Stored {len(df)} feature records")
            return len(df)
            
        except Exception as e:
            logger.error(f"Error storing features: {e}")
            raise
    
    def store_feature_metadata(self, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """
        Store feature metadata in TimescaleDB.
        
        Args:
            df: DataFrame with feature metadata
            if_exists: How to behave if the table exists ('fail', 'replace', 'append')
            
        Returns:
            Number of rows stored
        """
        if df.empty:
            logger.warning("No feature metadata to store")
            return 0
        
        # Ensure required columns are present
        required_columns = ['feature_name', 'description', 'created_at', 'updated_at', 'version', 'is_active']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure timestamps are in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['created_at']):
            df['created_at'] = pd.to_datetime(df['created_at'])
        if not pd.api.types.is_datetime64_any_dtype(df['updated_at']):
            df['updated_at'] = pd.to_datetime(df['updated_at'])
        
        try:
            # Store data using upsert (ON CONFLICT DO UPDATE)
            with self.engine.connect() as conn:
                # Create a temporary table
                temp_table_name = 'temp_feature_metadata'
                df.to_sql(temp_table_name, conn, if_exists='replace', index=False)
                
                # Perform upsert
                conn.execute(text(f"""
                    INSERT INTO feature_metadata
                    SELECT * FROM {temp_table_name}
                    ON CONFLICT (feature_name)
                    DO UPDATE SET
                        description = EXCLUDED.description,
                        formula = EXCLUDED.formula,
                        parameters = EXCLUDED.parameters,
                        updated_at = EXCLUDED.updated_at,
                        version = EXCLUDED.version,
                        is_active = EXCLUDED.is_active
                """))
                
                # Drop the temporary table
                conn.execute(text(f"DROP TABLE {temp_table_name}"))
            
            logger.info(f"Stored {len(df)} feature metadata records")
            return len(df)
            
        except Exception as e:
            logger.error(f"Error storing feature metadata: {e}")
            raise
    
    # Model Training Storage Methods
    def store_models(self, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """
        Store model data in TimescaleDB.
        
        Args:
            df: DataFrame with model data
            if_exists: How to behave if the table exists ('fail', 'replace', 'append')
            
        Returns:
            Number of rows stored
        """
        if df.empty:
            logger.warning("No model data to store")
            return 0
        
        # Ensure required columns are present
        required_columns = ['model_id', 'model_name', 'model_type', 'target', 'features', 
                           'parameters', 'metrics', 'created_at', 'trained_at', 'version', 
                           'status', 'file_path']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure timestamps are in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['created_at']):
            df['created_at'] = pd.to_datetime(df['created_at'])
        if not pd.api.types.is_datetime64_any_dtype(df['trained_at']):
            df['trained_at'] = pd.to_datetime(df['trained_at'])
        
        try:
            # Store data using upsert (ON CONFLICT DO UPDATE)
            with self.engine.connect() as conn:
                # Create a temporary table
                temp_table_name = 'temp_models'
                df.to_sql(temp_table_name, conn, if_exists='replace', index=False)
                
                # Perform upsert
                conn.execute(text(f"""
                    INSERT INTO models
                    SELECT * FROM {temp_table_name}
                    ON CONFLICT (model_id)
                    DO UPDATE SET
                        model_name = EXCLUDED.model_name,
                        model_type = EXCLUDED.model_type,
                        target = EXCLUDED.target,
                        features = EXCLUDED.features,
                        parameters = EXCLUDED.parameters,
                        metrics = EXCLUDED.metrics,
                        trained_at = EXCLUDED.trained_at,
                        version = EXCLUDED.version,
                        status = EXCLUDED.status,
                        file_path = EXCLUDED.file_path
                """))
                
                # Drop the temporary table
                conn.execute(text(f"DROP TABLE {temp_table_name}"))
            
            logger.info(f"Stored {len(df)} model records")
            return len(df)
            
        except Exception as e:
            logger.error(f"Error storing models: {e}")
            raise
    
    def store_model_training_runs(self, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """
        Store model training run data in TimescaleDB.
        
        Args:
            df: DataFrame with model training run data
            if_exists: How to behave if the table exists ('fail', 'replace', 'append')
            
        Returns:
            Number of rows stored
        """
        if df.empty:
            logger.warning("No model training run data to store")
            return 0
        
        # Ensure required columns are present
        required_columns = ['run_id', 'model_id', 'start_time', 'status', 'parameters']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure timestamps are in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['start_time']):
            df['start_time'] = pd.to_datetime(df['start_time'])
        if 'end_time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['end_time']):
            df['end_time'] = pd.to_datetime(df['end_time'])
        
        try:
            # Store data using upsert (ON CONFLICT DO UPDATE)
            with self.engine.connect() as conn:
                # Create a temporary table
                temp_table_name = 'temp_model_training_runs'
                df.to_sql(temp_table_name, conn, if_exists='replace', index=False)
                
                # Perform upsert
                conn.execute(text(f"""
                    INSERT INTO model_training_runs
                    SELECT * FROM {temp_table_name}
                    ON CONFLICT (run_id)
                    DO UPDATE SET
                        end_time = EXCLUDED.end_time,
                        status = EXCLUDED.status,
                        parameters = EXCLUDED.parameters,
                        metrics = EXCLUDED.metrics,
                        logs = EXCLUDED.logs
                """))
                
                # Drop the temporary table
                conn.execute(text(f"DROP TABLE {temp_table_name}"))
            
            logger.info(f"Stored {len(df)} model training run records")
            return len(df)
            
        except Exception as e:
            logger.error(f"Error storing model training runs: {e}")
            raise
    
    # Trading Strategy Storage Methods
    def store_trading_signals(self, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """
        Store trading signal data in TimescaleDB.
        
        Args:
            df: DataFrame with trading signal data
            if_exists: How to behave if the table exists ('fail', 'replace', 'append')
            
        Returns:
            Number of rows stored
        """
        if df.empty:
            logger.warning("No trading signal data to store")
            return 0
        
        # Ensure required columns are present
        required_columns = ['signal_id', 'timestamp', 'symbol', 'signal_type', 'confidence', 'timeframe']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        try:
            # Store data
            df.to_sql('trading_signals', self.engine, if_exists=if_exists, index=False, 
                      method='multi', chunksize=10000)
            
            logger.info(f"Stored {len(df)} trading signal records")
            return len(df)
            
        except Exception as e:
            logger.error(f"Error storing trading signals: {e}")
            raise
    
    def store_orders(self, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """
        Store order data in TimescaleDB.
        
        Args:
            df: DataFrame with order data
            if_exists: How to behave if the table exists ('fail', 'replace', 'append')
            
        Returns:
            Number of rows stored
        """
        if df.empty:
            logger.warning("No order data to store")
            return 0
        
        # Ensure required columns are present
        required_columns = ['order_id', 'timestamp', 'symbol', 'order_type', 'side', 'quantity', 'status', 'updated_at']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure timestamps are in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        if not pd.api.types.is_datetime64_any_dtype(df['updated_at']):
            df['updated_at'] = pd.to_datetime(df['updated_at'])
        
        try:
            # Store data using upsert (ON CONFLICT DO UPDATE)
            with self.engine.connect() as conn:
                # Create a temporary table
                temp_table_name = 'temp_orders'
                df.to_sql(temp_table_name, conn, if_exists='replace', index=False)
                
                # Perform upsert
                conn.execute(text(f"""
                    INSERT INTO orders
                    SELECT * FROM {temp_table_name}
                    ON CONFLICT (order_id)
                    DO UPDATE SET
                        external_order_id = EXCLUDED.external_order_id,
                        status = EXCLUDED.status,
                        filled_quantity = EXCLUDED.filled_quantity,
                        filled_price = EXCLUDED.filled_price,
                        commission = EXCLUDED.commission,
                        updated_at = EXCLUDED.updated_at
                """))
                
                # Drop the temporary table
                conn.execute(text(f"DROP TABLE {temp_table_name}"))
            
            logger.info(f"Stored {len(df)} order records")
            return len(df)
            
        except Exception as e:
            logger.error(f"Error storing orders: {e}")
            raise
    
    def store_positions(self, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """
        Store position data in TimescaleDB.
        
        Args:
            df: DataFrame with position data
            if_exists: How to behave if the table exists ('fail', 'replace', 'append')
            
        Returns:
            Number of rows stored
        """
        if df.empty:
            logger.warning("No position data to store")
            return 0
        
        # Ensure required columns are present
        required_columns = ['position_id', 'symbol', 'quantity', 'entry_price', 'current_price', 
                           'entry_time', 'last_update', 'status', 'pnl', 'pnl_percentage']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure timestamps are in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['entry_time']):
            df['entry_time'] = pd.to_datetime(df['entry_time'])
        if not pd.api.types.is_datetime64_any_dtype(df['last_update']):
            df['last_update'] = pd.to_datetime(df['last_update'])
        
        try:
            # Store data using upsert (ON CONFLICT DO UPDATE)
            with self.engine.connect() as conn:
                # Create a temporary table
                temp_table_name = 'temp_positions'
                df.to_sql(temp_table_name, conn, if_exists='replace', index=False)
                
                # Perform upsert
                conn.execute(text(f"""
                    INSERT INTO positions
                    SELECT * FROM {temp_table_name}
                    ON CONFLICT (position_id)
                    DO UPDATE SET
                        quantity = EXCLUDED.quantity,
                        current_price = EXCLUDED.current_price,
                        last_update = EXCLUDED.last_update,
                        status = EXCLUDED.status,
                        pnl = EXCLUDED.pnl,
                        pnl_percentage = EXCLUDED.pnl_percentage,
                        metadata = EXCLUDED.metadata
                """))
                
                # Drop the temporary table
                conn.execute(text(f"DROP TABLE {temp_table_name}"))
            
            logger.info(f"Stored {len(df)} position records")
            return len(df)
            
        except Exception as e:
            logger.error(f"Error storing positions: {e}")
            raise
    
    # Monitoring Storage Methods
    def store_system_metrics(self, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """
        Store system metrics in TimescaleDB.
        
        Args:
            df: DataFrame with system metrics
            if_exists: How to behave if the table exists ('fail', 'replace', 'append')
            
        Returns:
            Number of rows stored
        """
        if df.empty:
            logger.warning("No system metrics to store")
            return 0
        
        # Ensure required columns are present
        required_columns = ['timestamp', 'metric_name', 'metric_value', 'component', 'host']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Ensure timestamp is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        try:
            # Store data
            df.to_sql('system_metrics', self.engine, if_exists=if_exists, index=False, 
                      method='multi', chunksize=10000)
            
            logger.info(f"Stored {len(df)} system metric records")
            return len(df)
            
        except Exception as e:
            logger.error(f"Error storing system metrics: {e}")
            raise
    
    # Schema-based Storage Methods
    def store_from_schema(self, data: Union[List[Any], Any], table_name: str, if_exists: str = 'append') -> int:
        """
        Store data from schema objects in TimescaleDB.
        
        This method converts schema objects to a DataFrame and stores it in the database.
        It's a generic method that can be used with any schema class from data_schema.py.
        
        Args:
            data: Schema object or list of schema objects
            table_name: Name of the table to store data in
            if_exists: How to behave if the table exists ('fail', 'replace', 'append')
            
        Returns:
            Number of rows stored
        """
        # Convert single object to list
        if not isinstance(data, list):
            data = [data]
        
        # Convert to DataFrame
        df = pd.DataFrame([vars(item) for item in data])
        
        # Store based on table name
        if table_name == 'stock_aggs':
            return self.store_stock_aggs(df, if_exists)
        elif table_name == 'crypto_aggs':
            return self.store_crypto_aggs(df, if_exists)
        elif table_name == 'quotes':
            return self.store_quotes(df, if_exists)
        elif table_name == 'trades':
            return self.store_trades(df, if_exists)
        elif table_name == 'options_aggs':
            return self.store_options_aggs(df, if_exists)
        elif table_name == 'options_flow':
            return self.store_options_flow(df, if_exists)
        elif table_name == 'ticker_details':
            return self.store_ticker_details(df, if_exists)
        elif table_name == 'market_status':
            return self.store_market_status(df, if_exists)
        elif table_name == 'market_holidays':
            return self.store_market_holidays(df, if_exists)
        elif table_name == 'news_articles':
            return self.store_news_articles(df, if_exists)
        elif table_name == 'news_sentiment':
            return self.store_news_sentiment(df, if_exists)
        elif table_name == 'features':
            return self.store_features(df, if_exists)
        elif table_name == 'feature_metadata':
            return self.store_feature_metadata(df, if_exists)
        elif table_name == 'models':
            return self.store_models(df, if_exists)
        elif table_name == 'model_training_runs':
            return self.store_model_training_runs(df, if_exists)
        elif table_name == 'trading_signals':
            return self.store_trading_signals(df, if_exists)
        elif table_name == 'orders':
            return self.store_orders(df, if_exists)
        elif table_name == 'positions':
            return self.store_positions(df, if_exists)
        elif table_name == 'system_metrics':
            return self.store_system_metrics(df, if_exists)
        else:
            raise ValueError(f"Unknown table name: {table_name}")
    
    # Query Methods
    def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Execute a SQL query and return the results as a DataFrame.
        
        Args:
            query: SQL query
            params: Query parameters
            
        Returns:
            DataFrame with query results
        """
        try:
            return pd.read_sql(query, self.engine, params=params)
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            raise
    
    def execute_statement(self, statement: str, params: Optional[Dict[str, Any]] = None) -> None:
        """
        Execute a SQL statement.
        
        Args:
            statement: SQL statement
            params: Statement parameters
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text(statement), params)
            logger.info("SQL statement executed successfully")
        except Exception as e:
            logger.error(f"Error executing statement: {e}")
            raise
    
    def get_latest_data(
        self, 
        table: str, 
        symbol: str, 
        timeframe: Optional[str] = None,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Get the latest data for a symbol from a table.
        
        Args:
            table: Table name
            symbol: Ticker symbol
            timeframe: Optional timeframe filter
            limit: Maximum number of rows to return
            
        Returns:
            DataFrame with latest data
        """
        try:
            query = f"SELECT * FROM {table} WHERE symbol = :symbol"
            params = {'symbol': symbol}
            
            if timeframe:
                query += " AND timeframe = :timeframe"
                params['timeframe'] = timeframe
            
            query += " ORDER BY timestamp DESC LIMIT :limit"
            params['limit'] = limit
            
            return pd.read_sql(query, self.engine, params=params)
        except Exception as e:
            logger.error(f"Error getting latest data: {e}")
            raise
    
    def get_data_range(
        self, 
        table: str, 
        symbol: str, 
        start_time: Union[str, datetime],
        end_time: Union[str, datetime],
        timeframe: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get data for a symbol within a time range.
        
        Args:
            table: Table name
            symbol: Ticker symbol
            start_time: Start time
            end_time: End time
            timeframe: Optional timeframe filter
            
        Returns:
            DataFrame with data in the specified range
        """
        try:
            query = f"SELECT * FROM {table} WHERE symbol = :symbol AND timestamp BETWEEN :start_time AND :end_time"
            params = {
                'symbol': symbol,
                'start_time': start_time,
                'end_time': end_time
            }
            
            if timeframe:
                query += " AND timeframe = :timeframe"
                params['timeframe'] = timeframe
            
            query += " ORDER BY timestamp ASC"
            
            return pd.read_sql(query, self.engine, params=params)
        except Exception as e:
            logger.error(f"Error getting data range: {e}")
            raise
    
    def get_symbols_with_data(self, table: str, timeframe: Optional[str] = None) -> List[str]:
        """
        Get a list of symbols that have data in a table.
        
        Args:
            table: Table name
            timeframe: Optional timeframe filter
            
        Returns:
            List of symbols
        """
        try:
            query = f"SELECT DISTINCT symbol FROM {table}"
            params = {}
            
            if timeframe:
                query += " WHERE timeframe = :timeframe"
                params['timeframe'] = timeframe
            
            df = pd.read_sql(query, self.engine, params=params)
            return df['symbol'].tolist()
        except Exception as e:
            logger.error(f"Error getting symbols with data: {e}")
            raise
    
    def get_timeframes_for_symbol(self, table: str, symbol: str) -> List[str]:
        """
        Get a list of timeframes available for a symbol in a table.
        
        Args:
            table: Table name
            symbol: Ticker symbol
            
        Returns:
            List of timeframes
        """
        try:
            query = f"SELECT DISTINCT timeframe FROM {table} WHERE symbol = :symbol"
            params = {'symbol': symbol}
            
            df = pd.read_sql(query, self.engine, params=params)
            return df['timeframe'].tolist()
        except Exception as e:
            logger.error(f"Error getting timeframes for symbol: {e}")
            raise