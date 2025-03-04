"""
TimescaleDB Storage Module

This module provides functionality for storing market data in TimescaleDB.
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
    
    def store_quotes(self, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """
        Store quotes in TimescaleDB.
        
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
    
    def store_trades(self, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """
        Store trades in TimescaleDB.
        
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
    
    def store_options_aggs(self, df: pd.DataFrame, if_exists: str = 'append') -> int:
        """
        Store options aggregates in TimescaleDB.
        
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