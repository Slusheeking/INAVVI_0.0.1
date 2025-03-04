"""
TimescaleDB Manager Module

This module provides a manager for TimescaleDB operations, including:
- Database connection management
- Hypertable management
- Compression and retention policies
- Database maintenance operations

It serves as a higher-level interface for managing TimescaleDB operations
compared to the lower-level TimescaleStorage class.
"""

import logging
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2.extras import execute_values
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from autonomous_trading_system.src.config.database_config import (
    get_db_connection_string,
    get_connection_params,
    HYPERTABLE_CONFIG,
    COMPRESSION_CONFIG,
    RETENTION_CONFIG,
    DB_POOL_CONFIG
)

logger = logging.getLogger(__name__)

class TimescaleManager:
    """Manager for TimescaleDB operations."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize the TimescaleDB manager.
        
        Args:
            connection_string: Database connection string (defaults to config)
        """
        self.connection_string = connection_string or get_db_connection_string()
        self.engine = create_engine(
            self.connection_string,
            pool_size=DB_POOL_CONFIG.get('pool_size', 10),
            max_overflow=DB_POOL_CONFIG.get('max_overflow', 20),
            pool_timeout=DB_POOL_CONFIG.get('pool_timeout', 30),
            pool_recycle=DB_POOL_CONFIG.get('pool_recycle', 1800)
        )
        
        # Initialize connection
        self._test_connection()
    
    def _test_connection(self) -> None:
        """
        Test the database connection.
        
        Raises:
            Exception: If the connection fails
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("Successfully connected to TimescaleDB")
        except Exception as e:
            logger.error(f"Failed to connect to TimescaleDB: {e}")
            raise
    
    def create_hypertable(self, table_name: str, time_column: str = 'timestamp', 
                         chunk_interval: str = '1 day') -> None:
        """
        Convert a regular table to a TimescaleDB hypertable.
        
        Args:
            table_name: Name of the table to convert
            time_column: Name of the time column
            chunk_interval: Chunk interval (e.g., '1 day', '1 hour')
        """
        try:
            with self.engine.connect() as conn:
                # Check if table is already a hypertable
                result = conn.execute(text(
                    "SELECT * FROM timescaledb_information.hypertables WHERE hypertable_name = :table_name"
                ), {"table_name": table_name})
                
                if result.fetchone() is not None:
                    logger.info(f"Table {table_name} is already a hypertable")
                    return
                
                # Convert to hypertable
                conn.execute(text(
                    f"SELECT create_hypertable('{table_name}', '{time_column}', "
                    f"chunk_time_interval => interval '{chunk_interval}');"
                ))
                
                logger.info(f"Converted {table_name} to a hypertable with chunk interval {chunk_interval}")
        except Exception as e:
            logger.error(f"Error creating hypertable {table_name}: {e}")
            raise
    
    def enable_compression(self, table_name: str, segment_by: str = 'symbol', 
                          order_by: str = 'timestamp', compress_after: str = '7 days') -> None:
        """
        Enable compression for a hypertable.
        
        Args:
            table_name: Name of the hypertable
            segment_by: Column to segment by
            order_by: Column to order by
            compress_after: When to compress chunks (e.g., '7 days')
        """
        if not COMPRESSION_CONFIG.get('enabled', False):
            logger.info(f"Compression is disabled in configuration")
            return
        
        try:
            with self.engine.connect() as conn:
                # Enable compression
                conn.execute(text(
                    f"""
                    ALTER TABLE {table_name} SET (
                        timescaledb.compress,
                        timescaledb.compress_segmentby = '{segment_by}',
                        timescaledb.compress_orderby = '{order_by}'
                    );
                    """
                ))
                
                # Add compression policy
                conn.execute(text(
                    f"SELECT add_compression_policy('{table_name}', interval '{compress_after}');"
                ))
                
                logger.info(f"Enabled compression for {table_name} after {compress_after}")
        except Exception as e:
            logger.error(f"Error enabling compression for {table_name}: {e}")
            raise
    
    def add_retention_policy(self, table_name: str, retention_period: str = '1 year') -> None:
        """
        Add a retention policy to a hypertable.
        
        Args:
            table_name: Name of the hypertable
            retention_period: Retention period (e.g., '1 year', '30 days')
        """
        # Check if retention is enabled for this table
        table_config = RETENTION_CONFIG.get(table_name, {})
        if not table_config.get('enabled', False):
            logger.info(f"Retention policy is disabled for {table_name}")
            return
        
        # Use configured retention period if available
        retention_period = table_config.get('retention_period', retention_period)
        
        try:
            with self.engine.connect() as conn:
                # Add retention policy
                conn.execute(text(
                    f"SELECT add_retention_policy('{table_name}', interval '{retention_period}');"
                ))
                
                logger.info(f"Added retention policy for {table_name} with period {retention_period}")
        except Exception as e:
            logger.error(f"Error adding retention policy for {table_name}: {e}")
            raise
    
    def setup_table(self, table_name: str) -> None:
        """
        Set up a table as a hypertable with compression and retention policies.
        
        Args:
            table_name: Name of the table to set up
        """
        # Get configuration for this table
        config = HYPERTABLE_CONFIG.get(table_name, {})
        if not config:
            logger.warning(f"No hypertable configuration found for {table_name}")
            return
        
        # Create hypertable
        self.create_hypertable(
            table_name, 
            time_column=config.get('time_column', 'timestamp'),
            chunk_interval=config.get('chunk_interval', '1 day')
        )
        
        # Enable compression if configured
        if COMPRESSION_CONFIG.get('enabled', False):
            self.enable_compression(
                table_name,
                segment_by='symbol',
                order_by=config.get('time_column', 'timestamp'),
                compress_after=COMPRESSION_CONFIG.get('compress_after', '7 days')
            )
        
        # Add retention policy if configured
        retention_config = RETENTION_CONFIG.get(table_name, {})
        if retention_config.get('enabled', False):
            self.add_retention_policy(
                table_name,
                retention_period=retention_config.get('retention_period', '1 year')
            )
    
    def setup_all_tables(self) -> None:
        """Set up all configured tables as hypertables with policies."""
        for table_name in HYPERTABLE_CONFIG.keys():
            self.setup_table(table_name)
    
    def get_chunk_info(self, table_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get information about chunks for a hypertable or all hypertables.
        
        Args:
            table_name: Optional name of the hypertable
            
        Returns:
            DataFrame with chunk information
        """
        try:
            with self.engine.connect() as conn:
                query = """
                SELECT 
                    h.hypertable_name, 
                    c.chunk_name, 
                    c.range_start, 
                    c.range_end, 
                    pg_size_pretty(c.total_bytes) as size,
                    c.total_bytes,
                    c.is_compressed
                FROM timescaledb_information.chunks c
                JOIN timescaledb_information.hypertables h 
                    ON c.hypertable_id = h.hypertable_id
                """
                
                if table_name:
                    query += " WHERE h.hypertable_name = :table_name"
                    result = conn.execute(text(query), {"table_name": table_name})
                else:
                    result = conn.execute(text(query))
                
                # Convert to DataFrame
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
        except Exception as e:
            logger.error(f"Error getting chunk info: {e}")
            raise
    
    def compress_chunks(self, table_name: str, older_than: str = '7 days') -> int:
        """
        Manually compress chunks for a hypertable.
        
        Args:
            table_name: Name of the hypertable
            older_than: Compress chunks older than this (e.g., '7 days')
            
        Returns:
            Number of chunks compressed
        """
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(
                    f"SELECT compress_chunks(older_than => interval '{older_than}', hypertable => '{table_name}');"
                ))
                
                # Get number of chunks compressed
                chunks_compressed = len(result.fetchall())
                logger.info(f"Compressed {chunks_compressed} chunks for {table_name}")
                return chunks_compressed
        except Exception as e:
            logger.error(f"Error compressing chunks for {table_name}: {e}")
            raise
    
    def reorder_chunk(self, chunk_name: str, index_name: str) -> None:
        """
        Reorder a chunk to optimize it for a specific index.
        
        Args:
            chunk_name: Name of the chunk
            index_name: Name of the index to optimize for
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text(
                    f"SELECT reorder_chunk('{chunk_name}', '{index_name}');"
                ))
                
                logger.info(f"Reordered chunk {chunk_name} for index {index_name}")
        except Exception as e:
            logger.error(f"Error reordering chunk {chunk_name}: {e}")
            raise
    
    def get_table_size(self, table_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get the size of a table or all tables.
        
        Args:
            table_name: Optional name of the table
            
        Returns:
            DataFrame with table size information
        """
        try:
            with self.engine.connect() as conn:
                query = """
                SELECT 
                    table_name,
                    pg_size_pretty(total_bytes) as total_size,
                    pg_size_pretty(index_bytes) as index_size,
                    pg_size_pretty(toast_bytes) as toast_size,
                    pg_size_pretty(table_bytes) as table_size
                FROM (
                    SELECT *, total_bytes-index_bytes-COALESCE(toast_bytes,0) AS table_bytes FROM (
                        SELECT
                            c.oid,
                            nspname AS table_schema,
                            relname AS table_name,
                            c.reltuples AS row_estimate,
                            pg_total_relation_size(c.oid) AS total_bytes,
                            pg_indexes_size(c.oid) AS index_bytes,
                            pg_total_relation_size(reltoastrelid) AS toast_bytes
                        FROM pg_class c
                        LEFT JOIN pg_namespace n ON n.oid = c.relnamespace
                        WHERE relkind = 'r'
                    ) a
                ) a
                """
                
                if table_name:
                    query += " WHERE table_name = :table_name"
                    result = conn.execute(text(query), {"table_name": table_name})
                else:
                    query += " ORDER BY total_bytes DESC"
                    result = conn.execute(text(query))
                
                # Convert to DataFrame
                df = pd.DataFrame(result.fetchall(), columns=result.keys())
                return df
        except Exception as e:
            logger.error(f"Error getting table size: {e}")
            raise
    
    def vacuum_table(self, table_name: str, full: bool = False) -> None:
        """
        Vacuum a table to reclaim storage and update statistics.
        
        Args:
            table_name: Name of the table
            full: Whether to perform a full vacuum
        """
        try:
            with self.engine.connect() as conn:
                # Set isolation level to AUTOCOMMIT for VACUUM
                connection = conn.connection
                old_isolation_level = connection.isolation_level
                connection.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
                
                # Execute VACUUM
                cursor = connection.cursor()
                if full:
                    cursor.execute(f"VACUUM FULL {table_name};")
                    logger.info(f"Performed full vacuum on {table_name}")
                else:
                    cursor.execute(f"VACUUM {table_name};")
                    logger.info(f"Performed vacuum on {table_name}")
                
                # Restore isolation level
                connection.set_isolation_level(old_isolation_level)
        except Exception as e:
            logger.error(f"Error vacuuming table {table_name}: {e}")
            raise
    
    def analyze_table(self, table_name: str) -> None:
        """
        Analyze a table to update statistics.
        
        Args:
            table_name: Name of the table
        """
        try:
            with self.engine.connect() as conn:
                conn.execute(text(f"ANALYZE {table_name};"))
                logger.info(f"Analyzed table {table_name}")
        except Exception as e:
            logger.error(f"Error analyzing table {table_name}: {e}")
            raise
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            stats = {}
            
            with self.engine.connect() as conn:
                # Get database size
                result = conn.execute(text(
                    "SELECT pg_size_pretty(pg_database_size(current_database())) as db_size;"
                ))
                stats['database_size'] = result.fetchone()[0]
                
                # Get number of tables
                result = conn.execute(text(
                    "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public';"
                ))
                stats['table_count'] = result.fetchone()[0]
                
                # Get number of hypertables
                result = conn.execute(text(
                    "SELECT count(*) FROM timescaledb_information.hypertables;"
                ))
                stats['hypertable_count'] = result.fetchone()[0]
                
                # Get number of chunks
                result = conn.execute(text(
                    "SELECT count(*) FROM timescaledb_information.chunks;"
                ))
                stats['chunk_count'] = result.fetchone()[0]
                
                # Get compression ratio
                result = conn.execute(text("""
                    SELECT 
                        hypertable_name,
                        pg_size_pretty(before_compression_total_bytes) as before_compression,
                        pg_size_pretty(after_compression_total_bytes) as after_compression,
                        round(100 * (1 - after_compression_total_bytes::numeric / before_compression_total_bytes), 2) as compression_ratio
                    FROM timescaledb_information.compression_settings
                    WHERE before_compression_total_bytes > 0;
                """))
                compression_stats = [dict(row) for row in result.fetchall()]
                stats['compression_stats'] = compression_stats
                
                return stats
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
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
    
    def close(self) -> None:
        """Close the database connection."""
        self.engine.dispose()
        logger.info("Database connection closed")