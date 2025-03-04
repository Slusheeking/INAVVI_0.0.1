#!/usr/bin/env python3
"""
Data Acquisition Script

This script runs the data acquisition pipeline for the Autonomous Trading System.
It collects market data from various sources and stores it in the database.

Usage:
    python run_data_acquisition.py [--symbols SYMBOLS] [--timeframes TIMEFRAMES] [--days-back DAYS_BACK]
                                  [--include-quotes] [--include-trades] [--include-options-flow]
                                  [--continuous] [--interval INTERVAL]

Options:
    --symbols SYMBOLS         Comma-separated list of ticker symbols (default: use universe selection)
    --timeframes TIMEFRAMES   Comma-separated list of timeframes (default: 1m,5m,15m,1h,1d)
    --days-back DAYS_BACK     Number of days to look back for historical data (default: 1)
    --include-quotes          Include quote data (default: False)
    --include-trades          Include trade data (default: False)
    --include-options-flow    Include options flow data (default: False)
    --continuous              Run in continuous mode (default: False)
    --interval INTERVAL       Interval in seconds for continuous mode (default: 60)
"""

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from typing import List, Optional
import uuid
# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from autonomous_trading_system.src.data_acquisition.api.polygon_client import PolygonClient
from autonomous_trading_system.src.data_acquisition.api.alpaca_client import AlpacaClient
from autonomous_trading_system.src.data_acquisition.api.unusual_whales_client import UnusualWhalesClient
from autonomous_trading_system.src.data_acquisition.pipeline.data_pipeline import DataPipeline
from autonomous_trading_system.src.data_acquisition.validation.data_validator import DataValidator
from autonomous_trading_system.src.data_acquisition.storage.timescale_storage import TimescaleStorage
from autonomous_trading_system.src.data_acquisition.storage.redis_storage import RedisStorage
from autonomous_trading_system.src.config.database_config import (
    get_connection_params,
    get_redis_connection_params,
)
from autonomous_trading_system.src.config.system_config import SYSTEM_CONFIG
from autonomous_trading_system.src.data_acquisition.storage.data_schema import SystemMetricsSchema
from autonomous_trading_system.src.utils.logging import setup_logging

# Set up logging
logger = logging.getLogger(__name__)
setup_logging()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Data Acquisition Script")
    parser.add_argument("--symbols", type=str, help="Comma-separated list of ticker symbols")
    parser.add_argument("--timeframes", type=str, default="1m,5m,15m,1h,1d", 
                        help="Comma-separated list of timeframes")
    parser.add_argument("--days-back", type=int, default=1, 
                        help="Number of days to look back for historical data")
    parser.add_argument("--include-quotes", action="store_true", 
                        help="Include quote data")
    parser.add_argument("--include-trades", action="store_true", 
                        help="Include trade data")
    parser.add_argument("--include-options-flow", action="store_true", 
                        help="Include options flow data")
    parser.add_argument("--continuous", action="store_true", 
                        help="Run in continuous mode")
    parser.add_argument("--interval", type=int, default=60, 
                        help="Interval in seconds for continuous mode")
    parser.add_argument("--disable-redis", action="store_true",
                        help="Disable Redis storage (use only TimescaleDB)")
    return parser.parse_args()

def get_default_symbols() -> List[str]:
    """Get default symbols from the system configuration."""
    return SYSTEM_CONFIG.get("default_symbols", ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA"])

def run_data_acquisition(
    symbols: Optional[List[str]] = None,
    timeframes: List[str] = ["1m", "5m", "15m", "1h", "1d"],
    days_back: int = 1,
    include_quotes: bool = False,
    include_trades: bool = False,
    include_options_flow: bool = False,
    use_redis: bool = True
) -> bool:
    """
    Run the data acquisition pipeline.
    
    Args:
        symbols: List of ticker symbols to collect data for
        timeframes: List of timeframes to collect data for
        days_back: Number of days to look back for historical data
        include_quotes: Whether to include quote data
        include_trades: Whether to include trade data
        include_options_flow: Whether to include options flow data
        use_redis: Whether to use Redis for caching
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Get database connection parameters
        db_params = get_connection_params()
        
        # Initialize TimescaleDB storage
        try:
            storage = TimescaleStorage(**db_params)
            logger.info("Successfully connected to TimescaleDB")
        except Exception as e:
            logger.error(f"Failed to connect to TimescaleDB: {e}")
            return False
        
        # Initialize Redis storage if enabled
        redis_storage = None
        if use_redis:
            try:
                redis_params = get_redis_connection_params()
                redis_storage = RedisStorage(redis_params)
                logger.info("Successfully connected to Redis")
                
                # Store system status in Redis
                redis_storage.store_system_status({
                    'status': 'running',
                    'component': 'data_acquisition',
                    'start_time': datetime.now().isoformat(),
                    'pid': os.getpid(),
                    'hostname': os.uname().nodename,
                    'version': '1.0.0'
                })
            except Exception as e:
                logger.warning(f"Failed to connect to Redis, continuing without caching: {e}")
                redis_storage = None
        
        # Initialize data validator
        validator = DataValidator()
        
        # Initialize API clients
        polygon_client = PolygonClient()
        alpaca_client = AlpacaClient()
        unusual_whales_client = UnusualWhalesClient() if include_options_flow else None
        
        # Initialize data pipeline
        pipeline = DataPipeline(
            storage=storage,
            validator=validator,
            redis_storage=redis_storage,
            polygon_client=polygon_client,
            alpaca_client=alpaca_client,
            unusual_whales_client=unusual_whales_client
        )
        
        # Use default symbols if none provided
        if symbols is None:
            symbols = get_default_symbols()
            
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Track system metrics
        run_id = str(uuid.uuid4())
        system_metrics = SystemMetricsSchema(
            timestamp=datetime.now(),
            metric_name="data_acquisition_start",
            metric_value=1.0,
            component="data_acquisition",
            host=os.uname().nodename,
            tags={"run_id": run_id, "symbols": ",".join(symbols), "timeframes": ",".join(timeframes)}
        )
        storage.store_from_schema(system_metrics, "system_metrics")
        start_time = time.time()
        
        logger.info(f"Starting data acquisition for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Run the pipeline
        result = pipeline.run(
            symbols=symbols,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
            include_quotes=include_quotes,
            include_trades=include_trades,
            include_options_flow=include_options_flow
        )
        
        # Calculate execution time
        execution_time = time.time() - start_time
        total_records = result.get('total_records', 0)
        
        # Track completion metrics
        completion_metrics = SystemMetricsSchema(
            timestamp=datetime.now(),
            metric_name="data_acquisition_complete",
            metric_value=total_records,
            component="data_acquisition",
            host=os.uname().nodename,
            tags={
                "run_id": run_id,
                "execution_time": str(execution_time),
                "symbols": ",".join(symbols),
                "timeframes": ",".join(timeframes)
            }
        )
        storage.store_from_schema(completion_metrics, "system_metrics")
        
        # Update Redis status if available
        if redis_storage:
            redis_storage.store_component_status("data_acquisition", {"status": "idle", "last_run": datetime.now().isoformat(), "records_collected": total_records})
        
        logger.info(f"Data acquisition completed successfully. Collected {total_records} records in {execution_time:.2f} seconds.")
        return True
        
    except Exception as e:
        logger.error(f"Error in data acquisition: {str(e)}")
        return False

def main():
    """Main entry point."""
    args = parse_args()
    
    # Parse arguments
    symbols = args.symbols.split(",") if args.symbols else None
    timeframes = args.timeframes.split(",")
    days_back = args.days_back
    include_quotes = args.include_quotes
    include_trades = args.include_trades
    include_options_flow = args.include_options_flow
    continuous = args.continuous
    use_redis = not args.disable_redis
    interval = args.interval
    
    if continuous:
        logger.info(f"Running in continuous mode with interval {interval} seconds")
        while True:
            success = run_data_acquisition(
                symbols=symbols,
                timeframes=timeframes,
                days_back=days_back,
                include_quotes=include_quotes,
                include_trades=include_trades,
                include_options_flow=include_options_flow,
                use_redis=use_redis
            )
            
            if not success:
                logger.warning("Data acquisition failed, retrying in 60 seconds")
                time.sleep(60)  # Always wait at least 60 seconds on failure
            else:
                logger.info(f"Waiting {interval} seconds until next acquisition")
                time.sleep(interval)
    else:
        success = run_data_acquisition(
            symbols=symbols,
            timeframes=timeframes,
            days_back=days_back,
            include_quotes=include_quotes,
            include_trades=include_trades,
            include_options_flow=include_options_flow,
            use_redis=use_redis
        )
        
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
