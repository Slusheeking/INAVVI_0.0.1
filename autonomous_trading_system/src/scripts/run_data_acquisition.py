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

# Add the parent directory to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from autonomous_trading_system.src.data_acquisition.api.polygon_client import PolygonClient
from autonomous_trading_system.src.data_acquisition.api.alpaca_client import AlpacaClient
from autonomous_trading_system.src.data_acquisition.api.unusual_whales_client import UnusualWhalesClient
from autonomous_trading_system.src.data_acquisition.pipeline.data_pipeline import DataPipeline
from autonomous_trading_system.src.data_acquisition.validation.data_validator import DataValidator
from autonomous_trading_system.src.data_acquisition.storage.timescale_storage import TimescaleStorage
from autonomous_trading_system.src.config.database_config import DB_CONFIG
from autonomous_trading_system.src.config.system_config import SYSTEM_CONFIG
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
    include_options_flow: bool = False
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
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Initialize storage
        storage = TimescaleStorage(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            database=DB_CONFIG["database"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"]
        )
        
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
        
        logger.info(f"Data acquisition completed successfully. Collected {result.get('total_records', 0)} records.")
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
                include_options_flow=include_options_flow
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
            include_options_flow=include_options_flow
        )
        
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
