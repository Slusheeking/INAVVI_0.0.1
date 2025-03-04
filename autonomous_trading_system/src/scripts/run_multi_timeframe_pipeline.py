#!/usr/bin/env python3
"""
Multi-Timeframe Pipeline Script

This script demonstrates the complete workflow from data acquisition to feature engineering
using the enhanced multi-timeframe validation and processing capabilities.

Usage:
    python run_multi_timeframe_pipeline.py --symbols AAPL,MSFT,GOOGL --timeframes 1m,5m,15m,1h,1d --days 5
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import traceback
from typing import Dict, List, Optional, Union, Any

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from autonomous_trading_system.src.data_acquisition.pipeline.data_pipeline import DataPipeline
from autonomous_trading_system.src.data_acquisition.validation.data_validator import DataValidator
from autonomous_trading_system.src.feature_engineering.pipeline.multi_timeframe_processor import MultiTimeframeProcessor
from autonomous_trading_system.src.feature_engineering.store.feature_cache import FeatureStoreCache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('multi_timeframe_pipeline.log')
    ]
)
logger = logging.getLogger('multi_timeframe_pipeline')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the multi-timeframe pipeline')
    
    parser.add_argument(
        '--symbols', 
        type=str, 
        default='AAPL,MSFT,GOOGL',
        help='Comma-separated list of ticker symbols'
    )
    
    parser.add_argument(
        '--timeframes', 
        type=str, 
        default='1m,5m,15m,1h,1d',
        help='Comma-separated list of timeframes'
    )
    
    parser.add_argument(
        '--days', 
        type=int, 
        default=5,
        help='Number of days to look back'
    )
    
    parser.add_argument(
        '--feature-groups', 
        type=str, 
        default='price,volume,volatility,momentum,trend',
        help='Comma-separated list of feature groups to calculate'
    )
    
    parser.add_argument(
        '--target-horizon', 
        type=int, 
        default=1,
        help='Horizon for target variable calculation (in bars)'
    )
    
    parser.add_argument(
        '--use-cache', 
        action='store_true',
        help='Use Redis cache for feature storage and retrieval'
    )
    
    parser.add_argument(
        '--parallel', 
        action='store_true',
        help='Use parallel processing'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=str, 
        default='./output',
        help='Directory to save output files'
    )
    
    parser.add_argument(
        '--save-features', 
        action='store_true',
        help='Save features to CSV files'
    )
    
    return parser.parse_args()

def run_pipeline(args):
    """Run the multi-timeframe pipeline."""
    try:
        # Parse arguments
        symbols = args.symbols.split(',')
        timeframes = args.timeframes.split(',')
        feature_groups = args.feature_groups.split(',')
        
        logger.info(f"Running multi-timeframe pipeline for {len(symbols)} symbols and {len(timeframes)} timeframes")
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Timeframes: {timeframes}")
        logger.info(f"Feature groups: {feature_groups}")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        
        logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Initialize components
        data_pipeline = DataPipeline()
        data_validator = DataValidator()
        feature_store_cache = FeatureStoreCache(use_redis=args.use_cache)
        multi_timeframe_processor = MultiTimeframeProcessor(
            feature_store_cache=feature_store_cache,
            use_redis_cache=args.use_cache,
            parallel_processing=args.parallel
        )
        
        # Step 1: Collect multi-timeframe data
        logger.info("Step 1: Collecting multi-timeframe data")
        raw_data = data_pipeline.collect_multi_timeframe_data(
            symbols=symbols,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
            adjusted=True,
            store=False  # Don't store yet, we'll validate first
        )
        
        # Log data collection results
        for symbol, timeframe_data in raw_data.items():
            for timeframe, df in timeframe_data.items():
                logger.info(f"Collected {len(df)} bars for {symbol} {timeframe}")
        
        # Step 2: Validate multi-timeframe data
        logger.info("Step 2: Validating multi-timeframe data")
        validated_data = data_validator.validate_multi_timeframe_data(raw_data)
        
        # Log validation results
        for symbol, timeframe_data in validated_data.items():
            for timeframe, df in timeframe_data.items():
                logger.info(f"Validated {len(df)} bars for {symbol} {timeframe}")
        
        # Step 3: Store validated data
        logger.info("Step 3: Storing validated data")
        data_pipeline._store_multi_timeframe_data(validated_data)
        
        # Step 4: Process multi-timeframe data to generate features
        logger.info("Step 4: Processing multi-timeframe data to generate features")
        features = multi_timeframe_processor.process_multi_timeframe_data(
            data=validated_data,
            feature_groups=feature_groups,
            include_target=True,
            target_horizon=args.target_horizon,
            force_recalculate=not args.use_cache
        )
        
        # Log feature generation results
        for symbol, timeframe_data in features.items():
            for timeframe, df in timeframe_data.items():
                logger.info(f"Generated {len(df.columns)} features for {symbol} {timeframe}")
        
        # Step 5: Save features to CSV files if requested
        if args.save_features:
            logger.info("Step 5: Saving features to CSV files")
            
            # Create output directory if it doesn't exist
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Save features for each symbol and timeframe
            for symbol, timeframe_data in features.items():
                for timeframe, df in timeframe_data.items():
                    # Create symbol directory if it doesn't exist
                    symbol_dir = os.path.join(args.output_dir, symbol)
                    os.makedirs(symbol_dir, exist_ok=True)
                    
                    # Save to CSV
                    csv_path = os.path.join(symbol_dir, f"{timeframe}_features.csv")
                    df.to_csv(csv_path)
                    logger.info(f"Saved features to {csv_path}")
        
        # Step 6: Generate summary statistics
        logger.info("Step 6: Generating summary statistics")
        
        # Create summary statistics
        summary = {
            'timestamp': datetime.now().isoformat(),
            'symbols': symbols,
            'timeframes': timeframes,
            'feature_groups': feature_groups,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'data_collection': {
                'total_symbols': len(raw_data),
                'total_timeframes': len(timeframes),
                'total_bars': sum(
                    len(df) for symbol_data in raw_data.values() 
                    for df in symbol_data.values()
                )
            },
            'validation': {
                'total_symbols': len(validated_data),
                'total_timeframes': len(timeframes),
                'total_bars': sum(
                    len(df) for symbol_data in validated_data.values() 
                    for df in symbol_data.values()
                )
            },
            'feature_generation': {
                'total_symbols': len(features),
                'total_timeframes': len(timeframes),
                'total_features': {
                    symbol: {
                        timeframe: len(df.columns) 
                        for timeframe, df in timeframe_data.items()
                    } 
                    for symbol, timeframe_data in features.items()
                }
            }
        }
        
        # Save summary to JSON file
        summary_path = os.path.join(args.output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved summary to {summary_path}")
        
        # Return summary
        return summary
        
    except Exception as e:
        logger.error(f"Error running multi-timeframe pipeline: {e}")
        logger.debug(traceback.format_exc())
        return None

def main():
    """Main function."""
    args = parse_args()
    summary = run_pipeline(args)
    
    if summary:
        logger.info("Multi-timeframe pipeline completed successfully")
        
        # Print summary
        print("\nMulti-Timeframe Pipeline Summary:")
        print(f"Processed {len(summary['symbols'])} symbols across {len(summary['timeframes'])} timeframes")
        print(f"Date range: {summary['start_date']} to {summary['end_date']}")
        print(f"Total bars collected: {summary['data_collection']['total_bars']}")
        print(f"Total bars validated: {summary['validation']['total_bars']}")
        
        # Print feature counts for each symbol and timeframe
        print("\nFeature counts:")
        for symbol, timeframe_data in summary['feature_generation']['total_features'].items():
            for timeframe, feature_count in timeframe_data.items():
                print(f"  {symbol} {timeframe}: {feature_count} features")
        
        return 0
    else:
        logger.error("Multi-timeframe pipeline failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())