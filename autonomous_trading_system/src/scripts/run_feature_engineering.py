#!/usr/bin/env python3
"""
Feature Engineering Script

This script runs the feature engineering pipeline, which calculates various features
from raw market data and stores them in the feature store. It includes technical indicators,
volatility metrics, and sentiment analysis from financial news.

Usage:
    python run_feature_engineering.py [--symbols SYMBOLS] [--days DAYS] [--continuous]

Options:
    --symbols SYMBOLS    Comma-separated list of symbols to process (default: all active tickers)
    --days DAYS          Number of days of historical data to process (default: 7)
    --continuous         Run in continuous mode, processing new data as it arrives
"""

import os
import sys
import time
import argparse
import logging
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.feature_engineering.calculators.price_features import PriceFeatureCalculator
from src.feature_engineering.calculators.volume_features import VolumeFeatureCalculator
from src.feature_engineering.calculators.volatility_features import VolatilityFeatureCalculator
from src.feature_engineering.calculators.momentum_features import MomentumFeatureCalculator
from src.feature_engineering.calculators.trend_features import TrendFeatureCalculator
from src.feature_engineering.calculators.pattern_features import PatternFeatureCalculator
from src.feature_engineering.calculators.microstructure_features import MicrostructureFeatureCalculator
from src.feature_engineering.text_analysis.finbert_sentiment import FinBERTSentimentAnalyzer
from src.feature_engineering.store.feature_store import FeatureStore
from src.feature_engineering.pipeline.feature_pipeline import FeaturePipeline
from src.config.database_config import get_db_connection_string
from src.utils.logging import setup_logging

# Configure logging
logger = logging.getLogger(__name__)
setup_logging()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run feature engineering pipeline')
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols to process')
    parser.add_argument('--days', type=int, default=7, help='Number of days of historical data to process')
    parser.add_argument('--continuous', action='store_true', help='Run in continuous mode')
    parser.add_argument('--skip-sentiment', action='store_true', help='Skip sentiment analysis')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for sentiment analysis')
    return parser.parse_args()

def get_active_tickers(db_engine) -> List[str]:
    """Get list of active tickers from the database."""
    query = """
    SELECT ticker FROM ticker_details
    WHERE active = TRUE
    ORDER BY market_cap DESC NULLS LAST
    LIMIT 500
    """
    return pd.read_sql(query, db_engine)['ticker'].tolist()

def initialize_feature_calculators(db_engine) -> Dict[str, Any]:
    """Initialize all feature calculators."""
    calculators = {
        'price': PriceFeatureCalculator(),
        'volume': VolumeFeatureCalculator(),
        'volatility': VolatilityFeatureCalculator(),
        'momentum': MomentumFeatureCalculator(),
        'trend': TrendFeatureCalculator(),
        'pattern': PatternFeatureCalculator(),
        'microstructure': MicrostructureFeatureCalculator(),
    }
    
    # Initialize feature store
    feature_store = FeatureStore(db_engine)
    
    # Initialize feature pipeline
    pipeline = FeaturePipeline(calculators, feature_store)
    
    return {
        'calculators': calculators,
        'feature_store': feature_store,
        'pipeline': pipeline
    }

def initialize_sentiment_analyzer(use_gpu: bool = False) -> FinBERTSentimentAnalyzer:
    """Initialize the FinBERT sentiment analyzer."""
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    logger.info(f"Initializing FinBERT sentiment analyzer with device: {device}")
    
    return FinBERTSentimentAnalyzer(
        model_name="ProsusAI/finbert",
        device=device,
        batch_size=16,
        max_length=512,
        use_spacy=True
    )

def run_historical_feature_engineering(
    pipeline: FeaturePipeline,
    symbols: List[str],
    days: int,
    timeframes: List[str] = ["1m", "5m", "15m", "1h", "1d"]
):
    """
    Run feature engineering on historical data.
    
    Args:
        pipeline: Feature engineering pipeline
        symbols: List of symbols to process
        days: Number of days of historical data to process
        timeframes: List of timeframes to process
    """
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    logger.info(f"Running historical feature engineering from {start_date} to {end_date}")
    logger.info(f"Processing {len(symbols)} symbols across {len(timeframes)} timeframes")
    
    # Process each symbol and timeframe
    for symbol in symbols:
        for timeframe in timeframes:
            try:
                logger.info(f"Processing {symbol} for timeframe {timeframe}")
                pipeline.process_symbol(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
            except Exception as e:
                logger.error(f"Error processing {symbol} for timeframe {timeframe}: {e}")
                continue
    
    logger.info("Historical feature engineering completed")

def run_continuous_feature_engineering(
    pipeline: FeaturePipeline,
    symbols: List[str],
    timeframes: List[str] = ["1m", "5m", "15m", "1h", "1d"],
    interval_seconds: int = 300  # Default: 5 minutes
):
    """
    Run feature engineering continuously on new data.
    
    Args:
        pipeline: Feature engineering pipeline
        symbols: List of symbols to process
        timeframes: List of timeframes to process
        interval_seconds: Interval between runs in seconds
    """
    logger.info(f"Starting continuous feature engineering (interval: {interval_seconds}s)")
    
    # Track the last processed timestamp
    last_timestamp = datetime.now()
    
    try:
        while True:
            # Wait for the next interval
            time.sleep(interval_seconds)
            
            # Calculate date range (from last timestamp to now)
            current_time = datetime.now()
            
            logger.info(f"Running feature engineering from {last_timestamp} to {current_time}")
            
            # Process each symbol and timeframe
            for symbol in symbols:
                for timeframe in timeframes:
                    try:
                        logger.info(f"Processing {symbol} for timeframe {timeframe}")
                        pipeline.process_symbol(
                            symbol=symbol,
                            timeframe=timeframe,
                            start_date=last_timestamp,
                            end_date=current_time
                        )
                    except Exception as e:
                        logger.error(f"Error processing {symbol} for timeframe {timeframe}: {e}")
                        continue
            
            # Update last timestamp
            last_timestamp = current_time
            
    except KeyboardInterrupt:
        logger.info("Continuous feature engineering stopped by user")

def run_sentiment_analysis(
    analyzer: FinBERTSentimentAnalyzer,
    symbols: Optional[List[str]] = None,
    days: int = 7,
    continuous: bool = False
):
    """
    Run sentiment analysis on news articles.
    
    Args:
        analyzer: FinBERT sentiment analyzer
        symbols: List of symbols to analyze (None for all)
        days: Number of days of historical data to analyze
        continuous: Whether to run in continuous mode
    """
    if continuous:
        # First run historical analysis
        run_historical_sentiment_analysis(analyzer, symbols, days)
        
        # Then run in continuous mode
        run_continuous_sentiment_analysis(analyzer, symbols)
    else:
        # Run historical analysis only
        run_historical_sentiment_analysis(analyzer, symbols, days)

def run_historical_sentiment_analysis(
    analyzer: FinBERTSentimentAnalyzer,
    symbols: Optional[List[str]] = None,
    days: int = 7
):
    """
    Run sentiment analysis on historical news articles.
    
    Args:
        analyzer: FinBERT sentiment analyzer
        symbols: List of symbols to analyze (None for all)
        days: Number of days of historical data to analyze
    """
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    logger.info(f"Running historical sentiment analysis from {start_date} to {end_date}")
    
    # Analyze news articles
    results_df = analyzer.analyze_news_articles(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        store=True
    )
    
    if results_df.empty:
        logger.warning("No news articles found for sentiment analysis")
        return
    
    logger.info(f"Analyzed sentiment for {len(results_df)} news articles")
    
    # Calculate and store sentiment features
    if symbols is None:
        # Get unique symbols from results
        unique_symbols = results_df["symbol"].unique().tolist()
    else:
        unique_symbols = symbols
    
    if not unique_symbols:
        logger.warning("No symbols found for feature calculation")
        return
    
    logger.info(f"Calculating sentiment features for {len(unique_symbols)} symbols")
    
    features_df = analyzer.get_sentiment_features(
        symbols=unique_symbols,
        start_date=start_date,
        end_date=end_date
    )
    
    if features_df.empty:
        logger.warning("No sentiment features generated")
        return
    
    # Store features
    num_stored = analyzer.store_sentiment_features(features_df)
    logger.info(f"Stored {num_stored} sentiment features")

def run_continuous_sentiment_analysis(
    analyzer: FinBERTSentimentAnalyzer,
    symbols: Optional[List[str]] = None,
    interval_seconds: int = 3600  # Default: 1 hour
):
    """
    Run sentiment analysis continuously on new news articles.
    
    Args:
        analyzer: FinBERT sentiment analyzer
        symbols: List of symbols to analyze (None for all)
        interval_seconds: Interval between runs in seconds
    """
    logger.info(f"Starting continuous sentiment analysis (interval: {interval_seconds}s)")
    
    # Track the last processed timestamp
    last_timestamp = datetime.now()
    
    try:
        while True:
            # Wait for the next interval
            time.sleep(interval_seconds)
            
            # Calculate date range (from last timestamp to now)
            current_time = datetime.now()
            
            logger.info(f"Running sentiment analysis from {last_timestamp} to {current_time}")
            
            # Analyze news articles
            results_df = analyzer.analyze_news_articles(
                start_date=last_timestamp,
                end_date=current_time,
                symbols=symbols,
                store=True
            )
            
            if results_df.empty:
                logger.info("No new news articles found for sentiment analysis")
            else:
                logger.info(f"Analyzed sentiment for {len(results_df)} news articles")
                
                # Calculate and store sentiment features
                if symbols is None:
                    # Get unique symbols from results
                    unique_symbols = results_df["symbol"].unique().tolist()
                else:
                    unique_symbols = symbols
                
                if unique_symbols:
                    logger.info(f"Calculating sentiment features for {len(unique_symbols)} symbols")
                    
                    features_df = analyzer.get_sentiment_features(
                        symbols=unique_symbols,
                        start_date=last_timestamp,
                        end_date=current_time
                    )
                    
                    if not features_df.empty:
                        # Store features
                        num_stored = analyzer.store_sentiment_features(features_df)
                        logger.info(f"Stored {num_stored} sentiment features")
            
            # Update last timestamp
            last_timestamp = current_time
            
    except KeyboardInterrupt:
        logger.info("Continuous sentiment analysis stopped by user")

def main():
    """Main function."""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    args = parse_args()
    
    # Parse symbols
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Create database engine
    db_engine = create_engine(get_db_connection_string())
    
    # Get active tickers if no symbols specified
    if symbols is None:
        symbols = get_active_tickers(db_engine)
        logger.info(f"Using {len(symbols)} active tickers")
    
    # Initialize feature calculators and pipeline
    components = initialize_feature_calculators(db_engine)
    pipeline = components['pipeline']
    
    # Initialize sentiment analyzer if not skipped
    if not args.skip_sentiment:
        try:
            import torch
            analyzer = initialize_sentiment_analyzer(use_gpu=args.gpu)
            
            # Run sentiment analysis in a separate thread
            from threading import Thread
            sentiment_thread = Thread(
                target=run_sentiment_analysis,
                args=(analyzer, symbols, args.days, args.continuous)
            )
            sentiment_thread.daemon = True
            sentiment_thread.start()
            logger.info("Started sentiment analysis thread")
        except ImportError as e:
            logger.warning(f"Could not initialize sentiment analyzer: {e}")
            logger.warning("Sentiment analysis will be skipped")
    
    # Run feature engineering
    if args.continuous:
        # First run historical feature engineering
        run_historical_feature_engineering(pipeline, symbols, args.days)
        
        # Then run in continuous mode
        run_continuous_feature_engineering(pipeline, symbols)
    else:
        # Run historical feature engineering only
        run_historical_feature_engineering(pipeline, symbols, args.days)
    
    logger.info("Feature engineering completed")

if __name__ == "__main__":
    main()