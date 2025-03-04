#!/usr/bin/env python3
"""
Sentiment Analysis Script

This script runs the FinBERT sentiment analysis on financial news articles.
It can be run in continuous mode to process new articles as they arrive,
or in batch mode to process articles from a specific time period.
"""

import os
import sys
import logging
import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Import after setting Python path
from dotenv import load_dotenv
from src.feature_engineering.text_analysis.finbert_sentiment import FinBERTSentimentAnalyzer
from src.config.database_config import get_db_connection_string

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{project_root}/logs/sentiment_analysis.log")
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run FinBERT sentiment analysis on financial news articles')
    
    parser.add_argument('--days', type=int, default=7,
                        help='Number of days to look back for articles (default: 7)')
    
    parser.add_argument('--symbols', type=str, default=None,
                        help='Comma-separated list of symbols to analyze (default: all active symbols)')
    
    parser.add_argument('--continuous', action='store_true',
                        help='Run in continuous mode, processing new articles as they arrive')
    
    parser.add_argument('--interval', type=int, default=60,
                        help='Interval in minutes between runs in continuous mode (default: 60)')
    
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for sentiment analysis if available')
    
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for processing articles (default: 16)')
    
    parser.add_argument('--window-sizes', type=str, default='1,3,7,14,30',
                        help='Comma-separated list of window sizes in days for feature calculation (default: 1,3,7,14,30)')
    
    parser.add_argument('--limit', type=int, default=1000,
                        help='Maximum number of articles to process per run (default: 1000)')
    
    return parser.parse_args()

def get_active_symbols():
    """Get the list of active symbols from the database."""
    import pandas as pd
    from sqlalchemy import create_engine, text
    
    try:
        engine = create_engine(get_db_connection_string())
        query = text("""
        SELECT DISTINCT ticker 
        FROM active_tickers 
        WHERE is_active = true
        ORDER BY ticker
        """)
        
        df = pd.read_sql(query, engine)
        if df.empty:
            logger.warning("No active tickers found in the database")
            return []
        
        return df['ticker'].tolist()
    except Exception as e:
        logger.error(f"Error getting active symbols: {e}")
        return []

def run_sentiment_analysis(args):
    """Run sentiment analysis on financial news articles."""
    # Load environment variables
    load_dotenv()
    
    # Determine device
    device = "cuda" if args.gpu and is_gpu_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Initialize sentiment analyzer
    analyzer = FinBERTSentimentAnalyzer(
        model_name="ProsusAI/finbert",
        device=device,
        batch_size=args.batch_size,
        max_length=512,
        use_spacy=True
    )
    
    # Parse window sizes
    window_sizes = [int(size) for size in args.window_sizes.split(',')]
    
    # Parse symbols
    symbols = args.symbols.split(',') if args.symbols else get_active_symbols()
    if not symbols:
        logger.error("No symbols specified and no active symbols found")
        return
    
    logger.info(f"Analyzing sentiment for {len(symbols)} symbols")
    
    # Run in continuous mode or batch mode
    if args.continuous:
        run_continuous(analyzer, symbols, args.interval, args.limit, window_sizes)
    else:
        run_batch(analyzer, symbols, args.days, args.limit, window_sizes)

def is_gpu_available():
    """Check if GPU is available."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

def run_batch(analyzer, symbols, days, limit, window_sizes):
    """Run sentiment analysis in batch mode."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    logger.info(f"Running sentiment analysis from {start_date} to {end_date}")
    
    # Analyze news articles
    results_df = analyzer.analyze_news_articles(
        start_date=start_date,
        end_date=end_date,
        symbols=symbols,
        limit=limit,
        store=True
    )
    
    if results_df.empty:
        logger.warning("No sentiment results found")
        return
    
    logger.info(f"Analyzed {len(results_df)} news articles")
    
    # Generate sentiment features
    features_df = analyzer.get_sentiment_features(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        window_sizes=window_sizes
    )
    
    if features_df.empty:
        logger.warning("No sentiment features generated")
        return
    
    # Store sentiment features
    num_features = analyzer.store_sentiment_features(features_df)
    logger.info(f"Stored {num_features} sentiment features")

def run_continuous(analyzer, symbols, interval, limit, window_sizes):
    """Run sentiment analysis in continuous mode."""
    logger.info(f"Starting continuous sentiment analysis with interval {interval} minutes")
    
    try:
        while True:
            # Calculate time window
            end_date = datetime.now()
            start_date = end_date - timedelta(minutes=interval)
            
            logger.info(f"Processing articles from {start_date} to {end_date}")
            
            # Analyze news articles
            results_df = analyzer.analyze_news_articles(
                start_date=start_date,
                end_date=end_date,
                symbols=symbols,
                limit=limit,
                store=True
            )
            
            if not results_df.empty:
                logger.info(f"Analyzed {len(results_df)} news articles")
                
                # Generate sentiment features for the last 30 days
                feature_start_date = end_date - timedelta(days=30)
                features_df = analyzer.get_sentiment_features(
                    symbols=symbols,
                    start_date=feature_start_date,
                    end_date=end_date,
                    window_sizes=window_sizes
                )
                
                if not features_df.empty:
                    # Store sentiment features
                    num_features = analyzer.store_sentiment_features(features_df)
                    logger.info(f"Stored {num_features} sentiment features")
            else:
                logger.info("No new articles found in this interval")
            
            # Sleep until next run
            logger.info(f"Sleeping for {interval} minutes")
            time.sleep(interval * 60)
    except KeyboardInterrupt:
        logger.info("Continuous sentiment analysis stopped by user")
    except Exception as e:
        logger.error(f"Error in continuous sentiment analysis: {e}")

def main():
    """Main function."""
    # Create logs directory if it doesn't exist
    logs_dir = Path(project_root) / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Parse command line arguments
    args = parse_args()
    
    try:
        # Run sentiment analysis
        run_sentiment_analysis(args)
    except Exception as e:
        logger.error(f"Error running sentiment analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()