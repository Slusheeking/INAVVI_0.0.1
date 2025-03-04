"""
FinBERT Sentiment Analyzer

This module provides a sentiment analysis component for financial news and text
using the FinBERT model, which is a BERT model fine-tuned on financial text.
"""

import os
import logging
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from tqdm import tqdm

# Configure logging
logger = logging.getLogger(__name__)

class FinBERTSentimentAnalyzer:
    """
    Sentiment analyzer for financial text using FinBERT.
    
    This class provides methods to analyze sentiment in financial news articles
    and generate sentiment features for trading models.
    """
    
    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 512,
        use_spacy: bool = True
    ):
        """
        Initialize the FinBERT sentiment analyzer.
        
        Args:
            model_name: Name of the pre-trained model to use
            device: Device to run the model on ('cpu' or 'cuda')
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            use_spacy: Whether to use spaCy for entity extraction
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_spacy = use_spacy
        self.MODEL_VERSION = "finbert-v1.0"
        
        # Label mapping for sentiment
        self.LABEL_MAPPING = {
            0: "positive",
            1: "negative",
            2: "neutral"
        }
        
        # Initialize database connection
        from src.config.database_config import get_db_connection_string
        self.db_engine = create_engine(get_db_connection_string())
        self.Session = sessionmaker(bind=self.db_engine)
        
        # Load models
        self._load_models()
    
    def _load_models(self):
        """Load the FinBERT model and tokenizer."""
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import torch
            
            logger.info(f"Loading FinBERT model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            
            # Move model to specified device
            self.model.to(self.device)
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info(f"FinBERT model loaded successfully on {self.device}")
            
            # Load spaCy model if enabled
            if self.use_spacy:
                import spacy
                logger.info("Loading spaCy model for entity extraction")
                try:
                    self.nlp = spacy.load("en_core_web_sm")
                except OSError:
                    logger.warning("Downloading spaCy model 'en_core_web_sm'")
                    import subprocess
                    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
                    self.nlp = spacy.load("en_core_web_sm")
                logger.info("spaCy model loaded successfully")
            
        except ImportError as e:
            logger.error(f"Error loading models: {e}")
            logger.error("Please install the required dependencies: pip install torch transformers spacy")
            raise
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        import torch
        
        # Tokenize text
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        # Get sentiment label and score
        sentiment_idx = torch.argmax(probabilities, dim=1).item()
        sentiment_label = self.LABEL_MAPPING[sentiment_idx]
        confidence = probabilities[0][sentiment_idx].item()
        
        # Convert sentiment score to range [-1, 1]
        # positive: 1, negative: -1, neutral: 0
        sentiment_score = 0.0
        if sentiment_label == "positive":
            sentiment_score = confidence
        elif sentiment_label == "negative":
            sentiment_score = -confidence
        
        # Extract entities and keywords if spaCy is enabled
        entities = {}
        keywords = []
        if self.use_spacy:
            entities = self._extract_entities(text)
            keywords = self._extract_keywords(text, top_n=10)
        
        return {
            "sentiment_label": sentiment_label,
            "sentiment_score": sentiment_score,
            "confidence": confidence,
            "probabilities": {
                "positive": probabilities[0][0].item(),
                "negative": probabilities[0][1].item(),
                "neutral": probabilities[0][2].item()
            },
            "entity_mentions": entities,
            "keywords": keywords
        }
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text using spaCy.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary of entity types and their mentions
        """
        if not self.use_spacy:
            return {}
        
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
        
        return entities
    
    def _extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract keywords from text using spaCy.
        
        Args:
            text: Text to extract keywords from
            top_n: Number of top keywords to return
            
        Returns:
            List of keywords
        """
        if not self.use_spacy:
            return []
        
        doc = self.nlp(text)
        keywords = []
        
        # Extract nouns and proper nouns
        for token in doc:
            if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop and len(token.text) > 1:
                keywords.append(token.text.lower())
        
        # Count occurrences and get top N
        keyword_counts = {}
        for keyword in keywords:
            if keyword in keyword_counts:
                keyword_counts[keyword] += 1
            else:
                keyword_counts[keyword] = 1
        
        # Sort by count and return top N
        sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
        return [k for k, v in sorted_keywords[:top_n]]
    
    def _analyze_batch(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """
        Analyze sentiment of a batch of sentences.
        
        Args:
            sentences: List of sentences to analyze
            
        Returns:
            List of sentiment analysis results
        """
        import torch
        
        # Tokenize sentences
        inputs = self.tokenizer(
            sentences,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding=True
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get model predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        results = []
        for i, sentence in enumerate(sentences):
            # Get sentiment label and score
            sentiment_idx = torch.argmax(probabilities[i]).item()
            sentiment_label = self.LABEL_MAPPING[sentiment_idx]
            confidence = probabilities[i][sentiment_idx].item()
            
            # Convert sentiment score to range [-1, 1]
            sentiment_score = 0.0
            if sentiment_label == "positive":
                sentiment_score = confidence
            elif sentiment_label == "negative":
                sentiment_score = -confidence
            
            # Extract entities and keywords if spaCy is enabled
            entities = {}
            keywords = []
            if self.use_spacy:
                entities = self._extract_entities(sentence)
                keywords = self._extract_keywords(sentence, top_n=10)
            
            results.append({
                "sentiment_label": sentiment_label,
                "sentiment_score": sentiment_score,
                "confidence": confidence,
                "probabilities": {
                    "positive": probabilities[i][0].item(),
                    "negative": probabilities[i][1].item(),
                    "neutral": probabilities[i][2].item()
                },
                "entity_mentions": entities,
                "keywords": keywords
            })
        
        return results
    
    def analyze_news_articles(
        self,
        start_date: datetime,
        end_date: datetime,
        symbols: Optional[List[str]] = None,
        limit: int = 1000,
        store: bool = True
    ) -> pd.DataFrame:
        """
        Analyze sentiment of news articles from the database.
        
        Args:
            start_date: Start date for articles
            end_date: End date for articles
            symbols: List of symbols to filter articles (None for all)
            limit: Maximum number of articles to analyze
            store: Whether to store results in the database
            
        Returns:
            DataFrame with sentiment analysis results
        """
        # Build query to get news articles
        query = """
        SELECT 
            a.article_id, 
            a.published_utc, 
            a.title, 
            a.description, 
            a.tickers,
            a.source
        FROM 
            news_articles a
        WHERE 
            a.published_utc BETWEEN :start_date AND :end_date
        """
        
        # Add symbol filter if provided
        if symbols is not None and len(symbols) > 0:
            symbols_list = "'{" + ",".join(symbols) + "}'"
            query += f" AND a.tickers && {symbols_list}::varchar[]"
        
        # Add limit
        query += " ORDER BY a.published_utc DESC LIMIT :limit"
        
        # Execute query
        params = {
            "start_date": start_date,
            "end_date": end_date,
            "limit": limit
        }
        
        try:
            articles_df = pd.read_sql(query, self.db_engine, params=params)
            
            if articles_df.empty:
                logger.warning(f"No news articles found between {start_date} and {end_date}")
                return pd.DataFrame()
            
            logger.info(f"Found {len(articles_df)} news articles to analyze")
            
            # Prepare data for sentiment analysis
            results = []
            
            # Process articles in batches
            for i in tqdm(range(0, len(articles_df), self.batch_size), desc="Analyzing articles"):
                batch = articles_df.iloc[i:i+self.batch_size]
                
                # Combine title and description for analysis
                texts = []
                for _, row in batch.iterrows():
                    text = row['title']
                    if pd.notna(row['description']) and row['description']:
                        text += " " + row['description']
                    texts.append(text)
                
                # Analyze batch
                batch_results = self._analyze_batch(texts)
                
                # Combine results with article data
                for j, (_, row) in enumerate(batch.iterrows()):
                    article_id = row['article_id']
                    published_utc = row['published_utc']
                    tickers = row['tickers'] if pd.notna(row['tickers']) else []
                    
                    # If no tickers, skip
                    if not tickers:
                        continue
                    
                    # Add result for each ticker
                    for ticker in tickers:
                        result = {
                            'article_id': article_id,
                            'timestamp': published_utc,
                            'symbol': ticker,
                            'sentiment_score': batch_results[j]['sentiment_score'],
                            'sentiment_label': batch_results[j]['sentiment_label'],
                            'confidence': batch_results[j]['confidence'],
                            'entity_mentions': batch_results[j]['entity_mentions'],
                            'keywords': batch_results[j]['keywords'],
                            'model_version': self.MODEL_VERSION
                        }
                        results.append(result)
            
            # Create DataFrame from results
            results_df = pd.DataFrame(results)
            
            # Store results in database if requested
            if store and not results_df.empty:
                self._store_sentiment_results(results_df)
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error analyzing news articles: {e}")
            return pd.DataFrame()
    
    def _store_sentiment_results(self, results_df: pd.DataFrame) -> int:
        """
        Store sentiment analysis results in the database.
        
        Args:
            results_df: DataFrame with sentiment analysis results
            
        Returns:
            Number of rows stored
        """
        if results_df.empty:
            return 0
        
        # Convert entity_mentions and keywords to JSON strings
        results_df['entity_mentions'] = results_df['entity_mentions'].apply(lambda x: json.dumps(x))
        results_df['keywords'] = results_df['keywords'].apply(lambda x: json.dumps(x))
        
        # Create session
        session = self.Session()
        
        try:
            # Insert or update sentiment results
            for _, row in results_df.iterrows():
                # Check if record exists
                query = text("""
                SELECT 1 FROM news_sentiment
                WHERE article_id = :article_id AND symbol = :symbol
                """)
                
                exists = session.execute(query, {
                    'article_id': row['article_id'],
                    'symbol': row['symbol']
                }).fetchone() is not None
                
                if exists:
                    # Update existing record
                    query = text("""
                    UPDATE news_sentiment
                    SET 
                        timestamp = :timestamp,
                        sentiment_score = :sentiment_score,
                        sentiment_label = :sentiment_label,
                        confidence = :confidence,
                        entity_mentions = :entity_mentions::jsonb,
                        keywords = :keywords::jsonb,
                        model_version = :model_version
                    WHERE 
                        article_id = :article_id AND symbol = :symbol
                    """)
                else:
                    # Insert new record
                    query = text("""
                    INSERT INTO news_sentiment (
                        article_id, timestamp, symbol, sentiment_score, 
                        sentiment_label, confidence, entity_mentions, 
                        keywords, model_version
                    ) VALUES (
                        :article_id, :timestamp, :symbol, :sentiment_score,
                        :sentiment_label, :confidence, :entity_mentions::jsonb,
                        :keywords::jsonb, :model_version
                    )
                    """)
                
                session.execute(query, {
                    'article_id': row['article_id'],
                    'timestamp': row['timestamp'],
                    'symbol': row['symbol'],
                    'sentiment_score': row['sentiment_score'],
                    'sentiment_label': row['sentiment_label'],
                    'confidence': row['confidence'],
                    'entity_mentions': row['entity_mentions'],
                    'keywords': row['keywords'],
                    'model_version': row['model_version']
                })
            
            # Commit changes
            session.commit()
            return len(results_df)
            
        except Exception as e:
            logger.error(f"Error storing sentiment results: {e}")
            session.rollback()
            return 0
            
        finally:
            session.close()
    
    def get_sentiment_for_symbol(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Get sentiment analysis results for a symbol.
        
        Args:
            symbol: Symbol to get sentiment for
            start_date: Start date
            end_date: End date
            limit: Maximum number of results to return
            
        Returns:
            DataFrame with sentiment analysis results
        """
        query = """
        SELECT 
            s.article_id,
            s.timestamp,
            s.symbol,
            s.sentiment_score,
            s.sentiment_label,
            s.confidence,
            s.entity_mentions,
            s.keywords,
            s.model_version,
            a.title,
            a.article_url,
            a.source
        FROM 
            news_sentiment s
        JOIN 
            news_articles a ON s.article_id = a.article_id
        WHERE 
            s.symbol = :symbol
            AND s.timestamp BETWEEN :start_date AND :end_date
        ORDER BY 
            s.timestamp DESC
        LIMIT :limit
        """
        
        params = {
            "symbol": symbol,
            "start_date": start_date,
            "end_date": end_date,
            "limit": limit
        }
        
        try:
            df = pd.read_sql(query, self.db_engine, params=params)
            return df
        except Exception as e:
            logger.error(f"Error getting sentiment for symbol {symbol}: {e}")
            return pd.DataFrame()
    
    def get_sentiment_features(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        window_sizes: List[int] = [1, 3, 7, 14, 30]
    ) -> pd.DataFrame:
        """
        Calculate sentiment features for symbols.
        
        Args:
            symbols: List of symbols to calculate features for
            start_date: Start date
            end_date: End date
            window_sizes: List of window sizes in days
            
        Returns:
            DataFrame with sentiment features
        """
        # Extend start date to include enough history for the largest window
        extended_start = start_date - timedelta(days=max(window_sizes))
        
        # Get sentiment data for all symbols
        query = """
        SELECT 
            s.timestamp,
            s.symbol,
            s.sentiment_score,
            s.confidence
        FROM 
            news_sentiment s
        WHERE 
            s.symbol IN :symbols
            AND s.timestamp BETWEEN :start_date AND :end_date
        ORDER BY 
            s.timestamp ASC
        """
        
        params = {
            "symbols": tuple(symbols),
            "start_date": extended_start,
            "end_date": end_date
        }
        
        try:
            df = pd.read_sql(query, self.db_engine, params=params)
            
            if df.empty:
                logger.warning(f"No sentiment data found for symbols {symbols}")
                return pd.DataFrame()
            
            # Calculate features for each symbol
            features = []
            
            for symbol in symbols:
                symbol_df = df[df['symbol'] == symbol].copy()
                
                if symbol_df.empty:
                    continue
                
                # Set timestamp as index
                symbol_df.set_index('timestamp', inplace=True)
                symbol_df.sort_index(inplace=True)
                
                # Resample to daily frequency
                daily_df = symbol_df.resample('D').agg({
                    'sentiment_score': 'mean',
                    'confidence': 'mean'
                }).fillna(method='ffill')
                
                # Calculate features for each day in the target range
                date_range = pd.date_range(start=start_date, end=end_date, freq='D')
                
                for date in date_range:
                    # Calculate features for each window size
                    for window in window_sizes:
                        window_start = date - timedelta(days=window)
                        window_data = daily_df.loc[window_start:date]
                        
                        if window_data.empty:
                            continue
                        
                        # Calculate features
                        avg_sentiment = window_data['sentiment_score'].mean()
                        sentiment_std = window_data['sentiment_score'].std()
                        sentiment_min = window_data['sentiment_score'].min()
                        sentiment_max = window_data['sentiment_score'].max()
                        sentiment_change = window_data['sentiment_score'].iloc[-1] - window_data['sentiment_score'].iloc[0] if len(window_data) > 1 else 0
                        avg_confidence = window_data['confidence'].mean()
                        
                        # Add features to list
                        features.append({
                            'timestamp': date,
                            'symbol': symbol,
                            'feature_name': f'sentiment_avg_{window}d',
                            'feature_value': avg_sentiment,
                            'timeframe': '1d',
                            'feature_group': 'sentiment'
                        })
                        
                        features.append({
                            'timestamp': date,
                            'symbol': symbol,
                            'feature_name': f'sentiment_std_{window}d',
                            'feature_value': sentiment_std,
                            'timeframe': '1d',
                            'feature_group': 'sentiment'
                        })
                        
                        features.append({
                            'timestamp': date,
                            'symbol': symbol,
                            'feature_name': f'sentiment_min_{window}d',
                            'feature_value': sentiment_min,
                            'timeframe': '1d',
                            'feature_group': 'sentiment'
                        })
                        
                        features.append({
                            'timestamp': date,
                            'symbol': symbol,
                            'feature_name': f'sentiment_max_{window}d',
                            'feature_value': sentiment_max,
                            'timeframe': '1d',
                            'feature_group': 'sentiment'
                        })
                        
                        features.append({
                            'timestamp': date,
                            'symbol': symbol,
                            'feature_name': f'sentiment_change_{window}d',
                            'feature_value': sentiment_change,
                            'timeframe': '1d',
                            'feature_group': 'sentiment'
                        })
                        
                        features.append({
                            'timestamp': date,
                            'symbol': symbol,
                            'feature_name': f'sentiment_confidence_{window}d',
                            'feature_value': avg_confidence,
                            'timeframe': '1d',
                            'feature_group': 'sentiment'
                        })
            
            # Create DataFrame from features
            features_df = pd.DataFrame(features)
            return features_df
            
        except Exception as e:
            logger.error(f"Error calculating sentiment features: {e}")
            return pd.DataFrame()
    
    def store_sentiment_features(self, features_df: pd.DataFrame) -> int:
        """
        Store sentiment features in the database.
        
        Args:
            features_df: DataFrame with sentiment features
            
        Returns:
            Number of features stored
        """
        if features_df.empty:
            return 0
        
        # Create session
        session = self.Session()
        
        try:
            # Insert features
            for _, row in features_df.iterrows():
                # Check if feature exists
                query = text("""
                SELECT 1 FROM features
                WHERE 
                    timestamp = :timestamp
                    AND symbol = :symbol
                    AND feature_name = :feature_name
                    AND timeframe = :timeframe
                """)
                
                exists = session.execute(query, {
                    'timestamp': row['timestamp'],
                    'symbol': row['symbol'],
                    'feature_name': row['feature_name'],
                    'timeframe': row['timeframe']
                }).fetchone() is not None
                
                if exists:
                    # Update existing feature
                    query = text("""
                    UPDATE features
                    SET feature_value = :feature_value
                    WHERE 
                        timestamp = :timestamp
                        AND symbol = :symbol
                        AND feature_name = :feature_name
                        AND timeframe = :timeframe
                    """)
                else:
                    # Insert new feature
                    query = text("""
                    INSERT INTO features (
                        timestamp, symbol, feature_name, feature_value, 
                        timeframe, feature_group
                    ) VALUES (
                        :timestamp, :symbol, :feature_name, :feature_value,
                        :timeframe, :feature_group
                    )
                    """)
                
                session.execute(query, {
                    'timestamp': row['timestamp'],
                    'symbol': row['symbol'],
                    'feature_name': row['feature_name'],
                    'feature_value': row['feature_value'],
                    'timeframe': row['timeframe'],
                    'feature_group': row['feature_group']
                })
            
            # Commit changes
            session.commit()
            return len(features_df)
            
        except Exception as e:
            logger.error(f"Error storing sentiment features: {e}")
            session.rollback()
            return 0
            
        finally:
            session.close()