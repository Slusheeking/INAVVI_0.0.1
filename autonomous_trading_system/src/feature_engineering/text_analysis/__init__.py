"""
Text Analysis Module

This module provides components for analyzing text data, particularly financial news
and reports, to extract sentiment, entities, and other features that can be used in
trading strategies.

Components:
- FinBERTSentimentAnalyzer: Analyzes sentiment in financial text using FinBERT
"""

from .finbert_sentiment import FinBERTSentimentAnalyzer

__all__ = ['FinBERTSentimentAnalyzer']