"""
Redis Storage Module

This module provides functionality for storing and retrieving data from Redis,
which is used for caching and real-time operations in the Autonomous Trading System.
"""

import logging
import json
import time
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import uuid
import redis
import pandas as pd

from autonomous_trading_system.src.config.database_config import get_redis_connection_params

logger = logging.getLogger(__name__)

class RedisStorage:
    """Class for storing and retrieving data from Redis."""
    
    def __init__(self, connection_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the Redis storage.
        
        Args:
            connection_params: Redis connection parameters (defaults to config)
        """
        self.connection_params = connection_params or get_redis_connection_params()
        self.redis = redis.Redis(**self.connection_params)

        # Configure Redis persistence settings
        self._configure_persistence()
        
        # Test connection
        self._test_connection()
    
    def _configure_persistence(self) -> None:
        """Configure Redis persistence settings."""
        try:
            self.redis.config_set('save', '900 1 300 10 60 10000')  # RDB persistence
            self.redis.config_set('appendonly', 'yes')  # AOF persistence
        except redis.ResponseError:
            logger.warning("Unable to configure Redis persistence (likely running in protected mode)")
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self) -> None:
        """
        Test the Redis connection.
        
        Raises:
            redis.ConnectionError: If the connection fails
        """
        try:
            self.redis.ping()
            logger.info("Successfully connected to Redis")
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise
    
    # Feature Storage Methods
    def store_feature(self, symbol: str, feature_name: str, feature_value: float, 
                     timeframe: str = '1m', timestamp: Optional[datetime] = None) -> None:
        """
        Store a feature value in Redis.
        
        Args:
            symbol: Ticker symbol
            feature_name: Name of the feature
            feature_value: Value of the feature
            timeframe: Data timeframe
            timestamp: Optional timestamp (defaults to current time)
        """
        timestamp = timestamp or datetime.now()
        timestamp_str = timestamp.isoformat()
        
        # Store latest feature value
        key = f"feature:{symbol}:{feature_name}:{timeframe}"
        value = json.dumps({
            'value': feature_value,
            'timestamp': timestamp_str
        })
        self.redis.set(key, value)
        
        # Store in history (sorted set)
        history_key = f"feature_history:{symbol}:{feature_name}:{timeframe}"
        score = timestamp.timestamp()
        self.redis.zadd(history_key, {value: score})
        
        # Trim history to keep only recent values (e.g., last 100)
        self.redis.zremrangebyrank(history_key, 0, -101)
        
        logger.debug(f"Stored feature {feature_name} for {symbol} ({timeframe})")
    
    def get_feature(self, symbol: str, feature_name: str, timeframe: str = '1m') -> Optional[Dict[str, Any]]:
        """
        Get the latest feature value from Redis.
        
        Args:
            symbol: Ticker symbol
            feature_name: Name of the feature
            timeframe: Data timeframe
            
        Returns:
            Dictionary with feature value and timestamp, or None if not found
        """
        key = f"feature:{symbol}:{feature_name}:{timeframe}"
        value = self.redis.get(key)
        
        if value:
            return json.loads(value)
        
        return None
    
    def get_feature_history(self, symbol: str, feature_name: str, timeframe: str = '1m', 
                           limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get the history of a feature from Redis.
        
        Args:
            symbol: Ticker symbol
            feature_name: Name of the feature
            timeframe: Data timeframe
            limit: Maximum number of values to return
            
        Returns:
            List of dictionaries with feature values and timestamps
        """
        history_key = f"feature_history:{symbol}:{feature_name}:{timeframe}"
        values = self.redis.zrevrange(history_key, 0, limit - 1, withscores=True)
        
        result = []
        for value, score in values:
            data = json.loads(value)
            data['score'] = score
            result.append(data)
        
        return result
    
    def store_feature_metadata(self, feature_name: str, metadata: Dict[str, Any]) -> None:
        """
        Store feature metadata in Redis.
        
        Args:
            feature_name: Name of the feature
            metadata: Feature metadata
        """
        key = f"feature_metadata:{feature_name}"
        
        # Add timestamps if not present
        if 'created_at' not in metadata:
            metadata['created_at'] = datetime.now().isoformat()
        if 'updated_at' not in metadata:
            metadata['updated_at'] = datetime.now().isoformat()
        
        # Convert any non-string values to strings
        metadata_str = {k: str(v) if not isinstance(v, str) else v for k, v in metadata.items()}
        
        self.redis.hmset(key, metadata_str)
        
        # Add to feature registry
        self.redis.sadd('feature_registry', feature_name)
        
        logger.debug(f"Stored metadata for feature {feature_name}")
    
    def get_feature_metadata(self, feature_name: str) -> Dict[str, Any]:
        """
        Get feature metadata from Redis.
        
        Args:
            feature_name: Name of the feature
            
        Returns:
            Dictionary with feature metadata
        """
        key = f"feature_metadata:{feature_name}"
        metadata = self.redis.hgetall(key)
        
        # Convert bytes to strings
        return {k.decode('utf-8'): v.decode('utf-8') for k, v in metadata.items()}
    
    def get_all_features(self) -> List[str]:
        """
        Get all registered features from Redis.
        
        Returns:
            List of feature names
        """
        features = self.redis.smembers('feature_registry')
        return [f.decode('utf-8') for f in features]
    
    # Model Storage Methods
    def store_model_prediction(self, model_id: str, symbol: str, prediction: float, 
                              timestamp: Optional[datetime] = None) -> None:
        """
        Store a model prediction in Redis.
        
        Args:
            model_id: Model identifier
            symbol: Ticker symbol
            prediction: Prediction value
            timestamp: Optional timestamp (defaults to current time)
        """
        timestamp = timestamp or datetime.now()
        timestamp_str = timestamp.isoformat()
        
        # Store in sorted set
        key = f"model:{model_id}:predictions:{symbol}"
        value = json.dumps({
            'prediction': prediction,
            'timestamp': timestamp_str
        })
        score = timestamp.timestamp()
        self.redis.zadd(key, {value: score})
        
        # Trim to keep only recent predictions (e.g., last 100)
        self.redis.zremrangebyrank(key, 0, -101)
        
        logger.debug(f"Stored prediction for model {model_id}, symbol {symbol}")
    
    def get_model_predictions(self, model_id: str, symbol: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get model predictions from Redis.
        
        Args:
            model_id: Model identifier
            symbol: Ticker symbol
            limit: Maximum number of predictions to return
            
        Returns:
            List of dictionaries with predictions and timestamps
        """
        key = f"model:{model_id}:predictions:{symbol}"
        values = self.redis.zrevrange(key, 0, limit - 1, withscores=True)
        
        result = []
        for value, score in values:
            data = json.loads(value)
            data['score'] = score
            result.append(data)
        
        return result
    
    def store_model_metadata(self, model_id: str, metadata: Dict[str, Any]) -> None:
        """
        Store model metadata in Redis.
        
        Args:
            model_id: Model identifier
            metadata: Model metadata
        """
        key = f"model:{model_id}:metadata"
        self.redis.hmset(key, metadata)
        logger.debug(f"Stored metadata for model {model_id}")
    
    def get_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """
        Get model metadata from Redis.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Dictionary with model metadata
        """
        key = f"model:{model_id}:metadata"
        metadata = self.redis.hgetall(key)
        
        # Convert bytes to strings
        return {k.decode('utf-8'): v.decode('utf-8') for k, v in metadata.items()}
    
    def store_model_training_run(self, run_id: str, model_id: str, status: str, 
                               parameters: Dict[str, Any], start_time: Optional[datetime] = None) -> None:
        """
        Store model training run in Redis.
        
        Args:
            run_id: Run identifier
            model_id: Model identifier
            status: Run status
            parameters: Training parameters
            start_time: Optional start time (defaults to current time)
        """
        start_time = start_time or datetime.now()
        key = f"model_training_run:{run_id}"
        
        data = {
            'model_id': model_id,
            'status': status,
            'parameters': json.dumps(parameters),
            'start_time': start_time.isoformat()
        }
        
        self.redis.hmset(key, data)
        
        # Add to model's training runs set
        self.redis.sadd(f"model:{model_id}:training_runs", run_id)
        
        logger.debug(f"Stored training run {run_id} for model {model_id}")
    
    def update_model_training_run(self, run_id: str, status: str, 
                                metrics: Optional[Dict[str, Any]] = None,
                                end_time: Optional[datetime] = None) -> None:
        """
        Update model training run in Redis.
        
        Args:
            run_id: Run identifier
            status: New status
            metrics: Optional training metrics
            end_time: Optional end time (defaults to current time if status is terminal)
        """
        key = f"model_training_run:{run_id}"
        
        updates = {'status': status}
        
        if metrics:
            updates['metrics'] = json.dumps(metrics)
        
        if status in ['completed', 'failed', 'stopped'] or end_time:
            updates['end_time'] = (end_time or datetime.now()).isoformat()
        
        self.redis.hmset(key, updates)
        logger.debug(f"Updated training run {run_id} status to {status}")
    
    def get_model_training_run(self, run_id: str) -> Dict[str, Any]:
        """
        Get model training run from Redis.
        
        Args:
            run_id: Run identifier
            
        Returns:
            Dictionary with training run data
        """
        key = f"model_training_run:{run_id}"
        run_data = self.redis.hgetall(key)
        
        # Convert bytes to strings
        run_data = {k.decode('utf-8'): v.decode('utf-8') for k, v in run_data.items()}
        
        # Parse JSON fields
        if 'parameters' in run_data:
            run_data['parameters'] = json.loads(run_data['parameters'])
        if 'metrics' in run_data:
            run_data['metrics'] = json.loads(run_data['metrics'])
            
        return run_data
    
    # Trading Signal Methods
    def store_trading_signal(self, signal_id: str, symbol: str, signal_type: str, 
                           confidence: float, timeframe: str = '1d',
                           model_id: Optional[str] = None,
                           timestamp: Optional[datetime] = None) -> None:
        """
        Store trading signal in Redis.
        
        Args:
            signal_id: Signal identifier
            symbol: Ticker symbol
            signal_type: Type of signal (e.g., 'buy', 'sell')
            confidence: Signal confidence (0-1)
            timeframe: Data timeframe
            model_id: Optional model identifier
            timestamp: Optional timestamp (defaults to current time)
        """
        timestamp = timestamp or datetime.now()
        key = f"signal:{signal_id}"
        
        data = {
            'symbol': symbol,
            'signal_type': signal_type,
            'confidence': str(confidence),
            'timeframe': timeframe,
            'timestamp': timestamp.isoformat()
        }
        
        if model_id:
            data['model_id'] = model_id
        
        self.redis.hmset(key, data)
        
        # Add to active signals set
        self.redis.sadd('active_signals', signal_id)
        
        # Add to symbol's signals set
        self.redis.sadd(f"symbol:{symbol}:signals", signal_id)
        
        # Set expiration (signals expire after 24 hours)
        self.redis.expire(key, 86400)  # 24 hours in seconds
        
        logger.debug(f"Stored trading signal {signal_id} for {symbol}")
    
    def get_trading_signal(self, signal_id: str) -> Dict[str, Any]:
        """
        Get trading signal from Redis.
        
        Args:
            signal_id: Signal identifier
            
        Returns:
            Dictionary with signal data
        """
        key = f"signal:{signal_id}"
        signal = self.redis.hgetall(key)
        
        # Convert bytes to strings
        return {k.decode('utf-8'): v.decode('utf-8') for k, v in signal.items()}
    
    def get_active_signals(self, symbol: Optional[str] = None) -> List[str]:
        """
        Get active trading signals from Redis.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            List of signal IDs
        """
        if symbol:
            signals = self.redis.smembers(f"symbol:{symbol}:signals")
        else:
            signals = self.redis.smembers('active_signals')
            
        return [s.decode('utf-8') for s in signals]
    
    def remove_trading_signal(self, signal_id: str) -> None:
        """
        Remove a trading signal from Redis.
        
        Args:
            signal_id: Signal identifier
        """
        # Get signal data to find the symbol
        signal_data = self.get_trading_signal(signal_id)
        symbol = signal_data.get('symbol')
        
        # Remove from active signals set
        self.redis.srem('active_signals', signal_id)
        
        # Remove from symbol's signals set if symbol is available
        if symbol:
            self.redis.srem(f"symbol:{symbol}:signals", signal_id)
        
        # Delete the signal key
        key = f"signal:{signal_id}"
        self.redis.delete(key)
        
        logger.debug(f"Removed trading signal {signal_id}")
    
    # Position and Order Methods
    def store_position(self, symbol: str, position_data: Dict[str, Any]) -> None:
        """
        Store position data in Redis.
        
        Args:
            symbol: Ticker symbol
            position_data: Position data
        """
        # Ensure position_id is present
        if 'position_id' not in position_data:
            position_data['position_id'] = str(uuid.uuid4())
        
        # Ensure timestamps are in ISO format
        if 'entry_time' in position_data and not isinstance(position_data['entry_time'], str):
            position_data['entry_time'] = position_data['entry_time'].isoformat()
        if 'last_update' in position_data and not isinstance(position_data['last_update'], str):
            position_data['last_update'] = position_data['last_update'].isoformat()
        
        key = f"position:{symbol}"
        self.redis.hmset(key, position_data)
        
        # Add to active positions set
        self.redis.sadd('active_positions', symbol)
        
        logger.debug(f"Stored position for {symbol}")
    
    def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        Get position data from Redis.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Dictionary with position data
        """
        key = f"position:{symbol}"
        position = self.redis.hgetall(key)
        
        # Convert bytes to strings
        return {k.decode('utf-8'): v.decode('utf-8') for k, v in position.items()}
    
    def get_active_positions(self) -> List[str]:
        """
        Get active positions from Redis.
        
        Returns:
            List of symbols with active positions
        """
        positions = self.redis.smembers('active_positions')
        return [p.decode('utf-8') for p in positions]
    
    def remove_position(self, symbol: str) -> None:
        """
        Remove a position from Redis.
        
        Args:
            symbol: Ticker symbol
        """
        key = f"position:{symbol}"
        self.redis.delete(key)
        
        # Remove from active positions set
        self.redis.srem('active_positions', symbol)
        
        logger.debug(f"Removed position for {symbol}")
    
    def store_order(self, order_id: str, order_data: Dict[str, Any]) -> None:
        """
        Store order data in Redis.
        
        Args:
            order_id: Order identifier
            order_data: Order data
        """
        # Ensure timestamps are in ISO format
        if 'timestamp' in order_data and not isinstance(order_data['timestamp'], str):
            order_data['timestamp'] = order_data['timestamp'].isoformat()
        if 'updated_at' in order_data and not isinstance(order_data['updated_at'], str):
            order_data['updated_at'] = order_data['updated_at'].isoformat()
        
        # Convert numeric values to strings
        for k, v in order_data.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                order_data[k] = str(v)
        
        key = f"order:{order_id}"
        self.redis.hmset(key, order_data)
        
        # Add to pending orders set if status is pending
        if order_data.get('status') in ['pending', 'open', 'new']:
            self.redis.sadd('pending_orders', order_id)
        
        logger.debug(f"Stored order {order_id}")
    
    def get_order(self, order_id: str) -> Dict[str, Any]:
        """
        Get order data from Redis.
        
        Args:
            order_id: Order identifier
            
        Returns:
            Dictionary with order data
        """
        key = f"order:{order_id}"
        order = self.redis.hgetall(key)
        
        # Convert bytes to strings
        return {k.decode('utf-8'): v.decode('utf-8') for k, v in order.items()}
    
    def get_pending_orders(self) -> List[str]:
        """
        Get pending orders from Redis.
        
        Returns:
            List of pending order IDs
        """
        orders = self.redis.smembers('pending_orders')
        return [o.decode('utf-8') for o in orders]
    
    def update_order_status(self, order_id: str, status: str) -> None:
        """
        Update order status in Redis.
        
        Args:
            order_id: Order identifier
            status: New status
        """
        key = f"order:{order_id}"
        self.redis.hset(key, 'status', status)
        
        # Update pending orders set
        if status in ['filled', 'canceled', 'rejected', 'expired']:
            self.redis.srem('pending_orders', order_id)
        elif status in ['pending', 'open', 'new']:
            self.redis.sadd('pending_orders', order_id)
        
        logger.debug(f"Updated status for order {order_id} to {status}")
    
    def store_system_status(self, status: Dict[str, Any]) -> None:
        """
        Store system status in Redis.
        
        Args:
            status: System status data
        """
        self.redis.hmset('system:status', status)
        
        # Store timestamp of last update
        self.redis.hset('system:status', 'last_updated', datetime.now().isoformat())
        
        # Set expiration (status expires after 1 hour)
        self.redis.expire('system:status', 3600)  # 1 hour in seconds
        logger.debug("Stored system status")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status from Redis.
        
        Returns:
            Dictionary with system status
        """
        status = self.redis.hgetall('system:status')
        
        # Convert bytes to strings
        return {k.decode('utf-8'): v.decode('utf-8') for k, v in status.items()}
    
    def store_component_status(self, component_name: str, status: Dict[str, Any]) -> None:
        """
        Store component status in Redis.
        
        Args:
            component_name: Name of the component
            status: Component status data
        """
        key = f"component:{component_name}:status"
        
        # Add timestamp of last update
        status['last_updated'] = datetime.now().isoformat()
        
        # Store status
        self.redis.hmset(key, status)
        logger.debug(f"Stored status for component {component_name}")
    
    def get_component_status(self, component_name: str) -> Dict[str, Any]:
        """
        Get component status from Redis.
        
        Args:
            component_name: Name of the component
            
        Returns:
            Dictionary with component status
        """
        key = f"component:{component_name}:status"
        status = self.redis.hgetall(key)
        
        # Convert bytes to strings
        return {k.decode('utf-8'): v.decode('utf-8') for k, v in status.items()}
    
    # Emergency Status Methods
    def set_emergency_status(self, is_emergency: bool) -> None:
        """
        Set emergency status flag in Redis.
        
        Args:
            is_emergency: Whether there is an emergency
        """
        self.redis.set('emergency:status', str(is_emergency).lower())
        
        # Store timestamp of status change
        self.redis.set('emergency:status:timestamp', datetime.now().isoformat())
        
        # Set expiration (emergency status never expires, must be explicitly cleared)
        logger.debug(f"Set emergency status to {is_emergency}")
    
    def get_emergency_status(self) -> bool:
        """
        Get emergency status flag from Redis.
        
        Returns:
            Whether there is an emergency
        """
        status = self.redis.get('emergency:status')
        return status is not None and status.decode('utf-8') == 'true'
    
    # WebSocket Data Methods
    def store_websocket_data(self, data_type: str, symbol: str, data: Dict[str, Any]) -> None:
        """
        Store WebSocket data in Redis.
        
        Args:
            data_type: Type of data ('trade', 'quote', 'agg')
            symbol: Ticker symbol
            data: WebSocket data
        """
        key = f"ws:{data_type}:{symbol}"
        self.redis.set(key, json.dumps(data))
        
        # Set expiration (WebSocket data expires after 5 minutes)
        self.redis.expire(key, 300)  # 5 minutes in seconds
        logger.debug(f"Stored WebSocket {data_type} data for {symbol}")
    
    def get_websocket_data(self, data_type: str, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get WebSocket data from Redis.
        
        Args:
            data_type: Type of data ('trade', 'quote', 'agg')
            symbol: Ticker symbol
            
        Returns:
            Dictionary with WebSocket data, or None if not found
        """
        key = f"ws:{data_type}:{symbol}"
        data = self.redis.get(key)
        
        if data:
            return json.loads(data)
        
        return None
    
    # Messaging Methods
    def publish_message(self, channel: str, message: Dict[str, Any]) -> int:
        """
        Publish a message to a Redis channel.
        
        Args:
            channel: Channel name
            message: Message to publish
            
        Returns:
            Number of clients that received the message
        """
        return self.redis.publish(channel, json.dumps(message))
    
    def subscribe_to_channel(self, channel: str) -> redis.client.PubSub:
        """
        Subscribe to a Redis channel.
        
        Args:
            channel: Channel name
            
        Returns:
            PubSub object for receiving messages
        """
        pubsub = self.redis.pubsub()
        pubsub.subscribe(channel)
        return pubsub
    
    # Database Management Methods
    def get_info(self) -> Dict[str, Any]:
        """Get Redis server information."""
        info = self.redis.info()
        return info
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get Redis memory statistics."""
        memory_stats = self.redis.info('memory')
        return memory_stats
    
    def flush_db(self) -> None:
        """
        Flush the Redis database (delete all keys).
        WARNING: This will delete all data in Redis.
        """
        self.redis.flushdb()
        logger.warning("Flushed Redis database")
    
    def close(self) -> None:
        """Close the Redis connection."""
        self.redis.close()
        logger.info("Redis connection closed")