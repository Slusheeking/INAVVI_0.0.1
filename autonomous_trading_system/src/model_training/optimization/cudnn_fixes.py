"""
CuDNN Fixes

This module provides fixes for CuDNN compatibility issues, especially for RNN operations.
These fixes ensure stable training of LSTM models in the trading system.
"""

import os
import logging
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Union, Any
import traceback

logger = logging.getLogger(__name__)

class CuDNNFixer:
    """
    Provides fixes for CuDNN compatibility issues.
    
    This class implements:
    1. Fixes for CuDNN compatibility issues with RNN operations
    2. Workarounds for known CuDNN bugs
    3. Performance optimizations for RNN operations
    """
    
    def __init__(
        self,
        enable_cudnn: bool = True,
        deterministic: bool = False,
        rnn_implementation: int = 2,
        lstm_implementation: int = 2
    ):
        """
        Initialize the CuDNNFixer.
        
        Args:
            enable_cudnn: Whether to enable CuDNN
            deterministic: Whether to use deterministic operations
            rnn_implementation: RNN implementation (1 = standard, 2 = CuDNN)
            lstm_implementation: LSTM implementation (1 = standard, 2 = CuDNN)
        """
        self.enable_cudnn = enable_cudnn
        self.deterministic = deterministic
        self.rnn_implementation = rnn_implementation
        self.lstm_implementation = lstm_implementation
        
        # Initialize state
        self.is_initialized = False
        self.original_settings = {}
        
        # Apply fixes
        self.initialize()
        
        logger.info("CuDNNFixer initialized")
    
    def initialize(self) -> bool:
        """
        Initialize and apply all fixes.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check for GPU availability
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                logger.warning("No GPUs available, fixes will have no effect")
                return False
            
            # Store original settings
            self.original_settings = {
                "enable_cudnn": os.environ.get("TF_ENABLE_CUDNN_RNN", "1"),
                "deterministic": os.environ.get("TF_DETERMINISTIC_OPS", "0"),
                "rnn_implementation": os.environ.get("TF_RNN_IMPLEMENTATION", "2"),
                "lstm_implementation": os.environ.get("TF_LSTM_IMPLEMENTATION", "2")
            }
            
            # Apply CuDNN settings
            self._apply_cudnn_settings()
            
            # Apply deterministic settings
            if self.deterministic:
                self._apply_deterministic_settings()
            
            # Apply RNN implementation settings
            self._apply_rnn_implementation_settings()
            
            # Apply known bug fixes
            self._apply_bug_fixes()
            
            self.is_initialized = True
            logger.info("CuDNN fixes applied successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing CuDNN fixer: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def _apply_cudnn_settings(self) -> None:
        """Apply CuDNN settings."""
        try:
            os.environ["TF_ENABLE_CUDNN_RNN"] = "1" if self.enable_cudnn else "0"
            logger.info(f"CuDNN for RNN operations: {'enabled' if self.enable_cudnn else 'disabled'}")
            
            # Apply additional CuDNN optimizations
            if self.enable_cudnn:
                os.environ["TF_CUDNN_WORKSPACE_LIMIT_IN_MB"] = "1024"  # 1GB workspace limit
                os.environ["TF_CUDNN_USE_AUTOTUNE"] = "1"  # Enable autotuning
            
        except Exception as e:
            logger.error(f"Error applying CuDNN settings: {e}")
    
    def _apply_deterministic_settings(self) -> None:
        """Apply deterministic settings for reproducibility."""
        try:
            # Set deterministic operations
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
            
            # Set seed for reproducibility
            tf.random.set_seed(42)
            
            # Disable parallel execution
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)
            
            logger.info("Deterministic operations enabled for reproducibility")
        except Exception as e:
            logger.error(f"Error applying deterministic settings: {e}")
    
    def _apply_rnn_implementation_settings(self) -> None:
        """Apply RNN implementation settings."""
        try:
            os.environ["TF_RNN_IMPLEMENTATION"] = str(self.rnn_implementation)
            os.environ["TF_LSTM_IMPLEMENTATION"] = str(self.lstm_implementation)
            
            logger.info(f"RNN implementation set to {self.rnn_implementation}")
            logger.info(f"LSTM implementation set to {self.lstm_implementation}")
        except Exception as e:
            logger.error(f"Error applying RNN implementation settings: {e}")
    
    def _apply_bug_fixes(self) -> None:
        """Apply fixes for known CuDNN bugs."""
        try:
            # Fix for CuDNN crash with small batch sizes
            os.environ["TF_MIN_GPU_MULTIPROCESSOR_COUNT"] = "4"
            
            # Fix for CuDNN memory leak in RNN operations
            os.environ["TF_CUDNN_RESET_RNN_DESCRIPTOR"] = "1"
            
            # Fix for CuDNN compatibility issues with certain GPU architectures
            os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
            
            logger.info("Applied fixes for known CuDNN bugs")
        except Exception as e:
            logger.error(f"Error applying bug fixes: {e}")
    
    def create_lstm_layer(
        self,
        units: int,
        return_sequences: bool = False,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        **kwargs
    ) -> tf.keras.layers.LSTM:
        """
        Create an LSTM layer with CuDNN compatibility.
        
        Args:
            units: Number of units in the LSTM layer
            return_sequences: Whether to return the full sequence
            dropout: Dropout rate
            recurrent_dropout: Recurrent dropout rate
            **kwargs: Additional arguments for the LSTM layer
            
        Returns:
            LSTM layer with CuDNN compatibility
        """
        try:
            # Check if CuDNN is enabled
            if not self.enable_cudnn:
                logger.info("Creating standard LSTM layer (CuDNN disabled)")
                return tf.keras.layers.LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    **kwargs
                )
            
            # Check if dropout or recurrent_dropout is used
            if dropout > 0.0 or recurrent_dropout > 0.0:
                logger.info("Creating standard LSTM layer (dropout/recurrent_dropout used)")
                return tf.keras.layers.LSTM(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    **kwargs
                )
            
            # Create CuDNN-compatible LSTM layer
            logger.info("Creating CuDNN-compatible LSTM layer")
            return tf.keras.layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                activation='tanh',
                recurrent_activation='sigmoid',
                use_bias=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                bias_initializer='zeros',
                unit_forget_bias=True,
                kernel_regularizer=None,
                recurrent_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                recurrent_constraint=None,
                bias_constraint=None,
                dropout=0.0,
                recurrent_dropout=0.0,
                implementation=self.lstm_implementation,
                **kwargs
            )
            
        except Exception as e:
            logger.error(f"Error creating LSTM layer: {e}")
            # Fall back to standard LSTM layer
            return tf.keras.layers.LSTM(
                units=units,
                return_sequences=return_sequences,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                **kwargs
            )
    
    def create_gru_layer(
        self,
        units: int,
        return_sequences: bool = False,
        dropout: float = 0.0,
        recurrent_dropout: float = 0.0,
        **kwargs
    ) -> tf.keras.layers.GRU:
        """
        Create a GRU layer with CuDNN compatibility.
        
        Args:
            units: Number of units in the GRU layer
            return_sequences: Whether to return the full sequence
            dropout: Dropout rate
            recurrent_dropout: Recurrent dropout rate
            **kwargs: Additional arguments for the GRU layer
            
        Returns:
            GRU layer with CuDNN compatibility
        """
        try:
            # Check if CuDNN is enabled
            if not self.enable_cudnn:
                logger.info("Creating standard GRU layer (CuDNN disabled)")
                return tf.keras.layers.GRU(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    **kwargs
                )
            
            # Check if dropout or recurrent_dropout is used
            if dropout > 0.0 or recurrent_dropout > 0.0:
                logger.info("Creating standard GRU layer (dropout/recurrent_dropout used)")
                return tf.keras.layers.GRU(
                    units=units,
                    return_sequences=return_sequences,
                    dropout=dropout,
                    recurrent_dropout=recurrent_dropout,
                    **kwargs
                )
            
            # Create CuDNN-compatible GRU layer
            logger.info("Creating CuDNN-compatible GRU layer")
            return tf.keras.layers.GRU(
                units=units,
                return_sequences=return_sequences,
                activation='tanh',
                recurrent_activation='sigmoid',
                use_bias=True,
                kernel_initializer='glorot_uniform',
                recurrent_initializer='orthogonal',
                bias_initializer='zeros',
                kernel_regularizer=None,
                recurrent_regularizer=None,
                bias_regularizer=None,
                activity_regularizer=None,
                kernel_constraint=None,
                recurrent_constraint=None,
                bias_constraint=None,
                dropout=0.0,
                recurrent_dropout=0.0,
                implementation=self.rnn_implementation,
                reset_after=True,  # Required for CuDNN compatibility
                **kwargs
            )
            
        except Exception as e:
            logger.error(f"Error creating GRU layer: {e}")
            # Fall back to standard GRU layer
            return tf.keras.layers.GRU(
                units=units,
                return_sequences=return_sequences,
                dropout=dropout,
                recurrent_dropout=recurrent_dropout,
                **kwargs
            )
    
    def cleanup(self) -> None:
        """Restore original settings."""
        try:
            # Restore original settings
            for key, value in self.original_settings.items():
                if key == "enable_cudnn":
                    os.environ["TF_ENABLE_CUDNN_RNN"] = value
                elif key == "deterministic":
                    os.environ["TF_DETERMINISTIC_OPS"] = value
                elif key == "rnn_implementation":
                    os.environ["TF_RNN_IMPLEMENTATION"] = value
                elif key == "lstm_implementation":
                    os.environ["TF_LSTM_IMPLEMENTATION"] = value
            
            logger.info("Restored original CuDNN settings")
        except Exception as e:
            logger.error(f"Error cleaning up CuDNN fixer: {e}")