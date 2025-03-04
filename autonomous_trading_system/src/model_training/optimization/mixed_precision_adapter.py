"""
Mixed Precision Adapter

This module provides mixed precision training capabilities for deep learning models,
which can significantly improve performance on GPUs with Tensor Cores (e.g., NVIDIA Volta, Turing, Ampere, and Hopper architectures).
"""

import os
import logging
import tensorflow as tf
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import traceback

logger = logging.getLogger(__name__)

class MixedPrecisionAdapter:
    """
    Provides mixed precision training capabilities for deep learning models.
    
    This class implements:
    1. Automatic mixed precision (AMP) for TensorFlow models
    2. Loss scaling to prevent underflow in gradients
    3. Dynamic loss scaling for stable training
    4. Utilities for converting models to mixed precision
    """
    
    def __init__(
        self,
        dtype: str = 'float16',
        loss_scale: Union[str, float] = 'dynamic',
        initial_scale: float = 2**15,
        dynamic_growth_steps: int = 2000
    ):
        """
        Initialize the MixedPrecisionAdapter.
        
        Args:
            dtype: Data type for mixed precision ('float16', 'bfloat16')
            loss_scale: Loss scaling method ('dynamic' or a fixed value)
            initial_scale: Initial loss scale value for dynamic scaling
            dynamic_growth_steps: Steps between loss scale increases
        """
        self.dtype = dtype
        self.loss_scale = loss_scale
        self.initial_scale = initial_scale
        self.dynamic_growth_steps = dynamic_growth_steps
        
        # Initialize state
        self.is_initialized = False
        self.original_policy = None
        self.optimizer_wrapper = None
        
        # Apply mixed precision
        self.initialize()
        
        logger.info("MixedPrecisionAdapter initialized")
    
    def initialize(self) -> bool:
        """
        Initialize and apply mixed precision settings.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check for GPU availability
            gpus = tf.config.list_physical_devices('GPU')
            if not gpus:
                logger.warning("No GPUs available, mixed precision will have limited benefit")
            
            # Store original policy
            self.original_policy = tf.keras.mixed_precision.global_policy()
            
            # Set mixed precision policy
            if self.dtype == 'float16':
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
                logger.info("Set mixed precision policy to mixed_float16")
            elif self.dtype == 'bfloat16':
                tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
                logger.info("Set mixed precision policy to mixed_bfloat16")
            else:
                logger.warning(f"Unknown dtype: {self.dtype}, using default")
                tf.keras.mixed_precision.set_global_policy('mixed_float16')
            
            self.is_initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Error initializing mixed precision: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def wrap_optimizer(self, optimizer: tf.keras.optimizers.Optimizer) -> tf.keras.optimizers.Optimizer:
        """
        Wrap an optimizer with loss scaling for mixed precision training.
        
        Args:
            optimizer: TensorFlow optimizer to wrap
            
        Returns:
            Wrapped optimizer with loss scaling
        """
        try:
            # Create loss scale
            if self.loss_scale == 'dynamic':
                loss_scale = tf.keras.mixed_precision.LossScaleOptimizer.dynamic(
                    initial_loss_scale=self.initial_scale,
                    increment_period=self.dynamic_growth_steps
                )
            else:
                try:
                    fixed_scale = float(self.loss_scale)
                    loss_scale = tf.keras.mixed_precision.LossScaleOptimizer.fixed(fixed_scale)
                except (ValueError, TypeError):
                    logger.warning(f"Invalid loss_scale: {self.loss_scale}, using dynamic")
                    loss_scale = tf.keras.mixed_precision.LossScaleOptimizer.dynamic(
                        initial_loss_scale=self.initial_scale,
                        increment_period=self.dynamic_growth_steps
                    )
            
            # Wrap optimizer
            wrapped_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(
                optimizer,
                loss_scale=loss_scale
            )
            
            self.optimizer_wrapper = wrapped_optimizer
            logger.info(f"Wrapped optimizer {optimizer.__class__.__name__} with loss scaling")
            
            return wrapped_optimizer
            
        except Exception as e:
            logger.error(f"Error wrapping optimizer: {e}")
            logger.debug(traceback.format_exc())
            return optimizer
    
    def convert_model(self, model: tf.keras.Model) -> tf.keras.Model:
        """
        Convert a model to use mixed precision.
        
        Args:
            model: TensorFlow model to convert
            
        Returns:
            Converted model with mixed precision
        """
        try:
            # Check if model is already using mixed precision
            if model.dtype == tf.float16 or model.dtype == tf.bfloat16:
                logger.info("Model is already using mixed precision")
                return model
            
            # Clone the model with mixed precision policy
            config = model.get_config()
            
            # Save weights
            weights = model.get_weights()
            
            # Create new model with mixed precision
            with tf.keras.mixed_precision.global_policy():
                new_model = model.__class__.from_config(config)
                new_model.build(model.input_shape)
                new_model.set_weights(weights)
            
            logger.info(f"Converted model to mixed precision ({self.dtype})")
            return new_model
            
        except Exception as e:
            logger.error(f"Error converting model: {e}")
            logger.debug(traceback.format_exc())
            return model
    
    def create_mixed_precision_model(
        self,
        model_fn: Callable[..., tf.keras.Model],
        *args,
        **kwargs
    ) -> tf.keras.Model:
        """
        Create a model with mixed precision.
        
        Args:
            model_fn: Function that creates a TensorFlow model
            *args: Arguments for model_fn
            **kwargs: Keyword arguments for model_fn
            
        Returns:
            Model created with mixed precision
        """
        try:
            # Create model with mixed precision policy
            with tf.keras.mixed_precision.global_policy():
                model = model_fn(*args, **kwargs)
            
            logger.info(f"Created model with mixed precision ({self.dtype})")
            return model
            
        except Exception as e:
            logger.error(f"Error creating mixed precision model: {e}")
            logger.debug(traceback.format_exc())
            
            # Fall back to creating model with original policy
            with tf.keras.mixed_precision.policy(self.original_policy):
                model = model_fn(*args, **kwargs)
            
            return model
    
    def get_custom_training_loop(
        self,
        model: tf.keras.Model,
        optimizer: tf.keras.optimizers.Optimizer,
        loss_fn: Callable
    ) -> Callable:
        """
        Get a custom training loop with mixed precision.
        
        Args:
            model: TensorFlow model
            optimizer: TensorFlow optimizer (should be wrapped with loss scaling)
            loss_fn: Loss function
            
        Returns:
            Custom training step function
        """
        # Ensure optimizer is wrapped
        if not isinstance(optimizer, tf.keras.mixed_precision.LossScaleOptimizer):
            optimizer = self.wrap_optimizer(optimizer)
        
        # Create training step function
        @tf.function
        def train_step(x, y):
            with tf.GradientTape() as tape:
                # Forward pass
                y_pred = model(x, training=True)
                
                # Calculate loss
                loss = loss_fn(y, y_pred)
                
                # Scale loss
                scaled_loss = optimizer.get_scaled_loss(loss)
            
            # Calculate gradients
            scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
            
            # Unscale gradients
            gradients = optimizer.get_unscaled_gradients(scaled_gradients)
            
            # Apply gradients
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            return loss
        
        return train_step
    
    def cleanup(self) -> None:
        """Restore original settings."""
        try:
            # Restore original policy
            if self.original_policy is not None:
                tf.keras.mixed_precision.set_global_policy(self.original_policy)
                logger.info(f"Restored original precision policy: {self.original_policy}")
            
        except Exception as e:
            logger.error(f"Error cleaning up mixed precision adapter: {e}")