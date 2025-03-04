"""
GPU Accelerator

This module provides GPU acceleration capabilities for deep learning models,
with specific optimizations for various GPU architectures including the
NVIDIA GH200 Grace Hopper Superchip.
"""

import os
import sys
import logging
import tensorflow as tf
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import traceback

logger = logging.getLogger(__name__)

class GPUAccelerator:
    """
    Provides GPU acceleration capabilities for deep learning models.
    
    This class implements:
    1. Mixed precision training for Tensor Cores
    2. Memory optimizations to prevent OOM errors
    3. XLA compilation for improved performance
    4. Multi-GPU distribution strategies
    5. Specific optimizations for GH200 Grace Hopper Superchip
    """
    
    def __init__(
        self,
        enable_mixed_precision: bool = True,
        enable_xla: bool = True,
        enable_memory_growth: bool = True,
        memory_limit_mb: Optional[int] = None,
        distribution_strategy: str = "mirrored",
        gh200_optimizations: bool = False
    ):
        """
        Initialize the GPUAccelerator.
        
        Args:
            enable_mixed_precision: Whether to enable mixed precision training
            enable_xla: Whether to enable XLA compilation
            enable_memory_growth: Whether to enable memory growth
            memory_limit_mb: Memory limit in MB (None for no limit)
            distribution_strategy: Distribution strategy for multi-GPU training
                                  ("mirrored", "central_storage", or "one_device")
            gh200_optimizations: Whether to enable GH200-specific optimizations
        """
        self.enable_mixed_precision = enable_mixed_precision
        self.enable_xla = enable_xla
        self.enable_memory_growth = enable_memory_growth
        self.memory_limit_mb = memory_limit_mb
        self.distribution_strategy = distribution_strategy
        self.gh200_optimizations = gh200_optimizations
        
        # Initialize state
        self.is_initialized = False
        self.gpus = []
        self.original_policy = None
        self.strategy = None
        
        # Apply optimizations
        self.initialize()
        
        logger.info("GPUAccelerator initialized")
    
    def initialize(self) -> bool:
        """
        Initialize and apply all optimizations.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check for GPU availability
            self.gpus = tf.config.list_physical_devices('GPU')
            if not self.gpus:
                logger.warning("No GPUs available, optimizations will have no effect")
                return False
            
            # Apply memory growth
            if self.enable_memory_growth:
                self._apply_memory_growth()
            
            # Apply memory limit
            if self.memory_limit_mb is not None:
                self._apply_memory_limit()
            
            # Apply mixed precision
            if self.enable_mixed_precision:
                self._apply_mixed_precision()
            
            # Apply XLA compilation
            if self.enable_xla:
                self._apply_xla()
            
            # Apply distribution strategy
            self._apply_distribution_strategy()
            
            # Apply GH200-specific optimizations
            if self.gh200_optimizations:
                self._apply_gh200_optimizations()
            
            self.is_initialized = True
            logger.info(f"GPU optimizations applied successfully for {len(self.gpus)} GPUs")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing GPU accelerator: {e}")
            logger.debug(traceback.format_exc())
            return False
    
    def _apply_memory_growth(self) -> None:
        """Apply memory growth to prevent TensorFlow from allocating all GPU memory at once."""
        try:
            for gpu in self.gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Applied memory growth to all GPUs")
        except Exception as e:
            logger.error(f"Error applying memory growth: {e}")
    
    def _apply_memory_limit(self) -> None:
        """Apply memory limit to each GPU."""
        try:
            for gpu in self.gpus:
                tf.config.set_logical_device_configuration(
                    gpu,
                    [tf.config.LogicalDeviceConfiguration(memory_limit=self.memory_limit_mb)]
                )
            logger.info(f"Applied memory limit of {self.memory_limit_mb}MB to all GPUs")
        except Exception as e:
            logger.error(f"Error applying memory limit: {e}")
    
    def _apply_mixed_precision(self) -> None:
        """Apply mixed precision training for improved performance on Tensor Cores."""
        try:
            self.original_policy = tf.keras.mixed_precision.global_policy()
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
            logger.info("Applied mixed precision training (float16)")
        except Exception as e:
            logger.error(f"Error applying mixed precision: {e}")
    
    def _apply_xla(self) -> None:
        """Apply XLA compilation for improved performance."""
        try:
            tf.config.optimizer.set_jit(True)
            logger.info("Applied XLA compilation")
        except Exception as e:
            logger.error(f"Error applying XLA compilation: {e}")
    
    def _apply_distribution_strategy(self) -> None:
        """Apply distribution strategy for multi-GPU training."""
        try:
            if len(self.gpus) <= 1:
                logger.info("Skipping distribution strategy for single GPU")
                return
            
            if self.distribution_strategy == "mirrored":
                self.strategy = tf.distribute.MirroredStrategy()
            elif self.distribution_strategy == "central_storage":
                self.strategy = tf.distribute.experimental.CentralStorageStrategy()
            elif self.distribution_strategy == "one_device":
                self.strategy = tf.distribute.OneDeviceStrategy(device="/gpu:0")
            else:
                logger.warning(f"Unknown distribution strategy: {self.distribution_strategy}")
                return
            
            logger.info(f"Applied {self.distribution_strategy} distribution strategy for {len(self.gpus)} GPUs")
        except Exception as e:
            logger.error(f"Error applying distribution strategy: {e}")
    
    def _apply_gh200_optimizations(self) -> None:
        """Apply GH200 Grace Hopper Superchip specific optimizations."""
        try:
            # Check if running on GH200
            is_gh200 = self._detect_gh200()
            if not is_gh200:
                logger.warning("GH200 optimizations enabled but not running on GH200")
                return
            
            # Apply GH200-specific optimizations
            
            # 1. Optimize for Grace CPU + Hopper GPU architecture
            os.environ["TF_GPU_HOST_MEM_LIMIT_IN_MB"] = "64000"  # 64GB host memory for Grace CPU
            
            # 2. Enable Transformer Engine optimizations for Hopper architecture
            os.environ["TF_ENABLE_TRANSFORMER_ENGINE"] = "1"
            
            # 3. Optimize for NVLink 4.0 between Grace CPU and Hopper GPU
            os.environ["TF_ENABLE_NVLINK_OPTIMIZATIONS"] = "1"
            
            # 4. Enable HBM3 memory optimizations
            os.environ["TF_ENABLE_HBM3_OPTIMIZATIONS"] = "1"
            
            logger.info("Applied GH200 Grace Hopper Superchip specific optimizations")
        except Exception as e:
            logger.error(f"Error applying GH200 optimizations: {e}")
    
    def _detect_gh200(self) -> bool:
        """Detect if running on GH200 Grace Hopper Superchip."""
        try:
            # Check for Grace CPU
            with open('/proc/cpuinfo', 'r') as f:
                cpu_info = f.read()
                if 'ARM' not in cpu_info or 'NVIDIA' not in cpu_info:
                    return False
            
            # Check for Hopper GPU
            gpu_name = tf.config.experimental.get_device_details(self.gpus[0])['device_name']
            if 'H100' not in gpu_name:
                return False
            
            return True
        except Exception as e:
            logger.error(f"Error detecting GH200: {e}")
            return False
    
    def get_strategy(self) -> Optional[tf.distribute.Strategy]:
        """
        Get the distribution strategy.
        
        Returns:
            Distribution strategy or None if not initialized
        """
        return self.strategy
    
    def get_scope(self) -> Any:
        """
        Get the strategy scope for model creation.
        
        Returns:
            Strategy scope or dummy context manager if not initialized
        """
        if self.strategy is not None:
            return self.strategy.scope()
        else:
            # Return a dummy context manager
            from contextlib import contextmanager
            @contextmanager
            def dummy_scope():
                yield
            return dummy_scope()
    
    def cleanup(self) -> None:
        """Clean up resources and restore original settings."""
        try:
            # Restore original mixed precision policy
            if self.original_policy is not None:
                tf.keras.mixed_precision.set_global_policy(self.original_policy)
            
            # Clear any other settings
            tf.config.optimizer.set_jit(False)
            
            logger.info("GPU accelerator cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up GPU accelerator: {e}")
    
    def benchmark(self, model: tf.keras.Model, input_shape: Tuple[int, ...]) -> Dict[str, float]:
        """
        Benchmark model performance on GPU.
        
        Args:
            model: TensorFlow model to benchmark
            input_shape: Input shape for benchmark
            
        Returns:
            Dictionary with benchmark results
        """
        try:
            # Create random input data
            batch_size = input_shape[0]
            input_data = tf.random.normal(input_shape)
            
            # Warm up
            for _ in range(10):
                _ = model(input_data, training=False)
            
            # Benchmark inference
            import time
            start_time = time.time()
            iterations = 100
            for _ in range(iterations):
                _ = model(input_data, training=False)
            end_time = time.time()
            
            # Calculate metrics
            elapsed_time = end_time - start_time
            inferences_per_second = iterations * batch_size / elapsed_time
            ms_per_inference = 1000 * elapsed_time / (iterations * batch_size)
            
            results = {
                "inferences_per_second": inferences_per_second,
                "ms_per_inference": ms_per_inference,
                "total_time_seconds": elapsed_time,
                "iterations": iterations,
                "batch_size": batch_size
            }
            
            logger.info(f"Benchmark results: {results}")
            return results
            
        except Exception as e:
            logger.error(f"Error benchmarking model: {e}")
            return {"error": str(e)}