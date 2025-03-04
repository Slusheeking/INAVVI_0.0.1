"""
Hardware Configuration Module for Autonomous Trading System

This module provides configuration settings for hardware resources,
particularly focused on GPU utilization for the NVIDIA GH200 480GB.
"""

import os
import logging
from typing import Dict, Any, Optional, List, Tuple
import json

# Configure logging
logger = logging.getLogger(__name__)

# Load environment variables
USE_GPU = os.environ.get('USE_GPU', 'true').lower() == 'true'


class HardwareConfig:
    """Hardware configuration class for the autonomous trading system."""
    
    # GPU Configuration
    GPU_CONFIG = {
        # NVIDIA GH200 480GB specific settings
        'memory_limit': 94000,  # MB, slightly below total to leave room for system
        'growth_memory': True,  # Allow TensorFlow to grow memory allocation
        'allow_memory_growth': True,  # For PyTorch
        'mixed_precision': True,  # Use mixed precision training (FP16/BF16)
        'cudnn_benchmark': True,  # Use cuDNN auto-tuner
        'tensor_cores': True,  # Enable tensor cores for faster matrix operations
        'device_id': 0,  # Default GPU device ID
        'num_streams': 4,  # Number of CUDA streams for parallel operations
        'max_workspace_size': 4096,  # MB, for TensorRT optimization
        'xla_compilation': True,  # Enable XLA compilation for TensorFlow
        'amp_level': 'O2',  # Automatic mixed precision level
        'tf32_precision': True,  # Enable TF32 precision mode
    }
    
    # CPU Configuration
    CPU_CONFIG = {
        'num_threads': os.cpu_count() or 16,  # Default to 16 if os.cpu_count() returns None
        'inter_op_parallelism': 4,  # TensorFlow inter-op parallelism
        'intra_op_parallelism': os.cpu_count() or 16,  # TensorFlow intra-op parallelism
        'mkl_threads': os.cpu_count() or 16,  # MKL threads
        'omp_threads': os.cpu_count() or 16,  # OpenMP threads
    }
    
    # Memory Configuration
    MEMORY_CONFIG = {
        'prefetch_buffer_size': 8,  # Number of batches to prefetch
        'dataset_buffer_size': 10000,  # Buffer size for dataset shuffling
        'pin_memory': True,  # Pin memory for faster GPU transfer
        'non_blocking': True,  # Non-blocking GPU memory transfers
        'persistent_workers': True,  # Keep worker processes alive between iterations
    }
    
    # Distributed Training Configuration
    DISTRIBUTED_CONFIG = {
        'enabled': False,  # Enable distributed training
        'backend': 'nccl',  # NCCL backend for GPU communication
        'world_size': 1,  # Number of processes
        'rank': 0,  # Process rank
        'init_method': 'env://',  # Initialization method
    }
    
    # Optimization Configuration
    OPTIMIZATION_CONFIG = {
        'jit_compile': True,  # JIT compilation for PyTorch
        'graph_optimization': True,  # TensorFlow graph optimization
        'kernel_fusion': True,  # Fuse multiple operations into a single kernel
        'cudnn_deterministic': False,  # Disable for better performance
        'benchmark_iterations': 10,  # Number of iterations for benchmarking
    }
    
    @classmethod
    def get_gpu_info(cls) -> Dict[str, Any]:
        """
        Get GPU information using NVIDIA System Management Interface.
        
        Returns:
            Dict[str, Any]: Dictionary containing GPU information
        """
        try:
            import subprocess
            import re
            
            # Run nvidia-smi command
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            
            # Parse output
            output = result.stdout.strip()
            if not output:
                logger.warning("No GPU information available from nvidia-smi")
                return {}
                
            # Parse the first GPU (we're using a single GPU)
            parts = output.split(',')
            if len(parts) >= 4:
                gpu_info = {
                    'name': parts[0].strip(),
                    'memory_total_mb': float(parts[1].strip()),
                    'memory_free_mb': float(parts[2].strip()),
                    'utilization_percent': float(parts[3].strip())
                }
                return gpu_info
            else:
                logger.warning(f"Unexpected nvidia-smi output format: {output}")
                return {}
                
        except Exception as e:
            logger.warning(f"Failed to get GPU information: {e}")
            return {}
    
    @classmethod
    def configure_tensorflow(cls) -> Dict[str, Any]:
        """
        Configure TensorFlow for optimal performance on the NVIDIA GH200.
        
        Returns:
            Dict[str, Any]: TensorFlow configuration
        """
        if not USE_GPU:
            return {'device': '/cpu:0'}
            
        try:
            import tensorflow as tf
            
            # Configure GPU memory growth
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, cls.GPU_CONFIG['growth_memory'])
                
                # Set memory limit
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=cls.GPU_CONFIG['memory_limit']
                    )]
                )
                
            # Configure mixed precision
            if cls.GPU_CONFIG['mixed_precision']:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                
            # Configure XLA
            if cls.GPU_CONFIG['xla_compilation']:
                tf.config.optimizer.set_jit(True)
                
            # Configure thread parallelism
            tf.config.threading.set_inter_op_parallelism_threads(cls.CPU_CONFIG['inter_op_parallelism'])
            tf.config.threading.set_intra_op_parallelism_threads(cls.CPU_CONFIG['intra_op_parallelism'])
            
            return {
                'device': '/gpu:0',
                'mixed_precision_enabled': cls.GPU_CONFIG['mixed_precision'],
                'xla_enabled': cls.GPU_CONFIG['xla_compilation']
            }
            
        except ImportError:
            logger.warning("TensorFlow not installed, skipping TensorFlow configuration")
            return {}
        except Exception as e:
            logger.warning(f"Failed to configure TensorFlow: {e}")
            return {'device': '/cpu:0'}
    
    @classmethod
    def configure_pytorch(cls) -> Dict[str, Any]:
        """
        Configure PyTorch for optimal performance on the NVIDIA GH200.
        
        Returns:
            Dict[str, Any]: PyTorch configuration
        """
        if not USE_GPU:
            return {'device': 'cpu'}
            
        try:
            import torch
            
            # Set default device
            device = torch.device(f'cuda:{cls.GPU_CONFIG["device_id"]}' if torch.cuda.is_available() else 'cpu')
            
            # Configure cuDNN
            torch.backends.cudnn.benchmark = cls.GPU_CONFIG['cudnn_benchmark']
            torch.backends.cudnn.deterministic = cls.OPTIMIZATION_CONFIG['cudnn_deterministic']
            
            # Configure TF32 precision
            torch.backends.cuda.matmul.allow_tf32 = cls.GPU_CONFIG['tf32_precision']
            torch.backends.cudnn.allow_tf32 = cls.GPU_CONFIG['tf32_precision']
            
            # Configure default tensor type
            if device.type == 'cuda':
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
                
            # Configure memory allocation
            if cls.GPU_CONFIG['allow_memory_growth']:
                # This is handled by PyTorch automatically
                pass
                
            return {
                'device': device,
                'cudnn_benchmark': cls.GPU_CONFIG['cudnn_benchmark'],
                'cudnn_deterministic': cls.OPTIMIZATION_CONFIG['cudnn_deterministic'],
                'tf32_enabled': cls.GPU_CONFIG['tf32_precision']
            }
            
        except ImportError:
            logger.warning("PyTorch not installed, skipping PyTorch configuration")
            return {'device': 'cpu'}
        except Exception as e:
            logger.warning(f"Failed to configure PyTorch: {e}")
            return {'device': 'cpu'}
    
    @classmethod
    def configure_rapids(cls) -> Dict[str, Any]:
        """
        Configure RAPIDS for GPU-accelerated data processing.
        
        Returns:
            Dict[str, Any]: RAPIDS configuration
        """
        if not USE_GPU:
            return {'enabled': False}
            
        try:
            # Import RAPIDS libraries
            import cudf
            import cuml
            
            # Configure memory pool
            memory_pool = cudf.rmm.PoolMemoryResource(
                cudf.rmm.CudaMemoryResource(),
                initial_pool_size=1 << 30,  # 1 GB
                maximum_pool_size=cls.GPU_CONFIG['memory_limit'] << 20  # Convert to bytes
            )
            cudf.rmm.mr.set_current_device_resource(memory_pool)
            
            return {
                'enabled': True,
                'memory_pool_size_gb': cls.GPU_CONFIG['memory_limit'] / 1024
            }
            
        except ImportError:
            logger.warning("RAPIDS not installed, skipping RAPIDS configuration")
            return {'enabled': False}
        except Exception as e:
            logger.warning(f"Failed to configure RAPIDS: {e}")
            return {'enabled': False}
    
    @classmethod
    def get_optimal_batch_sizes(cls) -> Dict[str, int]:
        """
        Calculate optimal batch sizes for different operations based on GPU memory.
        
        Returns:
            Dict[str, int]: Dictionary of optimal batch sizes for different operations
        """
        gpu_info = cls.get_gpu_info()
        memory_total_gb = gpu_info.get('memory_total_mb', 0) / 1024
        
        # Default batch sizes for NVIDIA GH200 480GB
        if memory_total_gb > 90:  # Confirm we're on the GH200
            return {
                'training': 4096,  # Large batch size for training
                'inference': 8192,  # Even larger for inference
                'feature_engineering': 16384,  # Very large for feature engineering
                'backtesting': 2048,  # Moderate for backtesting (more complex operations)
                'data_loading': 32768,  # Extremely large for data loading
            }
        elif memory_total_gb > 20:  # High-end GPU but not GH200
            return {
                'training': 1024,
                'inference': 2048,
                'feature_engineering': 4096,
                'backtesting': 512,
                'data_loading': 8192,
            }
        elif memory_total_gb > 8:  # Mid-range GPU
            return {
                'training': 256,
                'inference': 512,
                'feature_engineering': 1024,
                'backtesting': 128,
                'data_loading': 2048,
            }
        else:  # Low-end GPU or CPU
            return {
                'training': 64,
                'inference': 128,
                'feature_engineering': 256,
                'backtesting': 32,
                'data_loading': 512,
            }
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """
        Get the complete hardware configuration.
        
        Returns:
            Dict[str, Any]: Complete hardware configuration
        """
        gpu_info = cls.get_gpu_info()
        tf_config = cls.configure_tensorflow()
        pytorch_config = cls.configure_pytorch()
        rapids_config = cls.configure_rapids()
        batch_sizes = cls.get_optimal_batch_sizes()
        
        return {
            'use_gpu': USE_GPU,
            'gpu_info': gpu_info,
            'gpu_config': cls.GPU_CONFIG,
            'cpu_config': cls.CPU_CONFIG,
            'memory_config': cls.MEMORY_CONFIG,
            'distributed_config': cls.DISTRIBUTED_CONFIG,
            'optimization_config': cls.OPTIMIZATION_CONFIG,
            'tensorflow_config': tf_config,
            'pytorch_config': pytorch_config,
            'rapids_config': rapids_config,
            'optimal_batch_sizes': batch_sizes,
        }


# Initialize hardware configuration
hardware_config = HardwareConfig.get_config()

# Export configuration
__all__ = ['HardwareConfig', 'hardware_config']