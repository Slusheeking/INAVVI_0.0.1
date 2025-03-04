"""
CI/CD Configuration Module

This module contains configuration settings for the CI/CD pipeline and deployment processes.
It provides centralized configuration for build, test, and deployment stages across
different environments (development, staging, production).
"""

import os
from enum import Enum
from typing import Dict, List, Optional, Union

# Environment types
class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

# Component types
class Component(str, Enum):
    DATA_ACQUISITION = "data-acquisition"
    FEATURE_ENGINEERING = "feature-engineering"
    MODEL_TRAINING = "model-training"
    TRADING_STRATEGY = "trading-strategy"
    MONITORING = "monitoring"
    CONTINUOUS_LEARNING = "continuous-learning"
    BACKTESTING = "backtesting"
    SYSTEM_CONTROLLER = "system-controller"

# Docker registry configuration
DOCKER_REGISTRY = os.environ.get("DOCKER_REGISTRY", "docker.io")
DOCKER_REPOSITORY = os.environ.get("DOCKER_REPOSITORY", "ats")
DOCKER_TAG_PREFIX = os.environ.get("DOCKER_TAG_PREFIX", "")

# Build configuration
BUILD_CONFIG = {
    "base_image": "python:3.10-slim",
    "gpu_base_image": "nvcr.io/nvidia/cuda:12.4.0-cudnn9-devel-ubuntu22.04",
    "build_args": {
        "PYTHON_VERSION": "3.10",
        "DEBIAN_FRONTEND": "noninteractive",
    },
    "labels": {
        "maintainer": "ATS Team <ats-team@example.com>",
        "version": "${VERSION}",
        "build-date": "${BUILD_DATE}",
        "vcs-ref": "${VCS_REF}",
    },
}

# Test configuration
TEST_CONFIG = {
    "unit_test_dirs": ["tests/unit"],
    "integration_test_dirs": ["tests/integration"],
    "performance_test_dirs": ["tests/performance"],
    "system_test_dirs": ["tests/system"],
    "test_timeout": 300,  # seconds
    "coverage_threshold": 80,  # percentage
    "lint_rules": {
        "max_line_length": 100,
        "max_complexity": 10,
    },
}

# Deployment configuration
DEPLOYMENT_CONFIG = {
    Environment.DEVELOPMENT: {
        "kubernetes_namespace": "ats-dev",
        "replicas": {
            Component.DATA_ACQUISITION: 1,
            Component.FEATURE_ENGINEERING: 1,
            Component.MODEL_TRAINING: 1,
            Component.TRADING_STRATEGY: 1,
            Component.MONITORING: 1,
            Component.CONTINUOUS_LEARNING: 1,
            Component.BACKTESTING: 1,
            Component.SYSTEM_CONTROLLER: 1,
        },
        "resource_limits": {
            "cpu": "1000m",
            "memory": "2Gi",
            "gpu": "0",
        },
        "resource_requests": {
            "cpu": "500m",
            "memory": "1Gi",
        },
        "auto_scaling": False,
        "rolling_update": {
            "max_surge": 1,
            "max_unavailable": 0,
        },
    },
    Environment.STAGING: {
        "kubernetes_namespace": "ats-staging",
        "replicas": {
            Component.DATA_ACQUISITION: 2,
            Component.FEATURE_ENGINEERING: 2,
            Component.MODEL_TRAINING: 2,
            Component.TRADING_STRATEGY: 2,
            Component.MONITORING: 1,
            Component.CONTINUOUS_LEARNING: 1,
            Component.BACKTESTING: 1,
            Component.SYSTEM_CONTROLLER: 1,
        },
        "resource_limits": {
            "cpu": "2000m",
            "memory": "4Gi",
            "gpu": "1",
        },
        "resource_requests": {
            "cpu": "1000m",
            "memory": "2Gi",
        },
        "auto_scaling": True,
        "auto_scaling_config": {
            "min_replicas": 1,
            "max_replicas": 5,
            "target_cpu_utilization": 70,
            "target_memory_utilization": 80,
        },
        "rolling_update": {
            "max_surge": 1,
            "max_unavailable": 0,
        },
    },
    Environment.PRODUCTION: {
        "kubernetes_namespace": "ats-prod",
        "replicas": {
            Component.DATA_ACQUISITION: 3,
            Component.FEATURE_ENGINEERING: 3,
            Component.MODEL_TRAINING: 3,
            Component.TRADING_STRATEGY: 3,
            Component.MONITORING: 2,
            Component.CONTINUOUS_LEARNING: 2,
            Component.BACKTESTING: 1,
            Component.SYSTEM_CONTROLLER: 2,
        },
        "resource_limits": {
            "cpu": "4000m",
            "memory": "8Gi",
            "gpu": "2",
        },
        "resource_requests": {
            "cpu": "2000m",
            "memory": "4Gi",
        },
        "auto_scaling": True,
        "auto_scaling_config": {
            "min_replicas": 2,
            "max_replicas": 10,
            "target_cpu_utilization": 70,
            "target_memory_utilization": 80,
        },
        "rolling_update": {
            "max_surge": 1,
            "max_unavailable": 0,
        },
        "node_selectors": {
            "cloud.google.com/gke-nodepool": "ats-prod-pool",
        },
        "tolerations": [
            {
                "key": "dedicated",
                "operator": "Equal",
                "value": "ats",
                "effect": "NoSchedule",
            }
        ],
    },
}

# GPU-specific configuration
GPU_CONFIG = {
    "enabled": os.environ.get("USE_GPU", "false").lower() == "true",
    "models": [
        Component.MODEL_TRAINING,
        Component.FEATURE_ENGINEERING,
        Component.TRADING_STRATEGY,
    ],
    "gh200_optimizations": {
        "enabled": os.environ.get("USE_GH200", "false").lower() == "true",
        "env_vars": {
            "CUDA_VISIBLE_DEVICES": "0",
            "TF_FORCE_GPU_ALLOW_GROWTH": "true",
            "TF_XLA_FLAGS": "--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit",
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512",
            "NCCL_P2P_LEVEL": "NVL",
            "NCCL_IB_HCA": "mlx5",
            "NCCL_NVLS_ENABLE": "1",
            "NCCL_NVLS_MEMOPS_ENABLE": "1",
            "NCCL_NVLS_MEMOPS_P2P_ENABLE": "1",
        },
        "batch_sizes": {
            "training": 4096,
            "inference": 8192,
            "feature_engineering": 16384,
        },
    },
}

# Notification configuration
NOTIFICATION_CONFIG = {
    "slack": {
        "enabled": True,
        "channels": {
            "deployments": "#ats-deployments",
            "alerts": "#ats-alerts",
            "ci_cd": "#ats-ci-cd",
        },
    },
    "email": {
        "enabled": True,
        "recipients": {
            "deployments": ["devops@example.com", "trading-team@example.com"],
            "alerts": ["alerts@example.com", "trading-team@example.com"],
        },
    },
}

# Security scanning configuration
SECURITY_SCAN_CONFIG = {
    "enabled": True,
    "scan_dependencies": True,
    "scan_docker_images": True,
    "scan_kubernetes_manifests": True,
    "fail_on_critical": True,
    "fail_on_high": True,
    "fail_on_medium": False,
}

# Feature flags for CI/CD pipeline
FEATURE_FLAGS = {
    "run_performance_tests": os.environ.get("RUN_PERFORMANCE_TESTS", "false").lower() == "true",
    "deploy_to_staging": os.environ.get("DEPLOY_TO_STAGING", "true").lower() == "true",
    "deploy_to_production": os.environ.get("DEPLOY_TO_PRODUCTION", "false").lower() == "true",
    "run_security_scans": os.environ.get("RUN_SECURITY_SCANS", "true").lower() == "true",
    "use_gpu_acceleration": os.environ.get("USE_GPU_ACCELERATION", "false").lower() == "true",
    "enable_auto_rollback": os.environ.get("ENABLE_AUTO_ROLLBACK", "true").lower() == "true",
}

def get_docker_image_name(component: Component, environment: Environment, version: str, use_gpu: bool = False) -> str:
    """
    Generate a Docker image name based on component, environment, and version.
    
    Args:
        component: The system component
        environment: The deployment environment
        version: The version tag
        use_gpu: Whether to use GPU-enabled image
        
    Returns:
        Fully qualified Docker image name
    """
    gpu_suffix = "-gpu" if use_gpu else ""
    tag = f"{DOCKER_TAG_PREFIX}{version}" if DOCKER_TAG_PREFIX else version
    
    return f"{DOCKER_REGISTRY}/{DOCKER_REPOSITORY}/{component}{gpu_suffix}:{tag}"

def get_deployment_config(environment: Environment) -> Dict:
    """
    Get deployment configuration for a specific environment.
    
    Args:
        environment: The deployment environment
        
    Returns:
        Deployment configuration dictionary
    """
    if environment not in DEPLOYMENT_CONFIG:
        raise ValueError(f"Unknown environment: {environment}")
    
    return DEPLOYMENT_CONFIG[environment]

def should_use_gpu(component: Component) -> bool:
    """
    Determine if a component should use GPU acceleration.
    
    Args:
        component: The system component
        
    Returns:
        True if the component should use GPU, False otherwise
    """
    return GPU_CONFIG["enabled"] and component in GPU_CONFIG["models"]

def get_resource_requirements(component: Component, environment: Environment) -> Dict:
    """
    Get resource requirements for a specific component and environment.
    
    Args:
        component: The system component
        environment: The deployment environment
        
    Returns:
        Resource requirements dictionary
    """
    env_config = get_deployment_config(environment)
    
    resources = {
        "limits": {
            "cpu": env_config["resource_limits"]["cpu"],
            "memory": env_config["resource_limits"]["memory"],
        },
        "requests": {
            "cpu": env_config["resource_requests"]["cpu"],
            "memory": env_config["resource_requests"]["memory"],
        }
    }
    
    # Add GPU resources if needed
    if should_use_gpu(component):
        resources["limits"]["nvidia.com/gpu"] = env_config["resource_limits"]["gpu"]
    
    return resources