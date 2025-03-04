# NVIDIA GH200 Optimized Dockerfile for Autonomous Trading System
# This Dockerfile is specifically optimized for the NVIDIA GH200 480GB accelerator

# Start with NVIDIA's CUDA 12.4 container with cuDNN
FROM nvcr.io/nvidia/cuda:12.4.0-cudnn9-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TF_ENABLE_ONEDNN_OPTS=1 \
    TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit" \
    TF_GPU_THREAD_MODE=gpu_private \
    TF_GPU_ALLOCATOR=cuda_malloc_async \
    CUDA_MODULE_LOADING=LAZY \
    TORCH_CUDNN_V8_API_ENABLED=1 \
    TORCH_ALLOW_TF32=1 \
    TORCH_CUDA_ARCH_LIST="8.9" \
    OMP_NUM_THREADS=16 \
    MKL_NUM_THREADS=16 \
    NUMEXPR_NUM_THREADS=16 \
    NUMEXPR_MAX_THREADS=16 \
    PYTHONPATH="/app:$PYTHONPATH"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    ca-certificates \
    libjpeg-dev \
    libpng-dev \
    librdmacm1 \
    libibverbs1 \
    ibverbs-providers \
    libhwloc-dev \
    libboost-all-dev \
    libffi-dev \
    liblz4-dev \
    libsnappy-dev \
    libbz2-dev \
    libzstd-dev \
    zlib1g-dev \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev \
    pkg-config \
    software-properties-common \
    openssh-client \
    openssh-server \
    libpq-dev \
    postgresql-client \
    redis-tools \
    htop \
    vim \
    tmux \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python 3.10
RUN add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-dev python3.10-distutils python3.10-venv && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install NVIDIA DALI for fast data loading
RUN pip install --no-cache-dir \
    nvidia-dali-cuda120

# Install RAPIDS for GPU-accelerated data processing
RUN pip install --no-cache-dir \
    cudf-cu12 \
    cuml-cu12 \
    cugraph-cu12 \
    cuspatial-cu12 \
    cuproj-cu12 \
    cucim-cu12 \
    pylibraft-cu12 \
    rmm-cu12 \
    --extra-index-url=https://pypi.nvidia.com

# Install TensorFlow with GPU support
RUN pip install --no-cache-dir \
    tensorflow==2.15.0 \
    tensorflow-probability==0.23.0 \
    tensorflow-addons==0.22.0 \
    tensorflow-io==0.34.0 \
    tensorrt==8.6.1 \
    tf2onnx==1.15.1

# Install PyTorch with GPU support
RUN pip install --no-cache-dir \
    torch==2.2.0 \
    torchvision==0.17.0 \
    torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Install JAX with GPU support
RUN pip install --no-cache-dir \
    "jax[cuda12_pip]" \
    -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Install XGBoost, LightGBM, and CatBoost with GPU support
RUN pip install --no-cache-dir \
    xgboost==2.0.2 \
    lightgbm==4.1.0 \
    catboost==1.2.2

# Install ONNX Runtime with GPU support
RUN pip install --no-cache-dir \
    onnxruntime-gpu==1.16.3

# Install other ML libraries
RUN pip install --no-cache-dir \
    scikit-learn==1.3.2 \
    statsmodels==0.14.0 \
    prophet==1.1.5 \
    pmdarima==2.0.4 \
    tsfresh==0.20.1 \
    featuretools==1.26.0 \
    shap==0.43.0 \
    optuna==3.4.0 \
    ray[tune]==2.9.0 \
    mlflow==2.10.0

# Install data processing libraries
RUN pip install --no-cache-dir \
    pandas==2.1.4 \
    numpy==1.26.3 \
    pyarrow==14.0.1 \
    polars==0.19.19 \
    dask==2023.12.1 \
    distributed==2023.12.1 \
    vaex==4.17.0 \
    modin==0.25.0 \
    fastparquet==2023.10.1

# Install financial libraries
RUN pip install --no-cache-dir \
    alpaca-trade-api==3.0.2 \
    polygon-api-client==1.13.3 \
    ccxt==4.1.22 \
    yfinance==0.2.35 \
    ta==0.11.0 \
    ta-lib==0.4.28 \
    pyfolio==0.9.2 \
    empyrical==0.5.5 \
    backtrader==1.9.78.123 \
    quantstats==0.0.62 \
    findatapy==0.1.32 \
    cvxpy==1.4.1

# Install monitoring and visualization libraries
RUN pip install --no-cache-dir \
    prometheus-client==0.19.0 \
    grafana-api==1.0.3 \
    dash==2.14.2 \
    plotly==5.18.0 \
    bokeh==3.3.0 \
    matplotlib==3.8.2 \
    seaborn==0.13.0 \
    holoviews==1.18.1 \
    hvplot==0.9.0

# Install utility libraries
RUN pip install --no-cache-dir \
    python-dotenv==1.0.0 \
    click==8.1.7 \
    tqdm==4.66.1 \
    loguru==0.7.2 \
    pydantic==2.5.2 \
    fastapi==0.104.1 \
    uvicorn==0.24.0.post1 \
    redis==5.0.1 \
    psycopg2-binary==2.9.9 \
    sqlalchemy==2.0.23 \
    alembic==1.12.1 \
    boto3==1.34.11 \
    slack-sdk==3.26.0 \
    pytest==7.4.3 \
    pytest-cov==4.1.0

# Install NVIDIA Nsight Systems for profiling
RUN wget -q https://developer.download.nvidia.com/devtools/nsight-systems/nsight-systems-2023.4.1_2023.4.1.122-1_amd64.deb && \
    dpkg -i nsight-systems-2023.4.1_2023.4.1.122-1_amd64.deb && \
    rm nsight-systems-2023.4.1_2023.4.1.122-1_amd64.deb

# Install NVIDIA Nsight Compute for kernel analysis
RUN wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nsight-compute-2023.3.1_2023.3.1.1-1_amd64.deb && \
    dpkg -i nsight-compute-2023.3.1_2023.3.1.1-1_amd64.deb && \
    rm nsight-compute-2023.3.1_2023.3.1.1-1_amd64.deb

# Create app directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY autonomous_trading_system/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY autonomous_trading_system/ .

# Set up NVIDIA Container Toolkit runtime configurations
ENV NVIDIA_REQUIRE_CUDA="cuda>=12.0"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

# Set up GH200-specific optimizations
RUN echo "export CUDA_VISIBLE_DEVICES=0" >> /etc/profile.d/cuda.sh && \
    echo "export TF_FORCE_GPU_ALLOW_GROWTH=true" >> /etc/profile.d/cuda.sh && \
    echo "export TF_XLA_FLAGS=\"--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit\"" >> /etc/profile.d/cuda.sh && \
    echo "export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512" >> /etc/profile.d/cuda.sh && \
    echo "export NCCL_P2P_LEVEL=NVL" >> /etc/profile.d/cuda.sh && \
    echo "export NCCL_IB_HCA=mlx5" >> /etc/profile.d/cuda.sh && \
    echo "export NCCL_DEBUG=INFO" >> /etc/profile.d/cuda.sh

# Add NVLINK optimizations for GH200
RUN echo "export NCCL_NVLS_ENABLE=1" >> /etc/profile.d/cuda.sh && \
    echo "export NCCL_NVLS_MEMOPS_ENABLE=1" >> /etc/profile.d/cuda.sh && \
    echo "export NCCL_NVLS_MEMOPS_P2P_ENABLE=1" >> /etc/profile.d/cuda.sh

# Add entrypoint script
COPY autonomous_trading_system/deployment/docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Set the entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["python", "-m", "src.scripts.system_controller"]