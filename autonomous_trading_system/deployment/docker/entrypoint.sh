#!/bin/bash
set -e

# Function to log messages with timestamp
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check GPU availability
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        log "Checking GPU availability..."
        if nvidia-smi &> /dev/null; then
            GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu --format=csv,noheader,nounits)
            log "GPU detected: $GPU_INFO"
            
            # Check if it's a GH200
            if echo "$GPU_INFO" | grep -q "GH200"; then
                log "NVIDIA GH200 detected - applying optimized settings"
                export CUDA_VISIBLE_DEVICES=0
                export TF_FORCE_GPU_ALLOW_GROWTH=true
                export TF_XLA_FLAGS="--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit"
                export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
                export NCCL_P2P_LEVEL=NVL
                export NCCL_IB_HCA=mlx5
                export NCCL_NVLS_ENABLE=1
                export NCCL_NVLS_MEMOPS_ENABLE=1
                export NCCL_NVLS_MEMOPS_P2P_ENABLE=1
                
                # Set optimal batch sizes for GH200
                export TRAINING_BATCH_SIZE=4096
                export INFERENCE_BATCH_SIZE=8192
                export FEATURE_ENGINEERING_BATCH_SIZE=16384
            else
                log "Standard GPU detected - applying default settings"
                export CUDA_VISIBLE_DEVICES=0
                export TF_FORCE_GPU_ALLOW_GROWTH=true
                
                # Set default batch sizes
                export TRAINING_BATCH_SIZE=256
                export INFERENCE_BATCH_SIZE=512
                export FEATURE_ENGINEERING_BATCH_SIZE=1024
            fi
            
            # Set USE_GPU environment variable
            export USE_GPU=true
            return 0
        else
            log "WARNING: nvidia-smi failed, GPU might not be accessible"
            export USE_GPU=false
            return 1
        fi
    else
        log "WARNING: nvidia-smi not found, running without GPU"
        export USE_GPU=false
        return 1
    fi
}

# Function to check and wait for database
wait_for_database() {
    log "Checking database connection..."
    
    # Get database connection details from environment
    DB_HOST=${TIMESCALEDB_HOST:-localhost}
    DB_PORT=${TIMESCALEDB_PORT:-5432}
    DB_NAME=${TIMESCALEDB_DATABASE:-ats_db}
    DB_USER=${TIMESCALEDB_USER:-ats_user}
    DB_PASSWORD=${TIMESCALEDB_PASSWORD:-secure_password_here}
    
    # Wait for database to be ready
    MAX_RETRIES=30
    RETRY_INTERVAL=2
    
    for i in $(seq 1 $MAX_RETRIES); do
        if PGPASSWORD=$DB_PASSWORD psql -h $DB_HOST -p $DB_PORT -U $DB_USER -d $DB_NAME -c "SELECT 1" &> /dev/null; then
            log "Successfully connected to database"
            return 0
        else
            log "Waiting for database to be ready... ($i/$MAX_RETRIES)"
            sleep $RETRY_INTERVAL
        fi
    done
    
    log "ERROR: Failed to connect to database after $MAX_RETRIES attempts"
    return 1
}

# Function to check and wait for Redis
wait_for_redis() {
    log "Checking Redis connection..."
    
    # Get Redis connection details from environment
    REDIS_HOST=${REDIS_HOST:-localhost}
    REDIS_PORT=${REDIS_PORT:-6379}
    
    # Wait for Redis to be ready
    MAX_RETRIES=30
    RETRY_INTERVAL=2
    
    for i in $(seq 1 $MAX_RETRIES); do
        if redis-cli -h $REDIS_HOST -p $REDIS_PORT PING | grep -q "PONG"; then
            log "Successfully connected to Redis"
            return 0
        else
            log "Waiting for Redis to be ready... ($i/$MAX_RETRIES)"
            sleep $RETRY_INTERVAL
        fi
    done
    
    log "ERROR: Failed to connect to Redis after $MAX_RETRIES attempts"
    return 1
}

# Function to initialize the system
initialize_system() {
    log "Initializing system..."
    
    # Create necessary directories
    mkdir -p /app/data
    mkdir -p /app/logs
    mkdir -p /app/models
    mkdir -p /app/results
    
    # Set permissions
    chmod -R 755 /app
    
    # Initialize database if needed
    if [[ "$INITIALIZE_DB" == "true" ]]; then
        log "Initializing database schema..."
        python -m src.scripts.setup_database
    fi
    
    # Set up environment
    log "Setting up environment..."
    python -m src.scripts.setup_environment
    
    log "System initialization complete"
}

# Function to start the monitoring exporters
start_monitoring() {
    log "Starting monitoring exporters..."
    
    # Start Prometheus exporter in the background
    python -m src.monitoring.exporters.prometheus_exporter &
    PROMETHEUS_PID=$!
    log "Prometheus exporter started with PID $PROMETHEUS_PID"
    
    # Export PIDs for proper shutdown
    echo $PROMETHEUS_PID > /app/prometheus_exporter.pid
}

# Function to handle graceful shutdown
handle_shutdown() {
    log "Received shutdown signal, terminating processes..."
    
    # Kill monitoring exporters
    if [ -f /app/prometheus_exporter.pid ]; then
        PROMETHEUS_PID=$(cat /app/prometheus_exporter.pid)
        kill -TERM $PROMETHEUS_PID 2>/dev/null || true
        log "Terminated Prometheus exporter (PID $PROMETHEUS_PID)"
    fi
    
    # Kill any other background processes
    jobs -p | xargs -r kill
    
    log "Shutdown complete"
    exit 0
}

# Register the shutdown handler
trap handle_shutdown SIGTERM SIGINT

# Main execution
log "Starting Autonomous Trading System container..."

# Check GPU availability
check_gpu

# Wait for dependencies
wait_for_database
wait_for_redis

# Initialize the system
initialize_system

# Start monitoring
start_monitoring

# Determine which component to run based on the COMPONENT environment variable
COMPONENT=${COMPONENT:-system_controller}

log "Starting component: $COMPONENT"

case $COMPONENT in
    data_acquisition)
        exec python -m src.scripts.run_data_acquisition "$@"
        ;;
    feature_engineering)
        exec python -m src.scripts.run_feature_engineering "$@"
        ;;
    model_training)
        exec python -m src.scripts.run_model_training "$@"
        ;;
    trading_strategy)
        exec python -m src.scripts.run_trading_strategy "$@"
        ;;
    monitoring)
        exec python -m src.scripts.run_monitoring "$@"
        ;;
    continuous_learning)
        exec python -m src.scripts.run_continuous_learning "$@"
        ;;
    backtest)
        exec python -m src.scripts.run_backtest "$@"
        ;;
    system_controller|*)
        exec python -m src.scripts.system_controller "$@"
        ;;
esac