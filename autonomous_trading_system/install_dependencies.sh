#!/bin/bash
# Install dependencies for the Autonomous Trading System
# This script installs all required packages, including those for FinBERT sentiment analysis

set -e  # Exit on error

echo "Installing dependencies for the Autonomous Trading System..."

# Create and activate virtual environment (optional)
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created."
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install base requirements
echo "Installing base requirements..."
pip install -r autonomous_trading_system/requirements.txt

# Install development requirements
echo "Installing development requirements..."
pip install -r autonomous_trading_system/requirements-dev.txt

# Install additional dependencies for FinBERT sentiment analysis
echo "Installing dependencies for FinBERT sentiment analysis..."
pip install torch transformers spacy tqdm

# Download spaCy model
echo "Downloading spaCy model..."
python -m spacy download en_core_web_sm

# Install TimescaleDB (if not already installed)
if ! command -v psql &> /dev/null; then
    echo "PostgreSQL not found. Installing TimescaleDB..."
    
    # Check OS
    if [ "$(uname)" == "Darwin" ]; then
        # macOS
        echo "Detected macOS. Installing TimescaleDB using Homebrew..."
        brew install timescaledb
        
        # Initialize TimescaleDB
        echo "Initializing TimescaleDB..."
        brew services start postgresql
        sleep 5  # Wait for PostgreSQL to start
        
    elif [ "$(uname)" == "Linux" ]; then
        # Linux
        echo "Detected Linux. Installing TimescaleDB using apt..."
        
        # Add TimescaleDB repository
        echo "Adding TimescaleDB repository..."
        sudo apt-get update
        sudo apt-get install -y gnupg postgresql-common apt-transport-https lsb-release wget
        sudo sh -c "$(wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey) | apt-key add -"
        sudo sh -c "echo 'deb https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -c -s) main' > /etc/apt/sources.list.d/timescaledb.list"
        
        # Install TimescaleDB
        echo "Installing TimescaleDB..."
        sudo apt-get update
        sudo apt-get install -y timescaledb-postgresql-14
        
        # Configure TimescaleDB
        echo "Configuring TimescaleDB..."
        sudo timescaledb-tune --quiet --yes
        sudo systemctl restart postgresql
    else
        echo "Unsupported OS. Please install TimescaleDB manually."
    fi
else
    echo "PostgreSQL already installed. Skipping TimescaleDB installation."
fi

# Install Redis (if not already installed)
if ! command -v redis-server &> /dev/null; then
    echo "Redis not found. Installing Redis..."
    
    # Check OS
    if [ "$(uname)" == "Darwin" ]; then
        # macOS
        echo "Detected macOS. Installing Redis using Homebrew..."
        brew install redis
        
        # Start Redis
        echo "Starting Redis..."
        brew services start redis
        
    elif [ "$(uname)" == "Linux" ]; then
        # Linux
        echo "Detected Linux. Installing Redis using apt..."
        sudo apt-get update
        sudo apt-get install -y redis-server
        
        # Start Redis
        echo "Starting Redis..."
        sudo systemctl start redis-server
    else
        echo "Unsupported OS. Please install Redis manually."
    fi
else
    echo "Redis already installed. Skipping Redis installation."
fi

# Install Docker (if not already installed)
if ! command -v docker &> /dev/null; then
    echo "Docker not found. Installing Docker..."
    
    # Check OS
    if [ "$(uname)" == "Darwin" ]; then
        # macOS
        echo "Detected macOS. Please install Docker Desktop manually from https://www.docker.com/products/docker-desktop"
        
    elif [ "$(uname)" == "Linux" ]; then
        # Linux
        echo "Detected Linux. Installing Docker using apt..."
        sudo apt-get update
        sudo apt-get install -y apt-transport-https ca-certificates curl gnupg lsb-release
        
        # Add Docker's official GPG key
        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
        
        # Set up the stable repository
        echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
        
        # Install Docker Engine
        sudo apt-get update
        sudo apt-get install -y docker-ce docker-ce-cli containerd.io
        
        # Add user to docker group
        sudo usermod -aG docker $USER
        
        # Start Docker
        sudo systemctl start docker
    else
        echo "Unsupported OS. Please install Docker manually."
    fi
else
    echo "Docker already installed. Skipping Docker installation."
fi

# Install Docker Compose (if not already installed)
if ! command -v docker-compose &> /dev/null; then
    echo "Docker Compose not found. Installing Docker Compose..."
    
    # Check OS
    if [ "$(uname)" == "Darwin" ]; then
        # macOS
        echo "Detected macOS. Docker Compose is included with Docker Desktop."
        
    elif [ "$(uname)" == "Linux" ]; then
        # Linux
        echo "Detected Linux. Installing Docker Compose..."
        sudo curl -L "https://github.com/docker/compose/releases/download/v2.18.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
        sudo chmod +x /usr/local/bin/docker-compose
    else
        echo "Unsupported OS. Please install Docker Compose manually."
    fi
else
    echo "Docker Compose already installed. Skipping Docker Compose installation."
fi

# Install NVIDIA CUDA Toolkit (if GPU is available)
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU detected. Installing CUDA Toolkit..."
    
    # Check OS
    if [ "$(uname)" == "Darwin" ]; then
        # macOS
        echo "CUDA is not supported on macOS with Apple Silicon. Using CPU for model training."
        
    elif [ "$(uname)" == "Linux" ]; then
        # Linux
        echo "Detected Linux. Installing CUDA Toolkit..."
        
        # Add NVIDIA repository
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
        sudo dpkg -i cuda-keyring_1.0-1_all.deb
        sudo apt-get update
        
        # Install CUDA Toolkit
        sudo apt-get install -y cuda-toolkit-12-0
        
        # Set up environment variables
        echo 'export PATH=/usr/local/cuda-12.0/bin${PATH:+:${PATH}}' >> ~/.bashrc
        echo 'export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
        
        # Install NVIDIA Docker
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
        sudo apt-get update
        sudo apt-get install -y nvidia-docker2
        sudo systemctl restart docker
    else
        echo "Unsupported OS. Please install CUDA Toolkit manually."
    fi
else
    echo "No NVIDIA GPU detected. Skipping CUDA Toolkit installation."
fi

# Set up the database
echo "Setting up the database..."
python autonomous_trading_system/src/scripts/setup_database.py

echo "All dependencies installed successfully!"
echo "You can now run the autonomous trading system."