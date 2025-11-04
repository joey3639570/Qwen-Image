#!/bin/bash
# Startup script for Qwen-Image-Edit FastAPI service

set -e

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    echo "Loading environment variables from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# Set default environment variables if not set
export NUM_GPUS_TO_USE=${NUM_GPUS_TO_USE:-$(python3 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 0)")}
export TASK_QUEUE_SIZE=${TASK_QUEUE_SIZE:-100}
export TASK_TIMEOUT=${TASK_TIMEOUT:-300}
export MODEL_REPO_ID=${MODEL_REPO_ID:-"Qwen/Qwen-Image-Edit"}
export HOST=${HOST:-"0.0.0.0"}
export PORT=${PORT:-8000}
export WORKERS=${WORKERS:-1}

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Check if CUDA is available
python3 << EOF
import torch
if not torch.cuda.is_available():
    print("ERROR: CUDA is not available!")
    print("Please ensure:")
    print("1. CUDA 12.8 is installed")
    print("2. PyTorch is installed with CUDA support")
    print("3. NVIDIA drivers are properly installed")
    exit(1)
print(f"✓ CUDA is available: {torch.cuda.device_count()} GPU(s)")
EOF

echo "=========================================="
echo "Starting Qwen-Image-Edit FastAPI Service"
echo "=========================================="
echo "Configuration:"
echo "  Host: $HOST"
echo "  Port: $PORT"
echo "  Workers: $WORKERS"
echo "  GPUs: $NUM_GPUS_TO_USE"
echo "  Task Queue Size: $TASK_QUEUE_SIZE"
echo "  Task Timeout: $TASK_TIMEOUT seconds"
echo "  Model: $MODEL_REPO_ID"
echo "=========================================="
echo ""

# Start the FastAPI application
cd "$(dirname "$0")"
python3 -m src.api.main

