#!/bin/bash
# Installation script for Qwen-Image-Edit FastAPI service

set -e

echo "=========================================="
echo "Qwen-Image-Edit FastAPI Service Installer"
echo "=========================================="

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.12"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 12) else 1)"; then
    echo "✓ Python version $PYTHON_VERSION is compatible"
else
    echo "✗ Python 3.12+ is required. Found: $PYTHON_VERSION"
    exit 1
fi

# Check CUDA version
echo "Checking CUDA version..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -c1-4)
    echo "✓ CUDA version $CUDA_VERSION detected"
else
    echo "⚠ Warning: nvcc not found, but continuing installation..."
    echo "  Make sure CUDA 12.8 is installed and accessible"
fi

# Check if CUDA is available in Python
echo "Checking CUDA availability in Python..."
python3 << EOF
import sys
try:
    import torch
    if torch.cuda.is_available():
        print(f"✓ CUDA is available in PyTorch")
        print(f"  CUDA Version: {torch.version.cuda}")
        print(f"  Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("⚠ Warning: CUDA is not available in PyTorch")
        print("  Make sure PyTorch is installed with CUDA support")
        sys.exit(1)
except ImportError:
    print("⚠ PyTorch not installed yet, will be installed next")
EOF

# Create virtual environment (optional)
if [ "$USE_VENV" = "true" ]; then
    echo "Creating virtual environment..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Upgrade pip
echo "Upgrading pip..."
python3 -m pip install --upgrade pip

# Install PyTorch with CUDA 12.1 support first (compatible with CUDA 12.8)
echo "Installing PyTorch with CUDA support..."
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
echo "Installing Python dependencies..."
python3 -m pip install -r requirements.txt

# Verify installation
echo ""
echo "Verifying installation..."
python3 << EOF
import sys
try:
    import torch
    import fastapi
    from diffusers import QwenImageEditPipeline
    print("✓ All required packages installed successfully")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA is available: {torch.cuda.device_count()} GPU(s)")
    else:
        print("⚠ Warning: CUDA is not available")
        sys.exit(1)
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)
EOF

echo ""
echo "=========================================="
echo "Installation completed successfully!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Configure environment variables (see .env.example)"
echo "2. Run './start.sh' to start the service"
echo ""

