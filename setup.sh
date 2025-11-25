#!/bin/bash
# Setup script for needle-ssm codebase
# This script sets up the virtual environment, installs dependencies, and builds the C++/CUDA extensions

set -e  # Exit on error

echo "Setting up needle-ssm..."

# Check if CUDA is available
if [ -d "/usr/local/cuda" ]; then
    echo "✓ CUDA found at /usr/local/cuda"
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
else
    echo "⚠ CUDA not found in /usr/local/cuda - CPU backend will be built"
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet

# Install Python dependencies
echo "Installing Python dependencies..."
pip install numpy pybind11 --quiet

# Clean previous builds
echo "Cleaning previous builds..."
make clean > /dev/null 2>&1 || true
rm -rf build

# Build C++/CUDA extensions
echo "🔨 Building C++/CUDA extensions..."
make lib

# Verify installation
echo "Verifying installation..."
python -c "import sys; sys.path.insert(0, 'python'); import needle as ndl; print('✓ Needle imported successfully'); print('✓ Backend:', ndl.backend_selection.BACKEND); print('✓ Available devices:', ndl.all_devices())"

echo ""
echo "Setup complete! To use needle:"
echo "   1. Activate venv: source venv/bin/activate"
echo "   2. Add python/ to path: export PYTHONPATH=\$PWD/python:\$PYTHONPATH"
echo "   3. Or use: python -c \"import sys; sys.path.insert(0, 'python'); import needle as ndl\""

