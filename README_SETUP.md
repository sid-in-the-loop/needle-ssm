# Needle-SSM Setup Guide

This guide will help you set up the needle-ssm codebase with CUDA support.

## Prerequisites

- Python 3.8+ (tested with Python 3.13.5)
- CMake 3.5+ (tested with CMake 3.26.5)
- GCC/G++ compiler with C++11 support
- CUDA Toolkit 12.x (optional, for GPU support)
- NVIDIA GPU with CUDA support (optional, for GPU backend)

## Quick Setup

### Option 1: Automated Setup Script

```bash
# Make setup script executable
chmod +x setup.sh

# Run setup script
./setup.sh
```

### Option 2: Manual Setup

#### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

#### 2. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
# Or manually:
pip install numpy pybind11
```

#### 3. Set Up CUDA (if available)

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

#### 4. Build C++/CUDA Extensions

```bash
make clean
make lib
```

#### 5. Verify Installation

```bash
python -c "import sys; sys.path.insert(0, 'python'); import needle as ndl; print('Success!'); print('Backend:', ndl.backend_selection.BACKEND); print('Devices:', ndl.all_devices())"
```

## Usage

### Basic Usage

```python
import sys
sys.path.insert(0, 'python')
import needle as ndl
import numpy as np

# Create a tensor on CPU
x = ndl.Tensor(np.array([1.0, 2.0, 3.0]))
print(x)

# Create a tensor on GPU (if CUDA is available)
if ndl.cuda().enabled():
    x_gpu = ndl.Tensor(np.array([1.0, 2.0, 3.0]), device=ndl.cuda())
    print(x_gpu)
```

### Using in Your Scripts

Add the python directory to your Python path:

```bash
export PYTHONPATH=$PWD/python:$PYTHONPATH
python your_script.py
```

Or in your Python scripts:

```python
import sys
sys.path.insert(0, 'path/to/needle-ssm/python')
import needle as ndl
```

## Troubleshooting

### CMake Policy Error

If you see `Policy "CMP0146" is not known`, this has been fixed in the CMakeLists.txt. Make sure you have the latest version.

### Python Version Mismatch

If CMake finds the wrong Python version:
1. Make sure venv is activated: `source venv/bin/activate`
2. Clean build: `make clean && rm -rf build`
3. Rebuild: `make lib`

### CUDA Not Found

If CUDA is not found:
- Check CUDA installation: `ls /usr/local/cuda`
- Add to PATH: `export PATH=/usr/local/cuda/bin:$PATH`
- Verify: `nvcc --version`

### Import Errors

If you get import errors:
- Make sure `python/` is in your Python path
- Verify the `.so` files exist: `ls python/needle/backend_ndarray/*.so`
- Check Python version matches: `.so` files should match your Python version (e.g., `cpython-313` for Python 3.13)

## File Structure

```
needle-ssm/
├── python/                    # Python package
│   └── needle/
│       ├── backend_ndarray/   # Compiled .so files go here
│       ├── nn/                # Neural network modules
│       ├── ops/               # Operations
│       └── ...
├── src/                       # C++/CUDA source files
│   ├── ndarray_backend_cpu.cc
│   └── ndarray_backend_cuda.cu
├── build/                     # CMake build directory
├── CMakeLists.txt            # CMake configuration
├── Makefile                  # Build commands
├── setup.sh                  # Automated setup script
└── requirements.txt          # Python dependencies
```

## Notes

- The codebase builds both CPU and CUDA backends if CUDA is available
- The `.so` files are built for your specific Python version
- If you switch Python versions, you'll need to rebuild: `make clean && make lib`

