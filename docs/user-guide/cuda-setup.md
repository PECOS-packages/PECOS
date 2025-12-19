# CUDA Setup Guide for GPU Simulators

This guide provides detailed instructions for setting up NVIDIA CUDA support to use GPU-accelerated quantum simulators in PECOS, specifically **CuStateVec** and **MPS** (Matrix Product State).

## Overview

PECOS supports GPU-accelerated quantum simulation through NVIDIA's cuQuantum SDK:

- **CuStateVec**: GPU-accelerated state vector simulator
- **MPS**: Matrix Product State simulator using cuTensorNet

Both simulators require:
- NVIDIA GPU hardware
- CUDA Toolkit (system-level installation)
- Python packages (cuQuantum, CuPy, pytket-cutensornet)

## System Requirements

### Hardware Requirements

- **NVIDIA GPU** with Compute Capability 7.0 or higher
  - To check your GPU: `nvidia-smi`
  - To check compute capability: Visit [NVIDIA's GPU Compute Capability List](https://developer.nvidia.com/cuda-gpus)

### Software Requirements

- **Operating System**: Linux (Ubuntu 20.04+, Pop!_OS, or other distributions)
  - Windows users: Use WSL2 (Windows Subsystem for Linux)
- **Python**: 3.10, 3.11, or 3.12
- **CUDA Toolkit**: Version 13.x (recommended) or 12.x

### Supported CUDA Versions

| CUDA Version | Support Status | Recommended |
|--------------|----------------|-------------|
| CUDA 13.x | Fully Supported | **Yes** (Latest) |
| CUDA 12.x | Fully Supported | Yes |
| CUDA 11.x | Deprecated | No (being phased out) |

**Note**: This guide focuses on CUDA 13.x as it's the latest and recommended version.

## Installation Guide

### Step 1: Verify GPU and Driver

First, ensure your NVIDIA GPU is detected and drivers are installed:

```bash
# Check GPU status
nvidia-smi
```

If `nvidia-smi` is not found, install NVIDIA drivers:

```bash
# Ubuntu/Pop!_OS
sudo apt update
sudo apt install nvidia-driver-550  # or latest version

# Reboot after installation
sudo reboot
```

### Step 2: Install CUDA Toolkit 13

The CUDA Toolkit must be installed at the system level (not as a Python package).

#### Option A: Using APT (Ubuntu/Pop!_OS)

```bash
# Add NVIDIA package repositories (if not already added)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install CUDA Toolkit 13
sudo apt install cuda-toolkit-13

# Add CUDA to PATH (add to ~/.bashrc or ~/.zshrc)
echo 'export PATH=/usr/local/cuda-13/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

#### Option B: Download from NVIDIA

1. Visit [NVIDIA CUDA Downloads](https://developer.nvidia.com/cuda-downloads)
2. Select your platform (Linux, x86_64, Ubuntu, version, deb/runfile)
3. Follow the installation instructions provided

### Step 3: Verify CUDA Installation

```bash
# Check CUDA version
nvcc --version

# Should show CUDA version 13.x
# Example output:
# cuda_compilation_tools: 13.0, release 13.0, V13.0.XXX
```

If `nvcc` is not found, ensure CUDA's bin directory is in your PATH.

### Step 4: Install Python Packages with uv

PECOS uses `uv` as the package manager. Install the CUDA-related Python packages:

```bash
# Install CUDA 13 packages
uv pip install cupy-cuda13x>=13.0.0
uv pip install cuquantum-python-cu13>=25.3.0
uv pip install pytket-cutensornet>=0.12.0
```

**Important**: Use packages matching your CUDA version:
- For CUDA 13: `cupy-cuda13x`, `cuquantum-python-cu13`
- For CUDA 12: `cupy-cuda12x`, `cuquantum-python-cu12`

### Step 5: Install PECOS with CUDA Support

#### Option A: Install from PyPI with CUDA extras

```bash
uv pip install quantum-pecos[cuda]
```

#### Option B: Install from source (for development)

```bash
# From the PECOS repository root
cd /path/to/PECOS

# Option 1: Use just commands (recommended)
just build-cuda  # Build with CUDA support
just devc        # Full dev cycle: clean + build-cuda + test
just devcl       # Dev cycle + linting

# Option 2: Manual installation
uv pip install -e "./python/quantum-pecos[all,cuda]"
```

## Verification

### Test CUDA Installation

```python
# Test CuPy
import cupy as cp

print(f"CuPy version: {cp.__version__}")
print(f"CUDA available: {cp.cuda.is_available()}")

# Test cuQuantum
from cuquantum import custatevec

print(f"cuStateVec available: {custatevec is not None}")
```

### Test PECOS Simulators

```python
from pecos.simulators import CuStateVec, MPS

# Test CuStateVec
try:
    sim = CuStateVec(2)
    print("SUCCESS: CuStateVec is working!")
except Exception as e:
    print(f"FAILED: CuStateVec failed: {e}")

# Test MPS
try:
    from pytket.extensions.cutensornet import simulate

    print("SUCCESS: MPS (pytket-cutensornet) is working!")
except Exception as e:
    print(f"FAILED: MPS failed: {e}")
```

### Run PECOS Tests

```bash
# Run tests for GPU simulators
uv run pytest python/quantum-pecos/tests/pecos/integration/state_sim_tests/test_statevec.py -v

# Tests with CuStateVec and MPS should pass (not skip)
```

## Package Versions

Current recommended versions (as of 2025):

| Package | Version | Release Date | Purpose |
|---------|---------|--------------|---------|
| cupy-cuda13x | 13.6.0+ | Aug 2025 | NumPy/SciPy for GPU |
| cuquantum-python-cu13 | 25.9.0+ | Sept 2025 | cuQuantum Python API |
| custatevec-cu13 | 1.10.0+ | Sept 2025 | State vector operations (included in cuquantum) |
| pytket-cutensornet | 0.12.0+ | 2025 | MPS simulator |

## Troubleshooting

### Common Issues

#### 1. `ImportError: libcudart.so.13 not found`

**Solution**: CUDA libraries are not in the library path.

```bash
# Add to ~/.bashrc
export LD_LIBRARY_PATH=/usr/local/cuda-13/lib64:$LD_LIBRARY_PATH
source ~/.bashrc
```

#### 2. `CuStateVec is None` or tests are skipped

**Solution**: Python packages not properly installed or CUDA Toolkit version mismatch.

```bash
# Verify installations
python -c "import cupy; print(cupy.__version__)"
python -c "from cuquantum import custatevec; print('OK')"

# Reinstall if needed
uv pip uninstall cupy-cuda13x cuquantum-python-cu13
uv pip install cupy-cuda13x cuquantum-python-cu13
```

#### 3. CUDA version mismatch errors

**Problem**: Mixing CUDA 12 and CUDA 13 packages.

**Solution**: Ensure consistency across all packages. Use either all CUDA 13 or all CUDA 12 packages.

```bash
# For CUDA 13 (recommended)
uv pip install cupy-cuda13x cuquantum-python-cu13

# For CUDA 12
uv pip install cupy-cuda12x cuquantum-python-cu12
```

#### 4. Out of memory errors

**Solution**: GPU memory is limited. Use smaller circuits or the MPS simulator for larger systems.

```python
# MPS can handle larger systems with less memory
from pecos.simulators import MPS

sim = MPS(num_qubits=20)  # Can go much larger than state vector
```

#### 5. Permission denied when installing CUDA Toolkit

**Solution**: CUDA Toolkit installation requires sudo/administrator privileges.

```bash
sudo apt install cuda-toolkit-13
```

### Getting Help

If you encounter issues:

1. Check [NVIDIA cuQuantum Documentation](https://docs.nvidia.com/cuda/cuquantum/latest/)
2. Check [pytket-cutensornet GitHub Issues](https://github.com/CQCL/pytket-cutensornet/issues)
3. Check [PECOS GitHub Issues](https://github.com/PECOS-packages/PECOS/issues)
4. Verify your GPU compute capability is 7.0 or higher

## Alternative: Using Conda

If you prefer using Conda instead of uv/pip, NVIDIA officially recommends it:

```bash
# Create conda environment
conda create -n pecos-cuda python=3.11
conda activate pecos-cuda

# Install cuQuantum via conda-forge
conda install -c conda-forge cuquantum-python cuda-version=13

# Install CuPy
conda install -c conda-forge cupy

# Install pytket-cutensornet
pip install pytket-cutensornet

# Install PECOS
pip install quantum-pecos
```

**Note**: When using Conda, there may be conflicts with Python virtual environments (venv) or uv. Choose one approach and stick with it.

## Performance Tips

1. **Use CuStateVec for exact simulation**: Up to ~30 qubits depending on GPU memory
2. **Use MPS for larger systems**: Can handle 50+ qubits with approximation
3. **Monitor GPU usage**: Use `nvidia-smi -l 1` to watch GPU utilization
4. **Batch multiple circuits**: Reduces overhead of data transfer to/from GPU

## Comparison: CPU vs GPU Simulators

| Simulator | Hardware | Qubits | Speed | Installation |
|-----------|----------|--------|-------|--------------|
| StateVec (CPU) | Any | ~25 | Baseline | Easy |
| Qulacs (CPU) | Any | ~28 | 2-3x faster | Easy |
| CuStateVec (GPU) | NVIDIA GPU | ~30 | 10-50x faster | Complex |
| MPS (GPU) | NVIDIA GPU | 50+ | Varies | Complex |

## GPU Simulators: Python vs Rust

PECOS provides GPU acceleration through two different backends:

### Python GPU Simulators (Recommended)

**Status**: Fully Working

- **CuStateVec**: GPU-accelerated state vector simulator using NVIDIA cuQuantum
- **MPS**: Matrix Product State simulator using pytket-cutensornet and cuTensorNet
- **CUDA Version**: Supports CUDA 12 and CUDA 13
- **Setup**: Install Python packages as described above

These are the **primary GPU simulators** that users should use. They provide excellent performance and are fully compatible with modern CUDA versions.

### Rust GPU Simulators (QuEST)

**Status**: Limited Support (CPU-only with CUDA 13)

- **Engine**: QuEST (Quantum Exact Simulation Toolkit)
- **CUDA Version**: Requires CUDA 11 or 12 (incompatible with CUDA 13)
- **Issue**: QuEST uses deprecated `thrust::unary_function` and `thrust::binary_function` classes that were removed in modern CUDA/Thrust versions
- **Workaround**: Automatically falls back to CPU-only QuEST build
- **Impact**: Minimal - Python GPU simulators (CuStateVec/MPS) provide better performance

The Rust QuEST simulator is currently incompatible with CUDA 13 due to deprecated `thrust::unary_function` and `thrust::binary_function` classes. However, this does not affect the recommended Python GPU simulators (CuStateVec and MPS).

## Summary

To use GPU simulators in PECOS:

1. **Verify NVIDIA GPU** (Compute Capability 7.0+)
2. **Install CUDA Toolkit 13** (system-level)
3. **Install Python packages**: `cupy-cuda13x`, `cuquantum-python-cu13`, `pytket-cutensornet`
4. **Install PECOS with `[cuda]` extras**:
   ```bash
   uv pip install quantum-pecos[cuda]
   # or for development:
   just build-cuda
   ```
5. **Verify GPU simulators**:
   ```python
   from pecos.simulators import CuStateVec, MPS

   sim = CuStateVec(2)  # Should work!
   sim = MPS(2)  # Should work!
   ```

For most users, **CUDA 13 with uv/pip** is recommended over Conda for better integration with PECOS's development workflow.

**Note**: If you see warnings about QuEST GPU compilation failing, this is expected with CUDA 13 and does not affect Python GPU simulators.
