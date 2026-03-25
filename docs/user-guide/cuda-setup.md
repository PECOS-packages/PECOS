# CUDA Setup Guide for GPU Simulators

This guide provides detailed instructions for setting up NVIDIA CUDA support to use GPU-accelerated quantum simulators in PECOS.

## Overview

PECOS supports GPU-accelerated quantum simulation through two approaches:

### Python cuQuantum Bindings (via cupy/cuquantum-python)
- **CuStateVec**: GPU-accelerated state vector simulator
- **MPS**: Matrix Product State simulator using cuTensorNet

### Rust cuQuantum Bindings (via pecos-rslib-cuda)
- **CudaStateVec**: GPU-accelerated state vector simulator (Rust bindings)
- **CudaStabilizer**: GPU-accelerated stabilizer simulator (Clifford-only, scales to 1000s of qubits)
- **CuTensorNet**: Tensor network handle for contractions
- **CuDensityMat**: Density matrix simulator for open quantum systems

The Rust bindings provide:
- Direct integration with PECOS's quantum-pecos framework
- Comprehensive gate coverage (Clifford + arbitrary rotations)
- Seed-based reproducibility
- No Python package dependencies beyond pecos-rslib-cuda

Both approaches require:
- NVIDIA GPU hardware
- CUDA Toolkit (system-level installation)
- cuQuantum SDK (for Rust bindings) or Python packages (for Python bindings)

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

<!--skip-if-no-cuda-->
```python
# Test CuPy
import cupy as cp

print(f"CuPy version: {cp.__version__}")
print(f"CUDA available: {cp.cuda.is_available()}")

# Test cuQuantum
from cuquantum.bindings import custatevec

print(f"cuStateVec available: {custatevec is not None}")
```

### Test PECOS Simulators

<!--skip-if-no-cuda-->
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

<!--skip-if-no-cuda-->
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
2. Check [pytket-cutensornet GitHub Issues](https://github.com/Quantinuum/pytket-cutensornet/issues)
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

| Simulator | Hardware | Qubits | Gates | Speed | Installation |
|-----------|----------|--------|-------|-------|--------------|
| StateVec (CPU) | Any | ~25 | All | Baseline | Easy |
| Qulacs (CPU) | Any | ~28 | All | 2-3x faster | Easy |
| CuStateVec (Python) | NVIDIA GPU | ~30 | All | 10-50x faster | Medium |
| CudaStateVec (Rust) | NVIDIA GPU | ~30 | All | 10-50x faster | Complex |
| CudaStabilizer (Rust) | NVIDIA GPU | 1000s | Clifford only | Very fast | Complex |
| MPS (GPU) | NVIDIA GPU | 50+ | All | Varies | Medium |

## GPU Simulators: Python vs Rust

PECOS provides GPU acceleration through multiple backends:

### Python GPU Simulators (cupy/cuquantum-python)

**Status**: Fully Working

- **CuStateVec**: GPU-accelerated state vector simulator using NVIDIA cuQuantum
- **MPS**: Matrix Product State simulator using pytket-cutensornet and cuTensorNet
- **CUDA Version**: Supports CUDA 12 and CUDA 13
- **Setup**: Install Python packages as described above

### Rust GPU Simulators (pecos-rslib-cuda)

**Status**: Fully Working

- **CudaStateVec**: GPU-accelerated state vector simulator (~30 qubits)
- **CudaStabilizer**: GPU-accelerated stabilizer simulator (Clifford-only, 1000s of qubits)
- **CuTensorNet**: Tensor network handle for advanced contractions
- **CuDensityMat**: Density matrix simulator for noisy/open quantum systems
- **CUDA Version**: Requires CUDA 12+ and cuQuantum SDK
- **Setup**: See "Rust cuQuantum Bindings Setup" section below

The Rust bindings provide direct cuQuantum integration without Python package dependencies. They are particularly useful for:
- Stabilizer simulations with many qubits (CudaStabilizer)
- Integration with quantum-pecos's HybridEngine
- Reproducible simulations with seed support

### Rust GPU Simulators (QuEST)

**Status**: Limited Support (CPU-only with CUDA 13)

- **Engine**: QuEST (Quantum Exact Simulation Toolkit)
- **CUDA Version**: Requires CUDA 11 or 12 (incompatible with CUDA 13)
- **Issue**: QuEST uses deprecated `thrust::unary_function` and `thrust::binary_function` classes that were removed in modern CUDA/Thrust versions
- **Workaround**: Automatically falls back to CPU-only QuEST build
- **Impact**: Minimal - use CudaStateVec or Python CuStateVec instead

The Rust QuEST simulator is currently incompatible with CUDA 13 due to deprecated thrust classes.

## Rust cuQuantum Bindings Setup

To use the Rust-based CUDA simulators (CudaStateVec, CudaStabilizer), you need:

### Requirements

1. **CUDA Toolkit 12+** (system installation)
2. **cuQuantum SDK** (download from NVIDIA)

### Installing cuQuantum SDK

1. Download from [NVIDIA cuQuantum](https://developer.nvidia.com/cuquantum-sdk)
2. Extract to a known location (e.g., `/opt/nvidia/cuquantum`)
3. Set environment variables:
   ```bash
   export CUQUANTUM_ROOT=/opt/nvidia/cuquantum
   export LD_LIBRARY_PATH=$CUQUANTUM_ROOT/lib:$LD_LIBRARY_PATH
   ```

### Building pecos-rslib-cuda

```bash
# From PECOS repository root
cd python/pecos-rslib-cuda
maturin develop --release
```

### Using Rust CUDA Simulators

```python
# Check availability
from pecos_rslib_cuda import is_cuquantum_available

print(f"cuQuantum available: {is_cuquantum_available()}")

# State vector simulator (up to ~30 qubits)
from pecos.simulators import CudaStateVec

sim = CudaStateVec(10)
sim.run_gate("H", [0])
sim.run_gate("CX", [(0, 1)])
results = sim.run_gate("Measure", [0, 1])

# Stabilizer simulator (Clifford-only, scales to 1000s of qubits)
from pecos.simulators import CudaStabilizer

sim = CudaStabilizer(1000)
sim.run_gate("H", [0])
for i in range(100):
    sim.run_gate("CX", [(i, i + 1)])
results = sim.run_gate("Measure", list(range(100)))

# Using with QuantumSimulator
from pecos.simulators.quantum_simulator import QuantumSimulator

qsim = QuantumSimulator(backend="CudaStateVec")
qsim.init(4)

# Direct access to cuQuantum components
from pecos_rslib_cuda import CuTensorNet, CuDensityMat

print(f"cuTensorNet version: {CuTensorNet.version()}")
print(f"cuDensityMat version: {CuDensityMat.version()}")
```

### Choosing Between Python and Rust Bindings

| Feature | Python (cupy/cuquantum-python) | Rust (pecos-rslib-cuda) |
|---------|-------------------------------|------------------------|
| State Vector | CuStateVec | CudaStateVec |
| Stabilizer | - | CudaStabilizer |
| MPS/Tensor Network | MPS (pytket) | CuTensorNet (handle only) |
| Density Matrix | - | CuDensityMat |
| Setup Complexity | Easier (pip install) | Requires cuQuantum SDK |
| Dependencies | cupy, cuquantum-python | None beyond pecos-rslib-cuda |
| Seed Support | Varies | Full support |
| HybridEngine Integration | Yes | Yes |

## Summary

To use GPU simulators in PECOS:

### Option A: Python cuQuantum Bindings (Easier Setup)

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
   <!--skip-if-no-cuda-->
   ```python
   from pecos.simulators import CuStateVec

   sim = CuStateVec(2)  # Should work with cupy + cuquantum!
   ```

   If you also installed `pytket-cutensornet`:
   <!--skip-if-no-cuda-->
   ```python
   from pecos.simulators import MPS

   sim = MPS(2)  # Should work with pytket-cutensornet!
   ```

### Option B: Rust cuQuantum Bindings (More Features)

1. **Verify NVIDIA GPU** (Compute Capability 7.0+)
2. **Install CUDA Toolkit 12+** (system-level)
3. **Install cuQuantum SDK** from NVIDIA
4. **Build pecos-rslib-cuda**:
   ```bash
   cd python/pecos-rslib-cuda
   maturin develop --release
   ```
5. **Verify GPU simulators**:
   ```python
   from pecos_rslib_cuda import is_cuquantum_available

   print(f"cuQuantum available: {is_cuquantum_available()}")

   from pecos.simulators import CudaStateVec, CudaStabilizer

   sim = CudaStateVec(4)  # State vector (~30 qubits max)
   sim = CudaStabilizer(1000)  # Stabilizer (1000s of qubits, Clifford only)
   ```

### Choosing an Approach

- **For most users**: Python cuQuantum bindings are easier to set up
- **For stabilizer simulations**: Use CudaStabilizer (Rust) for 1000s of qubits
- **For reproducibility**: Rust bindings have full seed support
- **For density matrices**: Use CuDensityMat (Rust) for open quantum systems

**Note**: If you see warnings about QuEST GPU compilation failing, this is expected with CUDA 13 and does not affect the cuQuantum-based simulators.
