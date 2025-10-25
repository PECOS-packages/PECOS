#!/bin/bash
# Copyright 2025 The PECOS Developers
#
# CUDA Setup Script for PECOS GPU Simulators
# This script installs CUDA Toolkit and Python packages required for CuStateVec and MPS simulators
#
# Usage: ./scripts/setup_cuda.sh [OPTIONS]
#   DO NOT run with sudo - the script will use sudo only when needed
#
# Options:
#   --cuda-version VERSION    Specify CUDA version (12 or 13, default: 13)
#   --skip-toolkit           Skip CUDA Toolkit installation (only install Python packages)
#   --dry-run                Show what would be done without making changes
#   --help                   Show this help message

set -e  # Exit on error

# Check if running as root/sudo
if [ "$EUID" -eq 0 ]; then
    echo "ERROR: This script should NOT be run with sudo or as root."
    echo ""
    echo "The script will automatically use sudo for commands that need it."
    echo "Running the entire script as root prevents access to your user's uv installation."
    echo ""
    echo "Please run as your normal user:"
    echo "  ./scripts/setup_cuda.sh"
    echo ""
    exit 1
fi

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CUDA_VERSION=13
SKIP_TOOLKIT=false
DRY_RUN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --skip-toolkit)
            SKIP_TOOLKIT=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            echo "CUDA Setup Script for PECOS"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --cuda-version VERSION    Specify CUDA version (12 or 13, default: 13)"
            echo "  --skip-toolkit           Skip CUDA Toolkit installation"
            echo "  --dry-run                Show what would be done without making changes"
            echo "  --help                   Show this help message"
            echo ""
            echo "For detailed documentation, see docs/user-guide/cuda-setup.md"
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate CUDA version
if [[ "$CUDA_VERSION" != "12" && "$CUDA_VERSION" != "13" ]]; then
    echo -e "${RED}Error: CUDA version must be 12 or 13${NC}"
    exit 1
fi

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}PECOS CUDA Setup Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "This script will install:"
echo "  - CUDA Toolkit ${CUDA_VERSION} (system-level)"
echo "  - Python packages: cupy-cuda${CUDA_VERSION}x, cuquantum-python-cu${CUDA_VERSION}, pytket-cutensornet"
echo ""

if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}DRY RUN MODE - No changes will be made${NC}"
    echo ""
fi

# Function to print status messages
print_status() {
    echo -e "${GREEN}[OK]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to run command (or skip in dry-run mode)
run_cmd() {
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN]${NC} Would run: $*"
    else
        "$@"
    fi
}

echo "========================================="
echo "Step 1: Checking Prerequisites"
echo "========================================="
echo ""

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_error "This script only supports Linux systems"
    echo "For other systems, see docs/user-guide/cuda-setup.md"
    exit 1
fi
print_status "Running on Linux"

# Check for NVIDIA GPU
echo ""
print_info "Checking for NVIDIA GPU..."
if command_exists nvidia-smi; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo "")
    if [ -n "$GPU_INFO" ]; then
        print_status "NVIDIA GPU detected:"
        echo "  $GPU_INFO" | sed 's/^/    /'

        # Check driver CUDA compatibility (shown in nvidia-smi header)
        DRIVER_CUDA=$(nvidia-smi | grep "CUDA Version" | sed -n 's/.*CUDA Version: \([0-9]\+\)\..*/\1/p')
        if [ -n "$DRIVER_CUDA" ] && [ "$DRIVER_CUDA" -ge "$CUDA_VERSION" ]; then
            print_status "Driver supports CUDA $DRIVER_CUDA (>= $CUDA_VERSION required)"
        elif [ -n "$DRIVER_CUDA" ]; then
            print_warning "Driver supports CUDA $DRIVER_CUDA, but CUDA $CUDA_VERSION requested"
            echo "  You may need to update your NVIDIA drivers"
        fi
    else
        print_error "NVIDIA GPU found but nvidia-smi returned no GPU info"
        exit 1
    fi
else
    print_error "NVIDIA GPU not detected (nvidia-smi not found)"
    echo ""
    echo "Please install NVIDIA drivers first:"
    echo "  sudo apt update"
    echo "  sudo apt install nvidia-driver-550  # or latest version"
    echo "  sudo reboot"
    exit 1
fi

# Check for uv
echo ""
print_info "Checking for uv package manager..."
if ! command_exists uv; then
    print_error "uv package manager not found"
    echo ""
    echo "Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
print_status "uv package manager found: $(uv --version)"

echo ""
echo "========================================="
echo "Step 2: CUDA Toolkit Installation"
echo "========================================="
echo ""

if [ "$SKIP_TOOLKIT" = true ]; then
    print_info "Skipping CUDA Toolkit installation (--skip-toolkit flag)"
else
    # Check if CUDA Toolkit is already installed
    print_info "Checking for CUDA Toolkit ${CUDA_VERSION}..."

    CUDA_INSTALLED=false
    if command_exists nvcc; then
        NVCC_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\)\..*/\1/p')
        if [ "$NVCC_VERSION" = "$CUDA_VERSION" ]; then
            print_status "CUDA Toolkit ${CUDA_VERSION} is already installed"
            nvcc --version | grep "release"
            CUDA_INSTALLED=true
        else
            print_warning "CUDA Toolkit version $NVCC_VERSION found, but version $CUDA_VERSION requested"
            echo "  Continuing with installation of CUDA $CUDA_VERSION..."
        fi
    fi

    if [ "$CUDA_INSTALLED" = false ]; then
        print_info "CUDA Toolkit ${CUDA_VERSION} not found, installing..."
        echo ""

        # Check if running as root/sudo
        if [ "$EUID" -ne 0 ]; then
            print_info "This step requires sudo privileges for system package installation"
        fi

        # Add NVIDIA CUDA repository if not already added
        print_info "Adding NVIDIA CUDA repository..."

        # Detect Ubuntu/Pop!_OS version
        if [ -f /etc/os-release ]; then
            . /etc/os-release
            OS_VERSION=$(echo "$VERSION_ID" | tr -d '.')

            # Map Ubuntu/Pop!_OS versions
            case "$OS_VERSION" in
                2004) UBUNTU_VERSION="ubuntu2004" ;;
                2204) UBUNTU_VERSION="ubuntu2204" ;;
                2404) UBUNTU_VERSION="ubuntu2404" ;;
                *)
                    print_warning "Unknown Ubuntu version: $VERSION_ID, trying ubuntu2404"
                    UBUNTU_VERSION="ubuntu2404"
                    ;;
            esac

            print_info "Detected $NAME $VERSION_ID (using $UBUNTU_VERSION repository)"
        else
            print_warning "Cannot detect OS version, using ubuntu2404 repository"
            UBUNTU_VERSION="ubuntu2404"
        fi

        # Download and install CUDA keyring
        KEYRING_DEB="cuda-keyring_1.1-1_all.deb"
        KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${UBUNTU_VERSION}/x86_64/${KEYRING_DEB}"

        if [ ! -f "/tmp/${KEYRING_DEB}" ]; then
            print_info "Downloading CUDA repository keyring..."
            run_cmd wget -q -O "/tmp/${KEYRING_DEB}" "$KEYRING_URL"
        fi

        print_info "Installing CUDA repository keyring..."
        run_cmd sudo dpkg -i "/tmp/${KEYRING_DEB}"

        print_info "Updating package lists..."
        run_cmd sudo apt update

        # Install CUDA Toolkit
        print_info "Installing CUDA Toolkit ${CUDA_VERSION}..."
        echo "  This may take several minutes..."
        run_cmd sudo apt install -y "cuda-toolkit-${CUDA_VERSION}"

        print_status "CUDA Toolkit ${CUDA_VERSION} installed successfully"

        # Add to PATH
        CUDA_PATH="/usr/local/cuda-${CUDA_VERSION}"
        BASHRC="$HOME/.bashrc"

        print_info "Checking PATH configuration..."
        if grep -q "cuda-${CUDA_VERSION}/bin" "$BASHRC" 2>/dev/null; then
            print_status "CUDA already in PATH configuration"
        else
            print_info "Adding CUDA to PATH in ~/.bashrc..."
            if [ "$DRY_RUN" = false ]; then
                echo "" >> "$BASHRC"
                echo "# CUDA ${CUDA_VERSION} paths (added by PECOS setup script)" >> "$BASHRC"
                echo "export PATH=\"${CUDA_PATH}/bin:\$PATH\"" >> "$BASHRC"
                echo "export LD_LIBRARY_PATH=\"${CUDA_PATH}/lib64:\$LD_LIBRARY_PATH\"" >> "$BASHRC"
                print_status "CUDA paths added to ~/.bashrc"
                print_warning "Please run 'source ~/.bashrc' or restart your shell to update PATH"
            fi
        fi

        # Export for current session
        export PATH="${CUDA_PATH}/bin:$PATH"
        export LD_LIBRARY_PATH="${CUDA_PATH}/lib64:$LD_LIBRARY_PATH"
    fi
fi

echo ""
echo "========================================="
echo "Step 3: Python CUDA Packages"
echo "========================================="
echo ""

# Determine package names based on CUDA version
if [ "$CUDA_VERSION" = "13" ]; then
    CUPY_PACKAGE="cupy-cuda13x"
    CUQUANTUM_PACKAGE="cuquantum-python-cu13"
elif [ "$CUDA_VERSION" = "12" ]; then
    CUPY_PACKAGE="cupy-cuda12x"
    CUQUANTUM_PACKAGE="cuquantum-python-cu12"
fi

PYTKET_PACKAGE="pytket-cutensornet"

# Function to check if Python package is installed
check_python_package() {
    uv pip list 2>/dev/null | grep -q "^$1 "
}

# Check and install CuPy
print_info "Checking for $CUPY_PACKAGE..."
if check_python_package "$CUPY_PACKAGE"; then
    CUPY_VERSION=$(uv pip list 2>/dev/null | grep "^$CUPY_PACKAGE " | awk '{print $2}')
    print_status "$CUPY_PACKAGE $CUPY_VERSION is already installed"
else
    print_info "Installing $CUPY_PACKAGE>=13.0.0..."
    run_cmd uv pip install "$CUPY_PACKAGE>=13.0.0"
    print_status "$CUPY_PACKAGE installed successfully"
fi

echo ""

# Check and install cuQuantum Python
print_info "Checking for $CUQUANTUM_PACKAGE..."
if check_python_package "$CUQUANTUM_PACKAGE"; then
    CUQUANTUM_VERSION=$(uv pip list 2>/dev/null | grep "^$CUQUANTUM_PACKAGE " | awk '{print $2}')
    print_status "$CUQUANTUM_PACKAGE $CUQUANTUM_VERSION is already installed"
else
    print_info "Installing $CUQUANTUM_PACKAGE>=25.3.0..."
    run_cmd uv pip install "$CUQUANTUM_PACKAGE>=25.3.0"
    print_status "$CUQUANTUM_PACKAGE installed successfully"
fi

echo ""

# Check and install pytket-cutensornet
print_info "Checking for $PYTKET_PACKAGE..."
if check_python_package "$PYTKET_PACKAGE"; then
    PYTKET_VERSION=$(uv pip list 2>/dev/null | grep "^$PYTKET_PACKAGE " | awk '{print $2}')
    print_status "$PYTKET_PACKAGE $PYTKET_VERSION is already installed"
else
    print_info "Installing $PYTKET_PACKAGE>=0.12.0..."
    run_cmd uv pip install "$PYTKET_PACKAGE>=0.12.0"
    print_status "$PYTKET_PACKAGE installed successfully"
fi

echo ""
echo "========================================="
echo "Step 4: Install PECOS with CUDA Support"
echo "========================================="
echo ""

# Find PECOS quantum-pecos directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PECOS_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
QUANTUM_PECOS_DIR="$PECOS_ROOT/python/quantum-pecos"

if [ -d "$QUANTUM_PECOS_DIR" ]; then
    print_info "Installing PECOS with CUDA extras..."
    cd "$QUANTUM_PECOS_DIR"
    run_cmd uv pip install -e ".[cuda]"
    print_status "PECOS installed with CUDA support"
else
    print_warning "quantum-pecos directory not found at $QUANTUM_PECOS_DIR"
    echo "  Skipping PECOS installation"
fi

echo ""
echo "========================================="
echo "Step 5: Verification"
echo "========================================="
echo ""

if [ "$DRY_RUN" = true ]; then
    print_info "Skipping verification in dry-run mode"
    echo ""
    echo -e "${GREEN}=========================================${NC}"
    echo -e "${GREEN}Dry Run Complete${NC}"
    echo -e "${GREEN}=========================================${NC}"
    echo ""
    echo "Re-run without --dry-run to perform actual installation"
    exit 0
fi

VERIFICATION_FAILED=false

# Test 1: CUDA Toolkit
print_info "Test 1: Verifying CUDA Toolkit..."
if command_exists nvcc; then
    NVCC_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\)\..*/\1/p')
    if [ "$NVCC_VERSION" = "$CUDA_VERSION" ]; then
        print_status "CUDA Toolkit ${CUDA_VERSION} verified"
    else
        print_warning "CUDA Toolkit version mismatch: found $NVCC_VERSION, expected $CUDA_VERSION"
        print_info "You may need to restart your shell and run: source ~/.bashrc"
    fi
else
    print_warning "nvcc not found in PATH"
    print_info "You may need to restart your shell and run: source ~/.bashrc"
fi

echo ""

# Test 2: CuPy
print_info "Test 2: Testing CuPy..."
CUPY_TEST=$(python3 -c "
import sys
try:
    import cupy as cp
    print(f'CuPy {cp.__version__}')
    print(f'CUDA available: {cp.cuda.is_available()}')
    if cp.cuda.is_available():
        print(f'CUDA runtime version: {cp.cuda.runtime.runtimeGetVersion()}')
        sys.exit(0)
    else:
        sys.exit(1)
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
" 2>&1)

if [ $? -eq 0 ]; then
    print_status "CuPy working correctly:"
    echo "$CUPY_TEST" | sed 's/^/    /'
else
    print_error "CuPy test failed:"
    echo "$CUPY_TEST" | sed 's/^/    /'
    VERIFICATION_FAILED=true
fi

echo ""

# Test 3: cuQuantum
print_info "Test 3: Testing cuQuantum..."
CUQUANTUM_TEST=$(python3 -c "
import sys
try:
    from cuquantum import custatevec
    print('cuStateVec imported successfully')
    sys.exit(0)
except Exception as e:
    print(f'Error: {e}')
    sys.exit(1)
" 2>&1)

if [ $? -eq 0 ]; then
    print_status "cuQuantum working correctly:"
    echo "$CUQUANTUM_TEST" | sed 's/^/    /'
else
    print_error "cuQuantum test failed:"
    echo "$CUQUANTUM_TEST" | sed 's/^/    /'
    VERIFICATION_FAILED=true
fi

echo ""

# Test 4: PECOS Simulators
print_info "Test 4: Testing PECOS GPU simulators..."
PECOS_TEST=$(python3 -c "
import sys
try:
    from pecos.simulators import CuStateVec, MPS

    # Test CuStateVec
    try:
        sim = CuStateVec(2)
        print('CuStateVec: Working')
    except Exception as e:
        print(f'CuStateVec: Failed - {e}')
        sys.exit(1)

    # Test MPS availability
    try:
        from pytket.extensions.cutensornet import simulate
        print('MPS (pytket-cutensornet): Working')
    except Exception as e:
        print(f'MPS: Failed - {e}')
        sys.exit(1)

    sys.exit(0)
except ImportError as e:
    print(f'Import error: {e}')
    sys.exit(1)
" 2>&1)

if [ $? -eq 0 ]; then
    print_status "PECOS GPU simulators working correctly:"
    echo "$PECOS_TEST" | sed 's/^/    /'
else
    print_error "PECOS GPU simulators test failed:"
    echo "$PECOS_TEST" | sed 's/^/    /'
    VERIFICATION_FAILED=true
fi

echo ""
echo "========================================="
if [ "$VERIFICATION_FAILED" = true ]; then
    echo -e "${YELLOW}Setup Complete with Warnings${NC}"
    echo "========================================="
    echo ""
    print_warning "Some verification tests failed"
    echo ""
    echo "Troubleshooting tips:"
    echo "  1. Restart your shell or run: source ~/.bashrc"
    echo "  2. Check CUDA paths: echo \$PATH | grep cuda"
    echo "  3. Check library paths: echo \$LD_LIBRARY_PATH | grep cuda"
    echo "  4. See docs/user-guide/cuda-setup.md for detailed troubleshooting"
    echo ""
else
    echo -e "${GREEN}Setup Complete Successfully!${NC}"
    echo "========================================="
    echo ""
    print_status "All verification tests passed!"
    echo ""
    echo "CUDA support is now enabled for PECOS GPU simulators:"
    echo "  - CuStateVec: GPU-accelerated state vector simulator"
    echo "  - MPS: Matrix Product State simulator with cuTensorNet"
    echo ""
    echo "You can now run GPU simulator tests:"
    echo "  cd python/quantum-pecos"
    echo "  uv run pytest tests/pecos/integration/state_sim_tests/test_statevec.py -v"
    echo ""
fi

echo "For more information, see docs/user-guide/cuda-setup.md"
