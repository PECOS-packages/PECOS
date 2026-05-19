#!/usr/bin/env bash
# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

# Standalone GPU benchmark: PECOS (wgpu + cuQuantum) vs QuEST CUDA vs Qulacs GPU
#
# Builds QuEST with CUDA and Qulacs with GPU support from source,
# compiles standalone benchmark programs, runs them, and compares
# against PECOS Rust GPU simulators.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPS_DIR="$HOME/.pecos/deps"
BUILD_DIR="$SCRIPT_DIR/build"

QUEST_SRC="$DEPS_DIR/quest-v4.2.0"
QULACS_SRC="$DEPS_DIR/qulacs-0.6.13"

# ---------------------------------------------------------------------------
# Find CUDA
# ---------------------------------------------------------------------------

CUDA_PATH=""
for candidate in /usr/local/cuda "$HOME/.pecos/deps/cuda" "${CUDA_PATH:-}"; do
    if [ -n "$candidate" ] && [ -x "$candidate/bin/nvcc" ]; then
        CUDA_PATH="$candidate"
        break
    fi
done

if [ -z "$CUDA_PATH" ]; then
    echo "ERROR: CUDA not found. Install CUDA or set CUDA_PATH."
    exit 1
fi

echo "Using CUDA at: $CUDA_PATH"
export PATH="$CUDA_PATH/bin:$PATH"
echo "nvcc version: $(nvcc --version | tail -1)"
echo ""

# ---------------------------------------------------------------------------
# Check sources exist
# ---------------------------------------------------------------------------

missing=0
if [ ! -d "$QUEST_SRC" ]; then
    echo "ERROR: QuEST sources not found at $QUEST_SRC"
    missing=1
fi
if [ ! -d "$QULACS_SRC" ]; then
    echo "ERROR: Qulacs sources not found at $QULACS_SRC"
    missing=1
fi
if [ "$missing" -eq 1 ]; then
    echo ""
    echo "Run the following to download the dependencies:"
    echo "  scripts/native_bench/fetch_deps.sh"
    exit 1
fi

echo "=== GPU Benchmark: PECOS vs QuEST vs Qulacs ==="
echo ""

# ---------------------------------------------------------------------------
# Build QuEST with CUDA
# ---------------------------------------------------------------------------

echo "--- Building QuEST (CMake, Release, CUDA) ---"
QUEST_GPU_BUILD="$BUILD_DIR/quest_gpu"
mkdir -p "$QUEST_GPU_BUILD"
cmake -S "$QUEST_SRC" -B "$QUEST_GPU_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-march=native" \
    -DCMAKE_CXX_FLAGS="-march=native" \
    -DCMAKE_CUDA_FLAGS="-O3" \
    -DENABLE_MULTITHREADING=OFF \
    -DENABLE_CUDA=ON \
    -DENABLE_HIP=OFF \
    -DENABLE_DISTRIBUTION=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    2>&1 | tail -5
cmake --build "$QUEST_GPU_BUILD" -j "$(nproc)" 2>&1 | tail -5
echo "QuEST (CUDA) built."
echo ""

# ---------------------------------------------------------------------------
# Build Qulacs with GPU
# ---------------------------------------------------------------------------

echo "--- Building Qulacs (CMake, Release, CUDA) ---"
QULACS_GPU_BUILD="$BUILD_DIR/qulacs_gpu"
mkdir -p "$QULACS_GPU_BUILD"

BOOST_DIR="$DEPS_DIR/boost-1.83.0"
if [ ! -d "$BOOST_DIR" ]; then
    echo "ERROR: Boost not found at $BOOST_DIR"
    echo "Run: scripts/native_bench/fetch_deps.sh"
    exit 1
fi

cmake -S "$QULACS_SRC" -B "$QULACS_GPU_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-march=native" \
    -DCMAKE_CXX_FLAGS="-march=native -DEIGEN_NO_DEBUG" \
    -DCMAKE_CUDA_FLAGS="-O3" \
    -DBoost_INCLUDE_DIR="$BOOST_DIR" \
    -DUSE_OMP=OFF \
    -DUSE_GPU=Yes \
    -DUSE_MPI=OFF \
    -DUSE_TEST=OFF \
    -DUSE_PYTHON=OFF \
    -DCMAKE_CUDA_ARCHITECTURES=89 \
    2>&1 | tail -5
cmake --build "$QULACS_GPU_BUILD" -j "$(nproc)" --target csim_static cppsim_static gpusim_static 2>&1 | tail -5
echo "Qulacs (CUDA) built."
echo ""

# ---------------------------------------------------------------------------
# Locate built libraries
# ---------------------------------------------------------------------------

# QuEST CUDA
QUEST_GPU_LIB="$(find "$QUEST_GPU_BUILD" -name 'libQuEST.a' | head -1)"
if [ -z "$QUEST_GPU_LIB" ]; then
    echo "ERROR: Could not find libQuEST.a in $QUEST_GPU_BUILD"
    exit 1
fi
QUEST_GPU_LIB_DIR="$(dirname "$QUEST_GPU_LIB")"

QUEST_INC_GEN="$QUEST_GPU_BUILD/include"
QUEST_INC_SRC="$QUEST_SRC/quest/include"
QUEST_INC_ROOT="$QUEST_SRC"

# Qulacs GPU: CMake puts archives at ${PROJECT_BINARY_DIR}/../lib (i.e. build/lib/)
QULACS_GPU_CSIM="$(find "$BUILD_DIR" -name 'libcsim_static.a' | head -1)"
QULACS_GPU_CPPSIM="$(find "$BUILD_DIR" -name 'libcppsim_static.a' | head -1)"
QULACS_GPU_GPUSIM="$(find "$BUILD_DIR" -name 'libgpusim_static.a' | head -1)"
if [ -z "$QULACS_GPU_CSIM" ] || [ -z "$QULACS_GPU_CPPSIM" ] || [ -z "$QULACS_GPU_GPUSIM" ]; then
    echo "ERROR: Could not find Qulacs GPU static libraries"
    echo "  csim: $QULACS_GPU_CSIM"
    echo "  cppsim: $QULACS_GPU_CPPSIM"
    echo "  gpusim: $QULACS_GPU_GPUSIM"
    exit 1
fi

QULACS_INC="$QULACS_SRC/src"
QULACS_CPPSIM_INC="$QULACS_SRC/include"
QULACS_EIGEN_INC="$QULACS_SRC/include"
if [ ! -d "$QULACS_EIGEN_INC/Eigen" ]; then
    QULACS_EIGEN_INC="$DEPS_DIR/eigen-3.4.0"
fi

# ---------------------------------------------------------------------------
# Compile GPU benchmark programs
# ---------------------------------------------------------------------------

echo "--- Compiling bench_quest (CUDA) ---"
# QuEST CUDA needs nvcc for linking since libQuEST.a contains CUDA objects
nvcc -O3 -std=c++14 \
    -I"$QUEST_INC_GEN" -I"$QUEST_INC_SRC" -I"$QUEST_INC_ROOT" \
    -Xcompiler "-march=native" \
    "$SCRIPT_DIR/bench_quest.c" \
    -L"$QUEST_GPU_LIB_DIR" -lQuEST \
    -lcudart -lcurand \
    -lstdc++ -lm \
    -o "$BUILD_DIR/bench_quest_gpu"
echo "Compiled."

echo "--- Compiling bench_qulacs_gpu ---"
nvcc -O3 -std=c++14 \
    -I"$QULACS_INC" -I"$QULACS_CPPSIM_INC" -I"$QULACS_EIGEN_INC" -I"$BOOST_DIR" \
    -Xcompiler "-march=native" \
    -D_USE_GPU -DEIGEN_NO_DEBUG \
    "$SCRIPT_DIR/bench_qulacs_gpu.cpp" \
    "$QULACS_GPU_CPPSIM" "$QULACS_GPU_GPUSIM" "$QULACS_GPU_CSIM" \
    -lcudart -lcurand -lcublas \
    -lm \
    -o "$BUILD_DIR/bench_qulacs_gpu"
echo "Compiled."

# cuStateVec standalone benchmark
CUQUANTUM_DIR="$(ls -d "$DEPS_DIR"/cuquantum-* 2>/dev/null | sort -V | tail -1)"
if [ -n "$CUQUANTUM_DIR" ] && [ -d "$CUQUANTUM_DIR" ]; then
    echo "--- Compiling bench_custatevec ---"
    nvcc -O3 -std=c++14 \
        -I"$CUQUANTUM_DIR/include" \
        -Xcompiler "-march=native" \
        "$SCRIPT_DIR/bench_custatevec.cu" \
        -L"$CUQUANTUM_DIR/lib" -lcustatevec \
        -lcudart \
        -Xlinker "-rpath,$CUQUANTUM_DIR/lib" \
        -o "$BUILD_DIR/bench_custatevec"
    echo "Compiled."
    HAS_CUSTATEVEC=1
else
    echo "--- Skipping bench_custatevec (cuQuantum not found) ---"
    HAS_CUSTATEVEC=0
fi
echo ""

# ---------------------------------------------------------------------------
# Build PECOS GPU benchmark (Rust, wgpu + cuQuantum)
# ---------------------------------------------------------------------------

echo "--- Building PECOS GPU benchmark (Rust, wgpu + cuQuantum) ---"
PECOS_BENCH_DIR="$SCRIPT_DIR/bench_pecos"
(cd "$PECOS_BENCH_DIR" && RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native" \
    cargo build --locked --release --features gpu,cuquantum 2>&1 | tail -5)
PECOS_BIN="$PECOS_BENCH_DIR/target/release/bench_pecos"
echo "PECOS GPU benchmark built."
echo ""

# ---------------------------------------------------------------------------
# Run GPU benchmarks
# ---------------------------------------------------------------------------

echo "--- Running QuEST CUDA benchmark ---"
"$BUILD_DIR/bench_quest_gpu" | tee "$BUILD_DIR/quest_gpu_results.txt"
echo ""

echo "--- Running Qulacs GPU benchmark ---"
"$BUILD_DIR/bench_qulacs_gpu" | tee "$BUILD_DIR/qulacs_gpu_results.txt"
echo ""

if [ "${HAS_CUSTATEVEC:-0}" -eq 1 ]; then
    echo "--- Running cuStateVec benchmark ---"
    LD_LIBRARY_PATH="$CUQUANTUM_DIR/lib:${LD_LIBRARY_PATH:-}" \
        "$BUILD_DIR/bench_custatevec" | tee "$BUILD_DIR/custatevec_results.txt"
    echo ""
fi

echo "--- Running PECOS GPU benchmark ---"
"$PECOS_BIN" | tee "$BUILD_DIR/pecos_gpu_results.txt"
echo ""

# ---------------------------------------------------------------------------
# Comparison summary
# ---------------------------------------------------------------------------

echo "============================================================"
echo "                  GPU COMPARISON SUMMARY"
echo "============================================================"
echo ""
echo "QuEST CUDA results:"
cat "$BUILD_DIR/quest_gpu_results.txt"
echo ""
echo "Qulacs GPU results:"
cat "$BUILD_DIR/qulacs_gpu_results.txt"
echo ""
if [ "${HAS_CUSTATEVEC:-0}" -eq 1 ]; then
    echo "cuStateVec standalone results:"
    cat "$BUILD_DIR/custatevec_results.txt"
    echo ""
fi
echo "PECOS GPU results:"
cat "$BUILD_DIR/pecos_gpu_results.txt"
echo ""
echo "============================================================"
echo "Done. Full outputs saved in: $BUILD_DIR/"
