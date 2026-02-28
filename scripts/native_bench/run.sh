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

# Standalone native benchmark: PECOS vs QuEST vs Qulacs
#
# Builds QuEST and Qulacs from source with their own CMake build systems,
# compiles standalone C/C++ benchmark programs, runs them, and compares
# the results against PECOS Rust criterion benchmarks.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPS_DIR="$HOME/.pecos/deps"
BUILD_DIR="$SCRIPT_DIR/build"

QUEST_SRC="$DEPS_DIR/quest-v4.1.0"
QULACS_SRC="$DEPS_DIR/qulacs-0.6.12"

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
    echo "  cargo build -p pecos-quest -p pecos-qulacs"
    exit 1
fi

echo "=== Native Benchmark: PECOS vs QuEST vs Qulacs ==="
echo ""

# ---------------------------------------------------------------------------
# Build QuEST via CMake (single-threaded CPU, no OpenMP/GPU/MPI)
# ---------------------------------------------------------------------------

echo "--- Building QuEST (CMake, Release, single-threaded) ---"
QUEST_BUILD="$BUILD_DIR/quest"
mkdir -p "$QUEST_BUILD"
cmake -S "$QUEST_SRC" -B "$QUEST_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-march=native" \
    -DCMAKE_CXX_FLAGS="-march=native" \
    -DENABLE_MULTITHREADING=OFF \
    -DENABLE_CUDA=OFF \
    -DENABLE_HIP=OFF \
    -DENABLE_DISTRIBUTION=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    2>&1 | tail -5
cmake --build "$QUEST_BUILD" -j "$(nproc)" 2>&1 | tail -3
echo "QuEST built."
echo ""

# ---------------------------------------------------------------------------
# Build Qulacs via CMake (no OpenMP)
# ---------------------------------------------------------------------------

echo "--- Building Qulacs (CMake, Release, single-threaded) ---"
QULACS_BUILD="$BUILD_DIR/qulacs"
mkdir -p "$QULACS_BUILD"

# Qulacs needs Boost headers; use the copy already downloaded by PECOS
BOOST_DIR="$DEPS_DIR/boost-1.83.0"
if [ ! -d "$BOOST_DIR" ]; then
    echo "ERROR: Boost not found at $BOOST_DIR"
    echo "Run: cargo build -p pecos-qulacs"
    exit 1
fi

cmake -S "$QULACS_SRC" -B "$QULACS_BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_FLAGS="-march=native" \
    -DCMAKE_CXX_FLAGS="-march=native -DEIGEN_NO_DEBUG" \
    -DBoost_INCLUDE_DIR="$BOOST_DIR" \
    -DUSE_OMP=OFF \
    -DUSE_GPU=OFF \
    -DUSE_MPI=OFF \
    -DUSE_TEST=OFF \
    -DUSE_PYTHON=OFF \
    2>&1 | tail -5
cmake --build "$QULACS_BUILD" -j "$(nproc)" --target csim_static cppsim_static 2>&1 | tail -3
echo "Qulacs built."
echo ""

# ---------------------------------------------------------------------------
# Locate built libraries
# ---------------------------------------------------------------------------

# QuEST: static library built by CMake
QUEST_LIB="$(find "$QUEST_BUILD" -name 'libQuEST.a' | head -1)"
if [ -z "$QUEST_LIB" ]; then
    echo "ERROR: Could not find libQuEST.a in $QUEST_BUILD"
    exit 1
fi
QUEST_LIB_DIR="$(dirname "$QUEST_LIB")"

# QuEST include paths: source headers + generated quest.h
QUEST_INC_GEN="$QUEST_BUILD/include"
QUEST_INC_SRC="$QUEST_SRC/quest/include"
QUEST_INC_ROOT="$QUEST_SRC"

# Qulacs: static libraries (csim + cppsim)
# Qulacs CMakeLists sets CMAKE_ARCHIVE_OUTPUT_DIRECTORY to ${PROJECT_BINARY_DIR}/../lib
QULACS_CSIM_LIB="$(find "$BUILD_DIR" -name 'libcsim_static.a' | head -1)"
QULACS_CPPSIM_LIB="$(find "$BUILD_DIR" -name 'libcppsim_static.a' | head -1)"
if [ -z "$QULACS_CSIM_LIB" ] || [ -z "$QULACS_CPPSIM_LIB" ]; then
    echo "ERROR: Could not find Qulacs static libraries in $BUILD_DIR"
    exit 1
fi

# Qulacs include: source tree + Eigen (downloaded by CMake ExternalProject)
QULACS_INC="$QULACS_SRC/src"
QULACS_EIGEN_INC="$QULACS_SRC/include"
# If Eigen wasn't installed by CMake into the source tree, fall back to PECOS's copy
if [ ! -d "$QULACS_EIGEN_INC/Eigen" ]; then
    QULACS_EIGEN_INC="$DEPS_DIR/eigen-3.4.0"
fi

# ---------------------------------------------------------------------------
# Compile standalone benchmark programs
# ---------------------------------------------------------------------------

echo "--- Compiling bench_quest ---"
cc -O3 -march=native -std=c11 \
    -I"$QUEST_INC_GEN" -I"$QUEST_INC_SRC" -I"$QUEST_INC_ROOT" \
    "$SCRIPT_DIR/bench_quest.c" \
    -L"$QUEST_LIB_DIR" -lQuEST \
    -lstdc++ -lm \
    -o "$BUILD_DIR/bench_quest"
echo "Compiled."

echo "--- Compiling bench_qulacs ---"
c++ -O3 -march=native -std=c++14 \
    -I"$QULACS_INC" -I"$QULACS_EIGEN_INC" -I"$BOOST_DIR" \
    -DEIGEN_NO_DEBUG \
    "$SCRIPT_DIR/bench_qulacs.cpp" \
    "$QULACS_CPPSIM_LIB" "$QULACS_CSIM_LIB" \
    -lm \
    -o "$BUILD_DIR/bench_qulacs"
echo "Compiled."
echo ""

# ---------------------------------------------------------------------------
# Run standalone benchmarks
# ---------------------------------------------------------------------------

echo "--- Running QuEST benchmark ---"
"$BUILD_DIR/bench_quest" | tee "$BUILD_DIR/quest_results.txt"
echo ""

echo "--- Running Qulacs benchmark ---"
"$BUILD_DIR/bench_qulacs" | tee "$BUILD_DIR/qulacs_results.txt"
echo ""

# ---------------------------------------------------------------------------
# Run PECOS Rust criterion benchmarks
# ---------------------------------------------------------------------------

echo "--- Running PECOS criterion benchmarks (--quick mode) ---"
cd "$REPO_ROOT"

# Capture criterion output; --quick runs minimal iterations for fast comparison
CRITERION_OUT="$BUILD_DIR/criterion_output.txt"
cargo bench -p benchmarks --profile native --bench benchmarks \
    --features quest,qulacs -- "Native" --quick 2>&1 | tee "$CRITERION_OUT"
echo ""

# ---------------------------------------------------------------------------
# Parse criterion results and print comparison table
# ---------------------------------------------------------------------------

echo "============================================================"
echo "                    COMPARISON SUMMARY"
echo "============================================================"
echo ""
echo "QuEST standalone results:"
cat "$BUILD_DIR/quest_results.txt"
echo ""
echo "Qulacs standalone results:"
cat "$BUILD_DIR/qulacs_results.txt"
echo ""
echo "PECOS criterion results (see above for full output):"
# Extract timing lines from criterion output
grep -E "time:.*\[" "$CRITERION_OUT" 2>/dev/null || echo "(parse criterion output above for timings)"
echo ""
echo "============================================================"
echo "Done. Full outputs saved in: $BUILD_DIR/"
