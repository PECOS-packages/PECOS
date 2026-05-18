# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# BinaryBuilder.jl script for PECOS_julia
# This creates pre-compiled binaries for all platforms
#
# To use:
# 1. Install BinaryBuilder: using Pkg; Pkg.add("BinaryBuilder")
# 2. Run: julia build_tarballs.jl --debug --verbose
# 3. Deploy: julia build_tarballs.jl --deploy

using BinaryBuilder, Pkg

name = "PECOS_julia"
version = v"0.1.0-dev0"

# Collection of sources required to build PECOS
sources = [
    # For release, use a specific tag/commit
    # Use environment variable or fallback to current branch/commit
    GitSource(
        "https://github.com/PECOS-packages/PECOS.git",
        get(ENV, "PECOS_BUILD_COMMIT", "dev"),  # Will be replaced by workflow
    ),
]

# Bash recipe for building across all platforms
script = raw"""
cd $WORKSPACE/srcdir/PECOS

# Install Rust
if [[ "${target}" == *-mingw* ]]; then
    # Windows: Download pre-built rustc
    curl -L https://win.rustup.rs/x86_64 -o rustup-init.exe
    ./rustup-init.exe -y --profile minimal --default-toolchain stable
    export PATH="$HOME/.cargo/bin:$PATH"
else
    # Unix-like: use rustup-init with its SHA-256 sidecar instead of curl-pipe-shell.
    bash scripts/ci/ensure-rust.sh stable minimal
    export PATH="$HOME/.cargo/bin:$PATH"
fi

# Verify Rust installation
rustc --version
cargo --version

# Build the library
cd julia/pecos-julia-ffi

# BinaryBuilder sets CARGO_BUILD_TARGET for cross-compilation
if [[ -n "${CARGO_BUILD_TARGET}" ]]; then
    echo "Cross-compiling for: ${CARGO_BUILD_TARGET}"
fi

# BinaryBuilder handles --target automatically with CARGO_BUILD_TARGET
cargo build --locked --release

# Find and install the built library
cd target/release
if [[ "${target}" == *-mingw* ]]; then
    install -Dvm 755 pecos_julia.dll "${libdir}/pecos_julia.dll"
elif [[ "${target}" == *-apple-* ]]; then
    install -Dvm 755 libpecos_julia.dylib "${libdir}/libpecos_julia.dylib"
else
    install -Dvm 755 libpecos_julia.so "${libdir}/libpecos_julia.so"
fi
"""

# These are the platforms we will build for
platforms = [
    # Linux
    Platform("x86_64", "linux"; libc = "glibc"),
    Platform("aarch64", "linux"; libc = "glibc"),

    # macOS
    Platform("x86_64", "macos"),
    Platform("aarch64", "macos"), # Apple Silicon

    # Windows
    Platform("x86_64", "windows"),
]

# The products that we will ensure are always built
products = [LibraryProduct("libpecos_julia", :libpecos_julia)]

# Dependencies that must be installed before this package can be built
dependencies = Dependency[]

# Build the tarballs
build_tarballs(
    ARGS,
    name,
    version,
    sources,
    script,
    platforms,
    products,
    dependencies;
    compilers = [:rust, :c],  # Need Rust compiler support
    julia_compat = "1.10",
    preferred_gcc_version = v"8",
)
