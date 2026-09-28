#!/usr/bin/env bash
# Run after `just ci-env`, with its exported environment intact. A clean-shell
# probe misses library overrides that can replace rustc's bundled LLVM.
set -euo pipefail

# macOS can strip DYLD_* when starting a system shell. Apply PECOS's actual
# exports inside this process, as the Python build does for its child process.
# Match maturin's direct compiler invocation: the rustup shim otherwise repairs
# library paths and masks rust-objcopy failures. Select before collecting env so
# its Rust library directory belongs to the toolchain we actually test.
export RUSTC
RUSTC="$(rustup which --toolchain stable rustc)"
cargo_bin="$(rustup which --toolchain stable cargo)"
build_env="$(cargo run --locked -p pecos-cli -- env)"
eval "$build_env"

probe_dir="$(mktemp -d)"
trap 'rm -rf "$probe_dir"' EXIT

rustc +stable -vV
printf 'DYLD_LIBRARY_PATH=%s\n' "${DYLD_LIBRARY_PATH:-}"
printf 'DYLD_FALLBACK_LIBRARY_PATH=%s\n' "${DYLD_FALLBACK_LIBRARY_PATH:-}"
cargo +stable init --lib --name pecos_build_env_probe "$probe_dir"

# Match the additional flag used by `pecos python build` on macOS. Cargo
# constructs the compiler target probe; retain stderr and propagate failure.
export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-Wl,-rpath,/usr/lib"
"$cargo_bin" check --offline --manifest-path "$probe_dir/Cargo.toml" \
    --target-dir "$probe_dir/target"

# `cargo check` never strips binaries. Exercise a build script and an executable
# with release stripping enabled, and fail on stripping warnings (rustc otherwise
# returns success even when rust-objcopy crashes).
printf 'fn main() {}\n' > "$probe_dir/build.rs"
printf 'fn main() {}\n' > "$probe_dir/src/main.rs"
CARGO_PROFILE_RELEASE_STRIP=true "$cargo_bin" build --release --offline \
    --manifest-path "$probe_dir/Cargo.toml" --target-dir "$probe_dir/target" \
    2>&1 | tee "$probe_dir/release.log"
if grep -q 'stripping debug info.*failed' "$probe_dir/release.log"; then
    echo "Rust release stripping failed in the PECOS build environment" >&2
    exit 1
fi
