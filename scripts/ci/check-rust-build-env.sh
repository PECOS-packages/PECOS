#!/usr/bin/env bash
# Run after `just ci-env`, with its exported environment intact. A clean-shell
# probe misses library overrides that can replace rustc's bundled LLVM.
set -euo pipefail

# macOS can strip DYLD_* when starting a system shell. Apply PECOS's actual
# exports inside this process, as the Python build does for its child process.
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
cargo +stable check --offline --manifest-path "$probe_dir/Cargo.toml" \
    --target-dir "$probe_dir/target"
