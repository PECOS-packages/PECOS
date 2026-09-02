#!/usr/bin/env bash
set -euo pipefail

workspace="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ $# -gt 1 ]]; then
    echo "usage: $0 [flattened-model.dem]" >&2
    exit 2
fi

if [[ $# -eq 1 ]]; then
    export FRONTIER_DEM_PATH="$(realpath "$1")"
else
    unset FRONTIER_DEM_PATH || true
fi
export RUSTFLAGS="${RUSTFLAGS:+${RUSTFLAGS} }--cfg getrandom_backend=\"unsupported\""

cargo build \
    --manifest-path "$workspace/Cargo.toml" \
    --release \
    --target wasm32-unknown-unknown \
    -p pecos-frontier-wasm

mkdir -p "$workspace/dist"
cp "$workspace/target/wasm32-unknown-unknown/release/pecos_frontier_wasm.wasm" \
    "$workspace/dist/pecos_frontier_wasm.wasm"
ls -lh "$workspace/dist/pecos_frontier_wasm.wasm"
