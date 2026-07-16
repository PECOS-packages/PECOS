#!/usr/bin/env bash
set -euo pipefail

ARCHIVE="${1:?usage: install-llvm-21-macos.sh ARCHIVE}"
LLVM_VERSION="${LLVM_VERSION:-21.1}"
INSTALL_DIR="${LLVM_INSTALL_DIR:-$HOME/.pecos/deps/llvm-$LLVM_VERSION}"
LLVM_CONFIG="$INSTALL_DIR/bin/llvm-config"

if [ ! -f "$ARCHIVE" ]; then
    echo "macOS LLVM archive not found: $ARCHIVE" >&2
    exit 1
fi

rm -rf "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
tar -xzf "$ARCHIVE" -C "$INSTALL_DIR"

"$LLVM_CONFIG" --version
"$LLVM_CONFIG" --shared-mode
"$LLVM_CONFIG" --libnames --link-shared

test "$("$LLVM_CONFIG" --shared-mode)" = shared
"$LLVM_CONFIG" --libnames --link-shared | grep -q 'libLLVM\.dylib'
find "$INSTALL_DIR/lib" -maxdepth 1 -name 'libclang*.dylib' | grep -q .

echo "Installed macOS-compatible LLVM at $INSTALL_DIR"
