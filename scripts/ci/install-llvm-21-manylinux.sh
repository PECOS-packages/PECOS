#!/usr/bin/env bash
set -euo pipefail

ARCHIVE="${1:?usage: install-llvm-21-manylinux.sh ARCHIVE}"
LLVM_VERSION="${LLVM_VERSION:-21.1}"
INSTALL_DIR="${LLVM_INSTALL_DIR:-$HOME/.pecos/deps/llvm-$LLVM_VERSION}"
LLVM_CONFIG="$INSTALL_DIR/bin/llvm-config"

if [ ! -f "$ARCHIVE" ]; then
    echo "manylinux LLVM archive not found: $ARCHIVE" >&2
    exit 1
fi

rm -rf "$INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
tar -xJf "$ARCHIVE" -C "$INSTALL_DIR"

"$LLVM_CONFIG" --version
"$LLVM_CONFIG" --shared-mode
"$LLVM_CONFIG" --libnames --link-shared

test "$("$LLVM_CONFIG" --shared-mode)" = shared
"$LLVM_CONFIG" --libnames --link-shared | grep -q 'libLLVM-21\.so'
find "$INSTALL_DIR/lib" -maxdepth 1 \
    \( -name 'libclang.so' -o -name 'libclang.so.*' \) \
    | grep -q .

echo "Installed manylinux-compatible LLVM at $INSTALL_DIR"
