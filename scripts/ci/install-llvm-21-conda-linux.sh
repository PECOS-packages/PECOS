#!/usr/bin/env bash
set -euo pipefail

LLVM_VERSION="${LLVM_VERSION:-21.1}"
LLVM_RELEASE_VERSION="${LLVM_RELEASE_VERSION:-21.1.8}"
INSTALL_DIR="${LLVM_INSTALL_DIR:-$HOME/.pecos/deps/llvm-$LLVM_VERSION}"
MAMBA_VERSION="${MAMBA_VERSION:-latest}"
MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.cache/pecos-micromamba}"

case "$(uname -m)" in
    x86_64|amd64)
        MAMBA_PLATFORM="linux-64"
        ;;
    aarch64|arm64)
        MAMBA_PLATFORM="linux-aarch64"
        ;;
    *)
        echo "Unsupported Linux architecture for conda-forge LLVM ${LLVM_RELEASE_VERSION}: $(uname -m)" >&2
        exit 1
        ;;
esac

llvm_is_valid() {
    local llvm_config="$1"
    local llvm_dir

    llvm_dir="$(dirname "$(dirname "$llvm_config")")"

    [ -x "$llvm_config" ] || return 1
    "$llvm_config" --version | grep -q '^21\.1' || return 1
    [ "$("$llvm_config" --shared-mode)" = "shared" ] || return 1
    "$llvm_config" --libnames --link-shared | grep -q 'libLLVM-21\.so'
    find "$llvm_dir/lib" -maxdepth 1 \
        \( -name 'libclang.so' -o -name 'libclang-*.so' -o -name 'libclang.so.*' -o -name 'libclang-*.so.*' \) \
        | grep -q .
}

LLVM_CONFIG="$INSTALL_DIR/bin/llvm-config"
if llvm_is_valid "$LLVM_CONFIG"; then
    echo "Shared LLVM $("$LLVM_CONFIG" --version) already installed at $INSTALL_DIR"
    exit 0
fi

if [ -e "$INSTALL_DIR" ]; then
    echo "Removing invalid or non-shared LLVM install at $INSTALL_DIR"
    rm -rf "$INSTALL_DIR"
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

if command -v micromamba >/dev/null 2>&1; then
    MAMBA_BIN="$(command -v micromamba)"
else
    MAMBA_URL="https://micro.mamba.pm/api/micromamba/${MAMBA_PLATFORM}/${MAMBA_VERSION}"
    MAMBA_ARCHIVE="$TMP_DIR/micromamba.tar.bz2"

    echo "Downloading micromamba for ${MAMBA_PLATFORM}"
    curl --fail --location --retry 5 --retry-delay 5 --output "$MAMBA_ARCHIVE" "$MAMBA_URL"
    tar -xjf "$MAMBA_ARCHIVE" -C "$TMP_DIR" bin/micromamba
    MAMBA_BIN="$TMP_DIR/bin/micromamba"
fi

echo "Installing conda-forge LLVM ${LLVM_RELEASE_VERSION} to $INSTALL_DIR"
MAMBA_ROOT_PREFIX="$MAMBA_ROOT_PREFIX" "$MAMBA_BIN" create \
    -y \
    -p "$INSTALL_DIR" \
    --override-channels \
    -c conda-forge \
    "llvmdev=${LLVM_RELEASE_VERSION}" \
    "libclang=${LLVM_RELEASE_VERSION}"

if ! llvm_is_valid "$LLVM_CONFIG"; then
    echo "conda-forge LLVM install did not provide shared LLVM and libclang ${LLVM_RELEASE_VERSION}" >&2
    "$LLVM_CONFIG" --version >&2 || true
    "$LLVM_CONFIG" --shared-mode >&2 || true
    "$LLVM_CONFIG" --libnames --link-shared >&2 || true
    find "$INSTALL_DIR/lib" -maxdepth 1 -name 'libclang*' -print >&2 || true
    exit 1
fi

"$LLVM_CONFIG" --version
"$LLVM_CONFIG" --shared-mode
"$LLVM_CONFIG" --libnames --link-shared
find "$INSTALL_DIR/lib" -maxdepth 1 \
    \( -name 'libclang.so' -o -name 'libclang-*.so' -o -name 'libclang.so.*' -o -name 'libclang-*.so.*' \) \
    -print
echo "Installed shared LLVM and libclang ${LLVM_RELEASE_VERSION} to $INSTALL_DIR"
