#!/usr/bin/env bash
set -euo pipefail

LLVM_VERSION="${LLVM_VERSION:-21.1}"
LLVM_RELEASE_VERSION="${LLVM_RELEASE_VERSION:-21.1.8}"
INSTALL_DIR="${LLVM_INSTALL_DIR:-$HOME/.pecos/deps/llvm-$LLVM_VERSION}"

case "$(uname -m)" in
    x86_64|amd64)
        ASSET="LLVM-${LLVM_RELEASE_VERSION}-Linux-X64.tar.xz"
        SHA256="b3b7f2801d15d50736acea3c73982994d025b01c2f035b91ae3b49d1b575732b"
        ;;
    aarch64|arm64)
        ASSET="LLVM-${LLVM_RELEASE_VERSION}-Linux-ARM64.tar.xz"
        SHA256="65ce0b329514e5643407db2d02a5bd34bf33d159055dafa82825c8385bd01993"
        ;;
    *)
        echo "Unsupported Linux architecture for LLVM ${LLVM_RELEASE_VERSION}: $(uname -m)" >&2
        exit 1
        ;;
esac

llvm_is_shared() {
    local llvm_config="$1"

    [ -x "$llvm_config" ] || return 1
    "$llvm_config" --version | grep -q '^21\.1' || return 1
    [ "$("$llvm_config" --shared-mode)" = "shared" ] || return 1
    "$llvm_config" --libnames --link-shared | grep -q 'libLLVM-21\.so'
}

if llvm_is_shared "$INSTALL_DIR/bin/llvm-config"; then
    echo "Shared LLVM $("$INSTALL_DIR/bin/llvm-config" --version) already installed at $INSTALL_DIR"
    exit 0
elif [ -e "$INSTALL_DIR" ]; then
    echo "Removing invalid or non-shared LLVM install at $INSTALL_DIR"
    rm -rf "$INSTALL_DIR"
fi

URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_RELEASE_VERSION}/${ASSET}"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

ARCHIVE="$TMP_DIR/$ASSET"
EXTRACT_DIR="$TMP_DIR/extract"

echo "Downloading official LLVM ${LLVM_RELEASE_VERSION} package: $ASSET"
if command -v curl >/dev/null 2>&1; then
    curl --fail --location --retry 5 --retry-delay 5 --output "$ARCHIVE" "$URL"
else
    python3 - "$URL" "$ARCHIVE" <<'PY'
import sys
import urllib.request

url, dest = sys.argv[1], sys.argv[2]
with urllib.request.urlopen(url) as response, open(dest, "wb") as out:
    out.write(response.read())
PY
fi

echo "$SHA256  $ARCHIVE" | sha256sum -c -

mkdir -p "$EXTRACT_DIR"
tar -xJf "$ARCHIVE" -C "$EXTRACT_DIR" --strip-components=1

rm -rf "$INSTALL_DIR"
mkdir -p "$(dirname "$INSTALL_DIR")"
mv "$EXTRACT_DIR" "$INSTALL_DIR"

"$INSTALL_DIR/bin/llvm-config" --version
"$INSTALL_DIR/bin/llvm-config" --shared-mode
if ! llvm_is_shared "$INSTALL_DIR/bin/llvm-config"; then
    echo "The official LLVM ${LLVM_RELEASE_VERSION} Linux archive does not provide libLLVM-21.so." >&2
    echo "PECOS CI needs a shared LLVM build; use scripts/ci/install-llvm-21-conda-linux.sh instead." >&2
    exit 1
fi
echo "Installed LLVM ${LLVM_RELEASE_VERSION} to $INSTALL_DIR"
