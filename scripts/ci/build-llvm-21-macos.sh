#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ARCHIVE="${1:?usage: build-llvm-21-macos.sh OUTPUT_ARCHIVE}"
LLVM_RELEASE_VERSION="${LLVM_RELEASE_VERSION:-21.1.8}"
MACOS_DEPLOYMENT_TARGET="${MACOS_WHEEL_DEPLOYMENT_TARGET:-13.0}"
LLVM_SOURCE_SHA256="4633a23617fa31a3ea51242586ea7fb1da7140e426bd62fc164261fe036aa142"
LLVM_SOURCE_ASSET="llvm-project-${LLVM_RELEASE_VERSION}.src.tar.xz"
LLVM_SOURCE_URL="https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_RELEASE_VERSION}/${LLVM_SOURCE_ASSET}"

case "$(uname -m)" in
    x86_64|amd64)
        LLVM_TARGETS="X86"
        ;;
    aarch64|arm64)
        LLVM_TARGETS="AArch64"
        ;;
    *)
        echo "Unsupported macOS LLVM build architecture: $(uname -m)" >&2
        exit 1
        ;;
esac

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

SOURCE_ARCHIVE="$WORK_DIR/$LLVM_SOURCE_ASSET"
SOURCE_DIR="$WORK_DIR/source"
BUILD_DIR="$WORK_DIR/build"
INSTALL_DIR="$WORK_DIR/install"

curl --fail --location --retry 5 --retry-delay 5 \
    --output "$SOURCE_ARCHIVE" \
    "$LLVM_SOURCE_URL"
echo "$LLVM_SOURCE_SHA256  $SOURCE_ARCHIVE" | shasum -a 256 -c -

mkdir -p "$SOURCE_DIR"
tar -xJf "$SOURCE_ARCHIVE" -C "$SOURCE_DIR" --strip-components=1

export CC="$(xcrun --find clang)"
export CXX="$(xcrun --find clang++)"
export MACOSX_DEPLOYMENT_TARGET="$MACOS_DEPLOYMENT_TARGET"
"$CC" --version
"$CXX" --version

cmake -S "$SOURCE_DIR/llvm" -B "$BUILD_DIR" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$MACOS_DEPLOYMENT_TARGET" \
    -DLLVM_ENABLE_PROJECTS=clang \
    -DLLVM_TARGETS_TO_BUILD="$LLVM_TARGETS" \
    -DLLVM_BUILD_LLVM_DYLIB=ON \
    -DLLVM_LINK_LLVM_DYLIB=ON \
    -DLLVM_ENABLE_ASSERTIONS=OFF \
    -DLLVM_ENABLE_BINDINGS=OFF \
    -DLLVM_ENABLE_CURL=OFF \
    -DLLVM_ENABLE_LIBEDIT=OFF \
    -DLLVM_ENABLE_LIBXML2=OFF \
    -DLLVM_ENABLE_TERMINFO=OFF \
    -DLLVM_ENABLE_ZLIB=OFF \
    -DLLVM_ENABLE_ZSTD=OFF \
    -DLLVM_INCLUDE_BENCHMARKS=OFF \
    -DLLVM_INCLUDE_EXAMPLES=OFF \
    -DLLVM_INCLUDE_TESTS=OFF \
    -DCLANG_INCLUDE_TESTS=OFF

cmake --build "$BUILD_DIR" --target \
    install-llvm-config \
    install-LLVM \
    install-llvm-headers \
    install-clang-headers \
    install-libclang \
    --parallel "$(sysctl -n hw.ncpu)"

"$INSTALL_DIR/bin/llvm-config" --version
"$INSTALL_DIR/bin/llvm-config" --shared-mode
"$INSTALL_DIR/bin/llvm-config" --libnames --link-shared

test "$("$INSTALL_DIR/bin/llvm-config" --shared-mode)" = shared
"$INSTALL_DIR/bin/llvm-config" --libnames --link-shared \
    | grep -Eq '(^|[[:space:]])libLLVM(-[0-9]+)?\.dylib($|[[:space:]])'
find "$INSTALL_DIR/lib" -maxdepth 1 -name 'libclang*.dylib' | grep -q .

mkdir -p "$(dirname "$OUTPUT_ARCHIVE")"
tar -czf "$OUTPUT_ARCHIVE" -C "$INSTALL_DIR" .
echo "Built macOS ${MACOS_DEPLOYMENT_TARGET} LLVM ${LLVM_RELEASE_VERSION}: $OUTPUT_ARCHIVE"
