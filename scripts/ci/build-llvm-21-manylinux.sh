#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ARCHIVE="${1:?usage: build-llvm-21-manylinux.sh OUTPUT_ARCHIVE}"
LLVM_RELEASE_VERSION="${LLVM_RELEASE_VERSION:-21.1.8}"
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
        echo "Unsupported manylinux LLVM build architecture: $(uname -m)" >&2
        exit 1
        ;;
esac

dnf install -y cmake curl gcc gcc-c++ make python3 tar xz

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT

SOURCE_ARCHIVE="$WORK_DIR/$LLVM_SOURCE_ASSET"
SOURCE_DIR="$WORK_DIR/source"
BUILD_DIR="$WORK_DIR/build"
INSTALL_DIR="$WORK_DIR/install"

curl --fail --location --retry 5 --retry-delay 5 \
    --output "$SOURCE_ARCHIVE" \
    "$LLVM_SOURCE_URL"
echo "$LLVM_SOURCE_SHA256  $SOURCE_ARCHIVE" | sha256sum -c -

mkdir -p "$SOURCE_DIR"
tar -xJf "$SOURCE_ARCHIVE" -C "$SOURCE_DIR" --strip-components=1

export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
"$CC" --version
"$CXX" --version

cmake -S "$SOURCE_DIR/llvm" -B "$BUILD_DIR" -G "Unix Makefiles" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
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
    install-llvm-libraries \
    install-llvm-headers \
    install-clang-headers \
    install-libclang \
    --parallel "$(nproc)"

"$INSTALL_DIR/bin/llvm-config" --version
"$INSTALL_DIR/bin/llvm-config" --shared-mode
"$INSTALL_DIR/bin/llvm-config" --libnames --link-shared
STATIC_LIB_NAMES="$("$INSTALL_DIR/bin/llvm-config" --libnames --link-static)"
read -r -a STATIC_LIBRARIES <<< "$STATIC_LIB_NAMES"
if [ "${#STATIC_LIBRARIES[@]}" -eq 0 ]; then
    echo "llvm-config did not report any static LLVM archives" >&2
    exit 1
fi
LLVM_LIB_DIR="$("$INSTALL_DIR/bin/llvm-config" --libdir)"
for archive in "${STATIC_LIBRARIES[@]}"; do
    if [ ! -f "$LLVM_LIB_DIR/$archive" ]; then
        echo "Missing static LLVM archive reported by llvm-config: $LLVM_LIB_DIR/$archive" >&2
        exit 1
    fi
done
echo "Validated ${#STATIC_LIBRARIES[@]} static LLVM archives in $LLVM_LIB_DIR"

test "$("$INSTALL_DIR/bin/llvm-config" --shared-mode)" = shared
"$INSTALL_DIR/bin/llvm-config" --libnames --link-shared | grep -q 'libLLVM-21\.so'
find "$INSTALL_DIR/lib" -maxdepth 1 \
    \( -name 'libclang.so' -o -name 'libclang.so.*' \) \
    | grep -q .

mkdir -p "$(dirname "$OUTPUT_ARCHIVE")"
tar -cJf "$OUTPUT_ARCHIVE" -C "$INSTALL_DIR" .
echo "Built manylinux LLVM ${LLVM_RELEASE_VERSION}: $OUTPUT_ARCHIVE"
