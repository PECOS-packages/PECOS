#!/usr/bin/env bash
# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
# except in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

# Download QuEST + Qulacs + Eigen + Boost source archives into ~/.pecos/deps/ for
# standalone native benchmarking. These are comparison-only vendored sources --
# PECOS does not link against them at runtime.

set -euo pipefail

DEPS_DIR="$HOME/.pecos/deps"
mkdir -p "$DEPS_DIR"

# name | version-dir | url | sha256
DEPS=(
    "quest|quest-v4.2.0|https://github.com/QuEST-Kit/QuEST/archive/refs/tags/v4.2.0.tar.gz|2c812a7ec4d727e0947ffd0daf05452963c3f1c10e428c8bc30c35164921fcba"
    "qulacs|qulacs-0.6.13|https://github.com/qulacs/qulacs/archive/v0.6.13.tar.gz|9ef25a988b9f483b97ea9501554a1ce5ee23ffaf89e7ca89969f0d03fcf94af0"
    "eigen|eigen-3.4.0|https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz|8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72"
    "boost|boost-1.83.0|https://archives.boost.io/release/1.83.0/source/boost_1_83_0.tar.bz2|6478edfe2f3305127cffe8caf73ea0176c53769f4bf1585be237eb30798c3b8e"
)

AUTO_YES="${AUTO_YES:-0}"
if [ "${1:-}" = "-y" ] || [ "${1:-}" = "--yes" ]; then
    AUTO_YES=1
fi

prompt_yes() {
    if [ "$AUTO_YES" = "1" ]; then
        return 0
    fi
    local msg="$1"
    read -r -p "$msg [Y/n] " reply
    case "$reply" in
        ""|[Yy]*) return 0 ;;
        *) return 1 ;;
    esac
}

fetch_one() {
    local name="$1" dir="$2" url="$3" sha256="$4"
    local target="$DEPS_DIR/$dir"

    if [ -d "$target" ]; then
        echo "[skip] $name already present at $target"
        return
    fi

    if ! prompt_yes "Download $name ($url)?"; then
        echo "[skip] $name (user declined)"
        return
    fi

    local tmpdir
    tmpdir="$(mktemp -d)"
    trap "rm -rf '$tmpdir'" RETURN

    local archive="$tmpdir/download"
    echo "[fetch] $url"
    curl -fL --retry 3 --retry-delay 2 -o "$archive" "$url"

    local got
    got="$(sha256sum "$archive" | awk '{print $1}')"
    if [ "$got" != "$sha256" ]; then
        echo "ERROR: sha256 mismatch for $name"
        echo "  expected: $sha256"
        echo "  got:      $got"
        return 1
    fi

    echo "[extract] -> $target"
    mkdir -p "$tmpdir/extract"
    case "$url" in
        *.tar.bz2) tar -xjf "$archive" -C "$tmpdir/extract" ;;
        *.tar.gz|*.tgz) tar -xzf "$archive" -C "$tmpdir/extract" ;;
        *) echo "ERROR: unknown archive type for $url"; return 1 ;;
    esac

    # Top-level directory inside the archive
    local inner
    inner="$(ls "$tmpdir/extract")"
    mv "$tmpdir/extract/$inner" "$target"
    echo "[done] $name at $target"
}

echo "Native bench dependencies will be fetched into $DEPS_DIR"
echo ""

for entry in "${DEPS[@]}"; do
    IFS='|' read -r name dir url sha256 <<<"$entry"
    fetch_one "$name" "$dir" "$url" "$sha256"
done

echo ""
echo "Done."
