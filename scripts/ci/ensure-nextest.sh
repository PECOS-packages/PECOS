#!/usr/bin/env bash
# Install the pinned cargo-nextest release into the cargo bin directory so
# `pecos rust test --nextest` (the PR gate's `just rstest debug nextest`) can
# run the workspace test binaries in parallel instead of one after another.
# Prebuilt release, verified against the sha256 nextest publishes next to it;
# a `cargo install` would spend minutes compiling on every run.
#
# Usage: scripts/ci/ensure-nextest.sh
set -euo pipefail

version="0.9.143"
target="x86_64-unknown-linux-gnu"
sha256="66786b9abe23920d022a182d1416b1bbc8130dd4872a9553d76985a1708dcd1e"

case "$(uname -s)-$(uname -m)" in
    Linux-x86_64) ;;
    *)
        echo "ensure-nextest: only ${target} is pinned; got $(uname -s)-$(uname -m)" >&2
        exit 1
        ;;
esac

dest="${CARGO_HOME:-$HOME/.cargo}/bin"
if [[ -x "$dest/cargo-nextest" ]] && "$dest/cargo-nextest" --version | grep -q "^cargo-nextest ${version} "; then
    echo "cargo-nextest ${version} already installed at $dest"
else
    tmp_dir="$(mktemp -d)"
    trap 'rm -rf "$tmp_dir"' EXIT
    archive="cargo-nextest-${version}-${target}.tar.gz"
    curl --proto '=https' --tlsv1.2 -fsSL --retry 3 --retry-all-errors \
        -o "$tmp_dir/$archive" \
        "https://github.com/nextest-rs/nextest/releases/download/cargo-nextest-${version}/${archive}"
    echo "${sha256}  ${tmp_dir}/${archive}" | sha256sum -c -
    mkdir -p "$dest"
    tar -xzf "$tmp_dir/$archive" -C "$dest" cargo-nextest
fi
"$dest/cargo-nextest" --version

# `cargo nextest` is resolved through PATH by cargo, so expose the install
# directory to later workflow steps (a custom CARGO_HOME is not the directory
# ensure-rust.sh adds).
if [[ -n "${GITHUB_PATH:-}" ]]; then
    echo "$dest" >>"$GITHUB_PATH"
fi
