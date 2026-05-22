#!/usr/bin/env bash
set -euo pipefail

toolchain="${1:-stable}"
profile="${2:-minimal}"

if command -v rustup >/dev/null 2>&1; then
    rustup toolchain install "$toolchain" --profile "$profile"
else
    case "$(uname -s)-$(uname -m)" in
        Linux-x86_64)
            target="x86_64-unknown-linux-gnu"
            ;;
        Linux-aarch64 | Linux-arm64)
            target="aarch64-unknown-linux-gnu"
            ;;
        Darwin-x86_64)
            target="x86_64-apple-darwin"
            ;;
        Darwin-arm64 | Darwin-aarch64)
            target="aarch64-apple-darwin"
            ;;
        *)
            echo "Unsupported platform for rustup bootstrap: $(uname -s)-$(uname -m)" >&2
            exit 1
            ;;
    esac

    tmp_dir="$(mktemp -d)"
    trap 'rm -rf "$tmp_dir"' EXIT

    base_url="https://static.rust-lang.org/rustup/dist/${target}"
    curl --proto '=https' --tlsv1.2 -fsSLo "$tmp_dir/rustup-init" "$base_url/rustup-init"
    curl --proto '=https' --tlsv1.2 -fsSLo "$tmp_dir/rustup-init.sha256" "$base_url/rustup-init.sha256"

    if command -v sha256sum >/dev/null 2>&1; then
        (cd "$tmp_dir" && sha256sum -c rustup-init.sha256)
    else
        (cd "$tmp_dir" && shasum -a 256 -c rustup-init.sha256)
    fi

    chmod +x "$tmp_dir/rustup-init"
    "$tmp_dir/rustup-init" -y --profile "$profile" --default-toolchain "$toolchain" --no-modify-path
fi

if [[ -n "${GITHUB_PATH:-}" ]]; then
    echo "$HOME/.cargo/bin" >>"$GITHUB_PATH"
fi
