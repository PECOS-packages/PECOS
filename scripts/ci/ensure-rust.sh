#!/usr/bin/env bash
set -euo pipefail

toolchain="${1:-stable}"
profile="${2:-minimal}"

# Every network step below talks to static.rust-lang.org, which resets
# connections from GitHub runners now and then (two PR gate failures within
# an hour on 2026-09-03, both `Connection reset by peer` on the channel
# manifest). Retry each one a few times with a growing pause before giving up.
retry() {
    local attempts=3 attempt
    for ((attempt = 1; attempt <= attempts; attempt++)); do
        if "$@"; then
            return 0
        fi
        if ((attempt < attempts)); then
            echo "ensure-rust: attempt $attempt/$attempts failed: $*; retrying in $((attempt * 15))s" >&2
            sleep $((attempt * 15))
        fi
    done
    echo "ensure-rust: giving up after $attempts attempts: $*" >&2
    return 1
}

if command -v rustup >/dev/null 2>&1; then
    retry rustup toolchain install "$toolchain" --profile "$profile"
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
    retry curl --proto '=https' --tlsv1.2 -fsSLo "$tmp_dir/rustup-init" "$base_url/rustup-init"
    retry curl --proto '=https' --tlsv1.2 -fsSLo "$tmp_dir/rustup-init.sha256" "$base_url/rustup-init.sha256"

    if command -v sha256sum >/dev/null 2>&1; then
        (cd "$tmp_dir" && sha256sum -c rustup-init.sha256)
    else
        (cd "$tmp_dir" && shasum -a 256 -c rustup-init.sha256)
    fi

    chmod +x "$tmp_dir/rustup-init"
    retry "$tmp_dir/rustup-init" -y --profile "$profile" --default-toolchain "$toolchain" --no-modify-path
fi

if [[ -n "${GITHUB_PATH:-}" ]]; then
    echo "$HOME/.cargo/bin" >>"$GITHUB_PATH"
fi
