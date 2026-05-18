#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

failures=0

section() {
    printf '\n==> %s\n' "$1"
}

fail() {
    printf 'ERROR: %s\n' "$1" >&2
    failures=$((failures + 1))
}

warn() {
    printf 'WARN: %s\n' "$1" >&2
}

require_tool() {
    if ! command -v "$1" >/dev/null 2>&1; then
        fail "$1 is required for dependency integrity checks"
        return 1
    fi
}

normalize_path() {
    printf '%s\n' "${1//\\//}"
}

list_contains() {
    local needle="$1"
    shift
    local item
    for item in "$@"; do
        if [[ "$item" == "$needle" ]]; then
            return 0
        fi
    done
    return 1
}

collect_files() {
    rg --files "$@"
}

KNOWN_BAD_PACKAGE_RE='(mistralai|guardrails-ai|lightning|@tanstack/|@mistralai/|@uipath/|@opensearch-project/|@squawk/|@tallyui/|@beproduct/|@draftauth/|@dirigible-ai/|@ml-toolkit-ts/|@supersurkhet/|agentwork-cli|cmux-agent-mcp|cross-stitch|git-branch-selector|git-git-git|nextmove-mcp|safe-action|ts-dna|wot-api|finch-rust|sha-rust|finch_cli_rust|finch-rst|sha-rst)'
SHAI_HULUD_IOC_RE='(shai[-_ ]?hulud|router_init\.js|router_runtime\.js|setup\.mjs|setup_bun\.js|bun_environment\.js|transformers\.pyz|git-tanstack\.com|api\.masscan\.cloud|getsession\.org|filev2\.getsession|gh-token-monitor|IfYouRevokeThisTokenItWillWipeTheComputerOfTheOwner|shai-hulud-workflow)'

RG_EXCLUDES=(
    --hidden
    --glob '!.git/**'
    --glob '!target/**'
    --glob '!.venv/**'
    --glob '!.ruff_cache/**'
    --glob '!scripts/dependency-integrity-check.sh'
)

section "Tooling"
tooling_failures_before=$failures
require_tool rg || true
require_tool cargo || true
require_tool uv || true
if ((failures > tooling_failures_before)); then
    printf '\nDependency integrity check failed: required tooling is missing.\n' >&2
    exit 1
fi

section "Known affected package names"
lockfiles=()
while IFS= read -r file; do
    lockfiles+=("$file")
done < <(collect_files \
    -g 'Cargo.lock' \
    -g 'uv.lock' \
    -g 'pylock.toml' \
    -g 'requirements*.txt' \
    -g 'package-lock.json' \
    -g 'npm-shrinkwrap.json' \
    -g 'pnpm-lock.yaml' \
    -g 'yarn.lock' \
    -g 'bun.lock' \
    -g 'bun.lockb')

manifests=()
while IFS= read -r file; do
    manifests+=("$file")
done < <(collect_files \
    -g 'Cargo.toml' \
    -g 'pyproject.toml' \
    -g 'requirements*.txt' \
    -g 'package.json' \
    -g 'pnpm-workspace.yaml' \
    -g 'bunfig.toml')

package_files=("${lockfiles[@]}" "${manifests[@]}")
if ((${#package_files[@]} == 0)); then
    fail "no supported package manifests or lockfiles found"
else
    if rg -n -i "$KNOWN_BAD_PACKAGE_RE" "${package_files[@]}"; then
        fail "known Shai-Hulud-affected package name found in package files"
    else
        echo "No current Shai-Hulud package-name hits in package manifests or lockfiles."
    fi
fi

section "Repository IoCs"
if rg -n -i "${RG_EXCLUDES[@]}" "$SHAI_HULUD_IOC_RE" .; then
    fail "Shai-Hulud indicator found in repository contents"
else
    echo "No current Shai-Hulud payload or persistence indicators found."
fi

section "npm lock discipline"
npm_manifests=()
while IFS= read -r file; do
    npm_manifests+=("$file")
done < <(collect_files -g 'package.json')
npm_locks=()
while IFS= read -r file; do
    npm_locks+=("$file")
done < <(collect_files \
    -g 'package-lock.json' \
    -g 'npm-shrinkwrap.json' \
    -g 'pnpm-lock.yaml' \
    -g 'yarn.lock' \
    -g 'bun.lock' \
    -g 'bun.lockb')

if ((${#npm_manifests[@]} > 0 && ${#npm_locks[@]} == 0)); then
    printf '%s\n' "${npm_manifests[@]}"
    fail "npm package manifests exist without a committed lockfile"
elif ((${#npm_manifests[@]} == 0)); then
    echo "No npm package manifests found."
else
    echo "npm manifests have a committed lockfile."
fi

section "Cargo lock discipline"
cargo_failures_before=$failures
cargo_locks=()
while IFS= read -r file; do
    cargo_locks+=("$file")
done < <(collect_files -g 'Cargo.lock')

if ((${#cargo_locks[@]} == 0)); then
    fail "no Cargo.lock files found"
else
    for lockfile in "${cargo_locks[@]}"; do
        manifest="$(dirname "$lockfile")/Cargo.toml"
        if [[ ! -f "$manifest" ]]; then
            fail "$lockfile has no adjacent Cargo.toml"
            continue
        fi
        if ! cargo metadata --locked --manifest-path "$manifest" --format-version 1 >/dev/null; then
            fail "$lockfile is missing or not current with $manifest"
        fi
    done
    if ((failures == cargo_failures_before)); then
        echo "Cargo lockfiles are current."
    fi
fi

section "Cargo git dependency pins"
cargo_manifests=()
while IFS= read -r file; do
    cargo_manifests+=("$file")
done < <(collect_files -g 'Cargo.toml')

if ((${#cargo_manifests[@]} > 0)); then
    if rg -n '^[[:space:]]*(tag|branch)[[:space:]]*=' "${cargo_manifests[@]}"; then
        fail "Cargo git dependencies must use full immutable rev pins, not tag/branch"
    fi
    if rg -n '^[[:space:]]*rev[[:space:]]*=[[:space:]]*"[0-9a-f]{1,39}"' "${cargo_manifests[@]}"; then
        fail "Cargo git dependency rev pins must use full 40-character commit SHAs"
    fi
fi

if rg -n 'git\+.*[?&](tag|branch)=' Cargo.lock >/dev/null 2>&1; then
    rg -n 'git\+.*[?&](tag|branch)=' Cargo.lock || true
    fail "Cargo.lock contains git sources resolved from mutable tag/branch refs"
elif rg -n 'git\+' Cargo.lock >/dev/null 2>&1; then
    echo "Cargo git sources are pinned by commit."
else
    echo "No Cargo git sources found."
fi

section "Rust unsafe boundary allowlist"
unsafe_allowlist_file="scripts/ci/unsafe-allowlist.txt"
if [[ ! -f "$unsafe_allowlist_file" ]]; then
    fail "$unsafe_allowlist_file is missing"
else
    unsafe_tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/pecos-unsafe.XXXXXX")"
    trap 'rm -rf "$unsafe_tmp_dir"' EXIT
    allowed_unsafe_roots_file="$unsafe_tmp_dir/allowed"
    actual_unsafe_roots_file="$unsafe_tmp_dir/actual"
    unsafe_files_without_manifest_file="$unsafe_tmp_dir/no-manifest"

    sed 's/[[:space:]]*#.*$//; /^[[:space:]]*$/d' "$unsafe_allowlist_file" | sort -u >"$allowed_unsafe_roots_file"
    : >"$actual_unsafe_roots_file"
    : >"$unsafe_files_without_manifest_file"

    while IFS= read -r unsafe_file; do
        dir="$(dirname "$unsafe_file")"
        root=""
        while [[ "$dir" != "." && "$dir" != "/" ]]; do
            if [[ -f "$dir/Cargo.toml" ]]; then
                root="$dir"
                break
            fi
            dir="$(dirname "$dir")"
        done

        if [[ -z "$root" ]]; then
            normalize_path "$unsafe_file" >>"$unsafe_files_without_manifest_file"
        else
            normalize_path "$root" >>"$actual_unsafe_roots_file"
        fi
    done < <(
        rg -l '\bunsafe\b' \
            crates python go julia exp \
            --glob '*.rs' \
            --glob '*.c' \
            --glob '*.cpp' \
            --glob '*.h' \
            --glob '*.hpp' \
            --glob '!crates/pecos-pymatching/tests/pymatching/**' \
            || true
    )

    sort -u "$actual_unsafe_roots_file" >"$actual_unsafe_roots_file.sorted"
    mv "$actual_unsafe_roots_file.sorted" "$actual_unsafe_roots_file"
    sort -u "$unsafe_files_without_manifest_file" >"$unsafe_files_without_manifest_file.sorted"
    mv "$unsafe_files_without_manifest_file.sorted" "$unsafe_files_without_manifest_file"

    unexpected_unsafe_roots="$(comm -23 "$actual_unsafe_roots_file" "$allowed_unsafe_roots_file" || true)"
    stale_unsafe_roots="$(comm -13 "$actual_unsafe_roots_file" "$allowed_unsafe_roots_file" || true)"

    if [[ -s "$unsafe_files_without_manifest_file" ]]; then
        cat "$unsafe_files_without_manifest_file"
        fail "unsafe usage found outside a Cargo package"
    fi
    if [[ -n "$unexpected_unsafe_roots" ]]; then
        printf '%s\n' "$unexpected_unsafe_roots"
        fail "new unsafe/FFI crate roots must be added to $unsafe_allowlist_file"
    fi
    if [[ -n "$stale_unsafe_roots" ]]; then
        printf '%s\n' "$stale_unsafe_roots"
        fail "stale unsafe/FFI allowlist entries should be removed"
    fi
    if [[ ! -s "$unsafe_files_without_manifest_file" &&
        -z "$unexpected_unsafe_roots" &&
        -z "$stale_unsafe_roots" ]]; then
        echo "Unsafe/FFI crate roots match the reviewed allowlist."
    fi
fi

section "uv lock discipline"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/target/uv-cache}"
if ! uv lock --check --project .; then
    fail "uv.lock is missing or not current with pyproject.toml"
else
    echo "uv.lock is current."
fi

section "GitHub Actions trigger posture"
if rg -n "pull_request_target|workflow_run" .github/workflows >/dev/null 2>&1; then
    rg -n "pull_request_target|workflow_run" .github/workflows || true
    fail "privileged workflow trigger found; review before running untrusted code"
else
    echo "No pull_request_target or workflow_run triggers found."
fi

section "GitHub Actions action pinning"
unpinned_actions="$(
    rg -n 'uses:[[:space:]]+[^[:space:]#]+@[^[:space:]#]+' .github/workflows .github/actions |
        awk -F: '{
            line = $0
            sub(/^[^:]+:[0-9]+:/, "", line)
            if (line ~ /uses:[[:space:]]+[^[:space:]#]+@[0-9a-f]{40}([[:space:]#]|$)/) {
                next
            }
            print
        }' ||
        true
)"
if [[ -n "$unpinned_actions" ]]; then
    printf '%s\n' "$unpinned_actions"
    fail "GitHub Actions uses entries must be pinned to immutable commit SHAs"
else
    echo "GitHub Actions uses entries are pinned to commit SHAs."
fi

section "Remote shell bootstrap posture"
remote_shell_bootstraps="$(
    rg -n '(curl|wget)[^\n|]*\|[^\n]*(sh|bash)' \
        .github/workflows \
        julia/PECOS.jl/deps/build_tarballs.jl \
        || true
)"
if [[ -n "$remote_shell_bootstraps" ]]; then
    printf '%s\n' "$remote_shell_bootstraps"
    fail "workflow and release build scripts must not pipe remote downloads into a shell"
else
    echo "Workflow and release build scripts avoid curl-pipe-shell bootstraps."
fi

section "Dependency review coverage"
if [[ ! -f .github/dependabot.yml && ! -f .github/dependabot.yaml ]]; then
    fail "Dependabot configuration is missing"
else
    echo "Dependabot configuration is present."
fi

dependency_review_workflow=""
if [[ -f .github/workflows/dependency-review.yml ]]; then
    dependency_review_workflow=".github/workflows/dependency-review.yml"
elif [[ -f .github/workflows/dependency-review.yaml ]]; then
    dependency_review_workflow=".github/workflows/dependency-review.yaml"
fi

if [[ -z "$dependency_review_workflow" ]]; then
    fail "GitHub dependency review workflow is missing"
else
    echo "GitHub dependency review workflow is present."
    if ! rg -q '^\s*push:\s*$' "$dependency_review_workflow"; then
        fail "GitHub dependency review workflow must run on push"
    fi
fi

actions_security_workflow=""
if [[ -f .github/workflows/github-actions-security.yml ]]; then
    actions_security_workflow=".github/workflows/github-actions-security.yml"
elif [[ -f .github/workflows/github-actions-security.yaml ]]; then
    actions_security_workflow=".github/workflows/github-actions-security.yaml"
fi

if [[ -z "$actions_security_workflow" ]]; then
    fail "GitHub Actions security analysis workflow is missing"
else
    echo "GitHub Actions security analysis workflow is present."
    if rg -q 'continue-on-error:\s*true' "$actions_security_workflow"; then
        fail "GitHub Actions security analysis workflow must be blocking"
    fi
fi

zizmor_config=""
if [[ -f .github/zizmor.yml ]]; then
    zizmor_config=".github/zizmor.yml"
elif [[ -f .github/zizmor.yaml ]]; then
    zizmor_config=".github/zizmor.yaml"
fi

if [[ -z "$zizmor_config" ]]; then
    fail "GitHub Actions security analysis configuration is missing"
else
    echo "GitHub Actions security analysis configuration is present."
    if ! rg -q 'hash-pin' "$zizmor_config"; then
        fail "GitHub Actions security analysis must enforce hash-pinned actions"
    fi
fi

if [[ ! -f .github/workflows/codeql.yml && ! -f .github/workflows/codeql.yaml ]]; then
    fail "CodeQL code scanning workflow is missing"
else
    echo "CodeQL code scanning workflow is present."
fi

if [[ ! -f .github/workflows/osv-scanner.yml && ! -f .github/workflows/osv-scanner.yaml ]]; then
    fail "OSV dependency vulnerability scanning workflow is missing"
else
    echo "OSV dependency vulnerability scanning workflow is present."
fi

if [[ ! -f deny.toml ]]; then
    fail "cargo-deny policy is missing"
else
    echo "cargo-deny policy is present."
fi

if [[ ! -f .github/workflows/cargo-deny.yml && ! -f .github/workflows/cargo-deny.yaml ]]; then
    fail "cargo-deny workflow is missing"
else
    echo "cargo-deny workflow is present."
fi

section "GitHub Actions cache write posture"
cache_policy_failures=()
while IFS=: read -r file line _; do
    if ! sed -n "${line},$((line + 16))p" "$file" | rg -q "save-if:.*github\.event_name == 'push'.*github\.ref_name == 'main'"; then
        cache_policy_failures+=("$file:$line rust-cache save-if must be restricted to trusted branch pushes")
    fi
done < <(rg -n 'uses:\s+Swatinem/rust-cache@' .github/workflows || true)

while IFS=: read -r file line _; do
    setup_uv_block="$(sed -n "${line},$((line + 16))p" "$file")"
    if printf '%s\n' "$setup_uv_block" | rg -q 'enable-cache:\s*true' &&
        ! printf '%s\n' "$setup_uv_block" | rg -q "save-cache:.*github\.event_name == 'push'.*github\.ref_name == 'main'"; then
        cache_policy_failures+=("$file:$line setup-uv save-cache must be restricted to trusted branch pushes")
    fi
done < <(rg -n 'uses:\s+astral-sh/setup-uv@' .github/workflows || true)

while IFS=: read -r file line _; do
    cache_policy_failures+=("$file:$line use actions/cache/restore plus an explicitly gated actions/cache/save step")
done < <(rg -n 'uses:\s+actions/cache@' .github/workflows || true)

while IFS=: read -r file line _; do
    if ! sed -n "$((line - 2)),$((line + 2))p" "$file" | rg -q "if:.*github\.event_name == 'push'.*github\.ref_name == 'main'"; then
        cache_policy_failures+=("$file:$line actions/cache/save must be restricted to trusted branch pushes")
    fi
done < <(rg -n 'uses:\s+actions/cache/save@' .github/workflows || true)

if ((${#cache_policy_failures[@]} > 0)); then
    printf '%s\n' "${cache_policy_failures[@]}"
    fail "cache writers must not save reusable caches from PR or untrusted branch runs"
else
    echo "Cache saves are restricted to trusted branch pushes."
fi

section "GitHub Actions lock enforcement"
cargo_workflow_commands="$(
    rg -n '^[[:space:]]*(run:[[:space:]]*)?cargo (build|check|clippy|run|install)([[:space:]]|$)' .github/workflows |
        rg -v '^[^:]+:[0-9]+:[[:space:]]*(run:[[:space:]]*)?cargo (build|check|clippy|run|install)[[:space:]]+--locked([[:space:]]|$)' ||
        true
)"
if [[ -n "$cargo_workflow_commands" ]]; then
    printf '%s\n' "$cargo_workflow_commands"
    fail "workflow Cargo build/check/run/install commands must use --locked"
else
    echo "Workflow Cargo build/check/run/install commands use --locked."
fi

uv_sync_without_lock="$(
    rg -n '^[[:space:]]*(run:[[:space:]]*)?uv sync([[:space:]]|$)' .github/workflows |
        rg -v -- '--locked' ||
        true
)"
if [[ -n "$uv_sync_without_lock" ]]; then
    printf '%s\n' "$uv_sync_without_lock"
    fail "workflow uv sync commands must use --locked"
else
    echo "Workflow uv sync commands use --locked."
fi

uv_lock_without_check="$(
    rg -n '^[[:space:]]*(run:[[:space:]]*)?uv lock([[:space:]]|$)' .github/workflows |
        rg -v -- '--check' ||
        true
)"
if [[ -n "$uv_lock_without_check" ]]; then
    printf '%s\n' "$uv_lock_without_check"
    fail "workflows must not regenerate uv.lock; use uv lock --check"
else
    echo "Workflows validate uv.lock instead of regenerating it."
fi

uv_run_without_frozen="$(
    rg -n '^[[:space:]]*(run:[[:space:]]*)?uv run([[:space:]]|$)' .github/workflows |
        rg -v '^[^:]+:[0-9]+:[[:space:]]*(run:[[:space:]]*)?uv run[[:space:]]+--frozen([[:space:]]|$)' ||
        true
)"
if [[ -n "$uv_run_without_frozen" ]]; then
    printf '%s\n' "$uv_run_without_frozen"
    fail "workflow uv run commands must use --frozen"
else
    echo "Workflow uv run commands use --frozen."
fi

section "Writable workflow permissions"
workflow_files=()
while IFS= read -r file; do
    workflow_files+=("$file")
done < <(collect_files .github/workflows -g '*.yml' -g '*.yaml')

missing_top_level_permissions=()
for file in "${workflow_files[@]}"; do
    if ! rg -q '^permissions:\s*$' "$file"; then
        missing_top_level_permissions+=("$file")
    fi
done

if ((${#missing_top_level_permissions[@]} > 0)); then
    printf '%s\n' "${missing_top_level_permissions[@]}"
    fail "workflow files must declare top-level read-only permissions"
fi

writable_permissions="$(rg -n '^[[:space:]]*(contents|packages|id-token|pull-requests|actions|security-events): write[[:space:]]*$' .github/workflows | sed 's#\\#/#g' || true)"
unexpected_writable_permissions="$(
    printf '%s\n' "$writable_permissions" | awk -F: '
        $1 == ".github/workflows/julia-update-hash.yml" &&
            $0 ~ /^[^:]+:[0-9]+:[[:space:]]+(contents|pull-requests): write[[:space:]]*$/ { next }
        $1 == ".github/workflows/codeql.yml" &&
            $0 ~ /^[^:]+:[0-9]+:[[:space:]]+security-events: write[[:space:]]*$/ { next }
        $1 == ".github/workflows/osv-scanner.yml" &&
            $0 ~ /^[^:]+:[0-9]+:[[:space:]]+security-events: write[[:space:]]*$/ { next }
        NF { print }
    '
)"

if [[ -n "$unexpected_writable_permissions" ]]; then
    printf '%s\n' "$unexpected_writable_permissions"
    fail "unexpected writable workflow permission found"
elif [[ -n "$writable_permissions" ]]; then
    echo "Only expected write permissions found."
else
    echo "No writable workflow permissions found."
fi

section "External binary download verification"
if rg -n 'sha256: None|checksum not available|does not publish SHA256' crates/pecos-build/src; then
    warn "some external binary installers cannot verify upstream checksums; prefer preinstalled dependencies in CI/release lanes"
else
    echo "External binary download paths have checksum verification."
fi

if ((failures > 0)); then
    printf '\nDependency integrity check failed with %d issue(s).\n' "$failures" >&2
    exit 1
fi

printf '\nDependency integrity check passed.\n'
