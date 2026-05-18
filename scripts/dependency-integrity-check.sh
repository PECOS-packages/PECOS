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
require_tool rg || true
require_tool cargo || true
require_tool uv || true

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
    if rg -n --pcre2 '^\s*(tag|branch)\s*=' "${cargo_manifests[@]}"; then
        fail "Cargo git dependencies must use full immutable rev pins, not tag/branch"
    fi
    if rg -n --pcre2 '^\s*rev\s*=\s*"[0-9a-f]{1,39}"' "${cargo_manifests[@]}"; then
        fail "Cargo git dependency rev pins must use full 40-character commit SHAs"
    fi
fi

if rg -n --pcre2 'git\+.*[?&](tag|branch)=' Cargo.lock >/dev/null 2>&1; then
    rg -n --pcre2 'git\+.*[?&](tag|branch)=' Cargo.lock || true
    fail "Cargo.lock contains git sources resolved from mutable tag/branch refs"
elif rg -n 'git\+' Cargo.lock >/dev/null 2>&1; then
    echo "Cargo git sources are pinned by commit."
else
    echo "No Cargo git sources found."
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

section "GitHub Actions lock enforcement"
if rg -n --pcre2 '^\s*(run:\s*)?cargo (build|check|clippy|run|install)(?! --locked)' .github/workflows; then
    fail "workflow Cargo build/check/run/install commands must use --locked"
else
    echo "Workflow Cargo build/check/run/install commands use --locked."
fi

if rg -n --pcre2 '^\s*(run:\s*)?uv sync(?!.*--locked)' .github/workflows; then
    fail "workflow uv sync commands must use --locked"
else
    echo "Workflow uv sync commands use --locked."
fi

if rg -n --pcre2 '^\s*(run:\s*)?uv lock(?!.*--check)' .github/workflows; then
    fail "workflows must not regenerate uv.lock; use uv lock --check"
else
    echo "Workflows validate uv.lock instead of regenerating it."
fi

if rg -n --pcre2 '^\s*(run:\s*)?uv run(?! --frozen)' .github/workflows; then
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

writable_permissions="$(rg -n '^\s*(contents|packages|id-token|pull-requests|actions|security-events): write\s*$' .github/workflows || true)"
unexpected_writable_permissions="$(
    printf '%s\n' "$writable_permissions" | awk -F: '
        $1 == ".github/workflows/julia-update-hash.yml" &&
            $0 ~ /^[^:]+:[0-9]+:[[:space:]]+(contents|pull-requests): write[[:space:]]*$/ { next }
        NF { print }
    '
)"

if [[ -n "$unexpected_writable_permissions" ]]; then
    printf '%s\n' "$unexpected_writable_permissions"
    fail "unexpected writable workflow permission found"
elif [[ -n "$writable_permissions" ]]; then
    echo "Only expected write permissions found in the tag-only Julia hash updater."
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
