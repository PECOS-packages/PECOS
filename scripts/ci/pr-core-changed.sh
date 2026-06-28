#!/usr/bin/env bash
# Decide whether a pull request's changes can affect the core Python/Rust test
# suites the pr-core gate runs (`just python-ci-core` / `just rstest`). Prints
# "true" or "false" on stdout; all diagnostics go to stderr. Requires a
# full-history checkout (fetch-depth: 0).
#
# Usage: scripts/ci/pr-core-changed.sh <pr-base-sha>
set -euo pipefail

base="${1:-}"
if [ -z "$base" ]; then
  echo "No PR base provided (e.g. manual run); running the core gate." >&2
  echo "true"
  exit 0
fi

changed="$(git diff --name-only "$base"...HEAD)"
{
  echo "Changed files:"
  printf '  %s\n' $changed
} >&2

# Allowlist of paths that PROVABLY cannot affect python-ci-core or
# `pecos rust test`:
#   - other-language CI workflows (julia/go/codeql) and issue templates
#   - root-level prose only ([^/]+); root README/CHANGELOG are not pyproject
#     `readme=` inputs (verified).
# docs/ is intentionally NOT ignored: docs/assets/** holds test fixtures (e.g.
# docs/assets/test-data/math.wat) and docs/*.md sources generated tests under
# python/quantum-pecos/tests/docs/, both collected by the non-slow core
# selection. Nested package READMEs (python/**/README.md) are also build inputs
# and run. Anything not on the allowlist runs.
ignore='^(\.github/ISSUE_TEMPLATE/|\.github/workflows/(julia-|go-|codeql)|[^/]+\.(md|rst|txt)$|LICENSE$|CITATION(\.cff)?$)'

if printf '%s\n' "$changed" | grep -qvE "$ignore"; then
  echo "true"
else
  echo "false"
fi
