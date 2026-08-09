#!/usr/bin/env bash
# Reject NumPy in documentation snippets.
#
# PECOS owns its numeric primitives; NumPy belongs in the test suite as an oracle. Ruff
# enforces that for .py and .ipynb via flake8-tidy-imports banned-api (TID251), but it does
# not lint Markdown, and the doctests generated from docs/ are not committed -- so docs are
# the one place the ban cannot reach. This closes that gap.
#
# Docs matter more than their line count suggests: a snippet in the user guide teaches the
# pattern, and readers copy it into their own code.
#
# Tracking issue for the remaining migration: #458.
set -euo pipefail

cd "$(dirname "$0")/.."

# Match `import numpy`, `from numpy import ...`, and `np.` usage inside docs.
pattern='(^|[^A-Za-z0-9_])(import[[:space:]]+numpy|from[[:space:]]+numpy[[:space:]]+import|np\.)'

hits=$(grep -rInE "$pattern" --include='*.md' docs/ 2>/dev/null || true)

if [ -n "$hits" ]; then
    echo "NumPy found in documentation snippets:" >&2
    echo "$hits" | sed 's/^/  /' >&2
    cat >&2 <<'MSG'

PECOS owns its numerics. Use the NumPy-compatible layer re-exported from `pecos`
(array, zeros, ones, arange, array_equal, dtypes, any, all, ...), or the standard
library where that is the honest answer -- `math.pi` rather than `np.pi`.

NumPy is permitted in tests only, as an oracle to compare PECOS's own results against.
See issue #458.
MSG
    exit 1
fi
