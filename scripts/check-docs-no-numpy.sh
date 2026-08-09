#!/usr/bin/env bash
# Reject NumPy in documentation code snippets.
#
# PECOS owns its numeric primitives; NumPy belongs in the test suite as an oracle. Ruff
# enforces that for .py and .ipynb via flake8-tidy-imports banned-api (TID251), but it does
# not lint Markdown, and the doctests generated from docs/ are not committed -- so docs are
# the one place the ban cannot reach. This closes that gap.
#
# Docs matter more than their line count suggests: a snippet in the user guide teaches the
# pattern, and readers copy it into their own code.
#
# Only fenced code blocks are scanned. Prose may name NumPy freely -- a migration guide has
# to be able to write "replace np.array(x) with array(x)" without tripping this check.
#
# Tracking issue for the remaining migration: #458.
set -euo pipefail

cd "$(dirname "$0")/.."

hits=$(
    find docs -name '*.md' -type f -print0 |
        xargs -0 awk '
            # Track fenced blocks. Only code inside a fence is subject to the ban.
            /^[[:space:]]*```/ {
                if (in_fence) { in_fence = 0 } else { in_fence = 1 }
                next
            }
            in_fence && /(^|[^A-Za-z0-9_])(import[[:space:]]+numpy|from[[:space:]]+numpy[[:space:]]+import|np\.)/ {
                printf "%s:%d: %s\n", FILENAME, FNR, $0
            }
        '
)

if [ -n "$hits" ]; then
    echo "NumPy found in documentation code snippets:" >&2
    echo "$hits" | sed 's/^/  /' >&2
    cat >&2 <<'MSG'

PECOS owns its numerics. Use the NumPy-compatible layer re-exported from `pecos`
(array, zeros, ones, arange, array_equal, dtypes, any, all, ...), or the standard
library where that is the honest answer -- `math.pi` rather than `np.pi`.

NumPy is permitted in tests only, as an oracle to compare PECOS's own results against.
Prose outside a code fence may mention NumPy freely; only fenced code is checked.
See issue #458.
MSG
    exit 1
fi
