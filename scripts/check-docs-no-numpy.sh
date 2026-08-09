#!/usr/bin/env bash
# Reject accidental NumPy in documentation code snippets.
#
# PECOS owns its numeric primitives and is self-sufficient: internal code must not depend on
# NumPy. Ruff enforces that for .py and .ipynb via flake8-tidy-imports banned-api (TID251),
# but it does not lint Markdown, and the doctests generated from docs/ are not committed --
# so docs are the one place the ban cannot reach. This closes that gap.
#
# Docs matter more than their line count suggests: a snippet in the user guide teaches the
# pattern, and readers copy it into their own code.
#
# Two deliberate exemptions, because the goal is "PECOS does not depend on NumPy", not
# "PECOS pretends NumPy does not exist":
#
#   1. Prose. Only fenced code blocks are scanned, so a migration guide can write
#      "replace np.array(x) with array(x)" without tripping this check.
#
#   2. Interop documentation. NumPy is ubiquitous in scientific computing, and users are
#      entitled to know how to move data between it and PECOS -- np.asarray(pecos_array)
#      and pecos.array(np_array) both work. Mark such a block with an HTML comment on the
#      line before the fence:
#
#          <!-- numpy-interop: converting a PECOS array for an existing NumPy workflow -->
#          ```python
#          import numpy as np
#          ...
#          ```
#
#      The marker is per-block and must state why, so the exemption is a deliberate act by
#      an author rather than something that spreads by copy-paste.
#
# Tracking issue for the remaining migration: #458.
set -euo pipefail

cd "$(dirname "$0")/.."

hits=$(
    find docs -name '*.md' -type f -print0 |
        xargs -0 awk '
            # Remember whether the most recent non-blank line opted this block in.
            /^[[:space:]]*$/ { next }

            /^[[:space:]]*```/ {
                if (in_fence) {
                    in_fence = 0
                } else {
                    in_fence = 1
                    exempt = pending_exempt
                }
                pending_exempt = 0
                next
            }

            {
                pending_exempt = (!in_fence && /numpy-interop/) ? 1 : 0
            }

            in_fence && !exempt && /(^|[^A-Za-z0-9_])(import[[:space:]]+numpy|from[[:space:]]+numpy[[:space:]]+import|np\.)/ {
                printf "%s:%d: %s\n", FILENAME, FNR, $0
            }
        '
)

if [ -n "$hits" ]; then
    echo "NumPy found in documentation code snippets:" >&2
    echo "$hits" | sed 's/^/  /' >&2
    cat >&2 <<'MSG'

PECOS owns its numerics and does not depend on NumPy. In documentation examples use the
NumPy-compatible layer re-exported from `pecos` (array, zeros, ones, arange, array_equal,
dtypes, any, all, ...), or the standard library where that is the honest answer --
`math.pi` rather than `np.pi`.

If the block is deliberately teaching NumPy interop, mark it and say why:

    <!-- numpy-interop: converting a PECOS array for an existing NumPy workflow -->

Prose outside a code fence may mention NumPy freely; only fenced code is checked.
NumPy also remains available in tests, as an oracle. See issue #458.
MSG
    exit 1
fi
