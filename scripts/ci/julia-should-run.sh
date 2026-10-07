#!/usr/bin/env bash
# Decide whether the Julia CI workflows should run for this event. Prints
# "true" or "false" on stdout; all diagnostics go to stderr. Pull requests
# require a full-history checkout (fetch-depth: 0).
#
# Pushes (merges into the main branches, jl-* tags) and manual runs always run.
# A pull request runs only when it carries the `ci:julia` label or changes
# julia/ or a Julia CI file.
#
# Environment: EVENT_NAME, HAS_JULIA_LABEL ("true"/"false"), PR_BASE_SHA.
set -euo pipefail

if [ "$EVENT_NAME" != "pull_request" ]; then
  echo "Event '$EVENT_NAME' always runs Julia CI." >&2
  echo "true"
  exit 0
fi

if [ "$HAS_JULIA_LABEL" = "true" ]; then
  echo "PR carries the ci:julia label; running Julia CI." >&2
  echo "true"
  exit 0
fi

# Git pathspecs, not a regex over printed names: git quotes unusual filenames,
# and a printed rename shows only its destination, hiding a move out of julia/.
julia_paths=(julia/ '.github/workflows/julia-*' scripts/ci/julia-should-run.sh)
diff_range="${PR_BASE_SHA:?PR_BASE_SHA is empty}...HEAD"

# --quiet exits 0 for no changes, 1 for changes, and anything else on error.
# --no-ext-diff/--no-textconv: compare blobs, not a configured driver's view.
status=0
git diff --quiet --no-ext-diff --no-textconv "$diff_range" -- "${julia_paths[@]}" || status=$?
case "$status" in
  0)
    echo "PR changes no Julia files and lacks the ci:julia label; skipping Julia CI." >&2
    echo "false"
    ;;
  1)
    echo "PR changes Julia files:" >&2
    git diff --name-only "$diff_range" -- "${julia_paths[@]}" | sed 's/^/  /' >&2
    echo "true"
    ;;
  *)
    echo "git diff failed with status $status" >&2
    exit "$status"
    ;;
esac
