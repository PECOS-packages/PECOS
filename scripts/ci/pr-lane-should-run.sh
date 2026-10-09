#!/usr/bin/env bash
# Decide whether an optional CI lane (Julia, Selene plugins) should run for this
# event. Prints "true" or "false" on stdout; all diagnostics go to stderr. Pull
# requests require a full-history checkout (fetch-depth: 0).
#
# Pushes (merges into the main branches, release tags) and manual runs always
# run. A pull request runs only when it carries the lane's label or changes one
# of the lane's paths. Changes to this script count for every lane.
#
# Usage: scripts/ci/pr-lane-should-run.sh <label> <pathspec>...
# Environment: EVENT_NAME, PR_LABELS (JSON array of label names), PR_BASE_SHA.
set -euo pipefail

label="$1"
shift
# Git pathspecs, not a regex over printed names: git quotes unusual filenames,
# and a printed rename shows only its destination, hiding a move out of a lane.
lane_paths=("$@" scripts/ci/pr-lane-should-run.sh)

if [ "$EVENT_NAME" != "pull_request" ]; then
  echo "Event '$EVENT_NAME' always runs this lane." >&2
  echo "true"
  exit 0
fi

# Case-insensitive, like the label comparisons in the workflow expressions.
has_label="$(jq -r --arg wanted "$label" 'map(ascii_downcase) | index($wanted | ascii_downcase) != null' <<<"$PR_LABELS")"
if [ "$has_label" = "true" ]; then
  echo "PR carries the $label label; running this lane." >&2
  echo "true"
  exit 0
fi

diff_range="${PR_BASE_SHA:?PR_BASE_SHA is empty}...HEAD"

# --quiet exits 0 for no changes, 1 for changes, and anything else on error.
# --no-ext-diff/--no-textconv: compare blobs, not a configured driver's view.
status=0
git diff --quiet --no-ext-diff --no-textconv "$diff_range" -- "${lane_paths[@]}" || status=$?
case "$status" in
  0)
    echo "PR changes no lane files and lacks the $label label; skipping this lane." >&2
    echo "false"
    ;;
  1)
    echo "PR changes lane files:" >&2
    git diff --name-only "$diff_range" -- "${lane_paths[@]}" | sed 's/^/  /' >&2
    echo "true"
    ;;
  *)
    echo "git diff failed with status $status" >&2
    exit "$status"
    ;;
esac
