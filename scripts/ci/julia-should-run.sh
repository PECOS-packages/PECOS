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

julia_paths='^(julia/|\.github/workflows/julia-|scripts/ci/julia-should-run\.sh$)'
changed="$(git diff --name-only "$PR_BASE_SHA"...HEAD)"

# Here-strings, not `printf | grep -q`: under pipefail an early-exiting grep can
# SIGPIPE the producer and turn a match into a failed pipeline.
if grep -qE "$julia_paths" <<<"$changed"; then
  echo "PR changes Julia files:" >&2
  grep -E "$julia_paths" <<<"$changed" | sed 's/^/  /' >&2
  echo "true"
else
  echo "PR changes no Julia files and lacks the ci:julia label; skipping Julia CI." >&2
  echo "false"
fi
