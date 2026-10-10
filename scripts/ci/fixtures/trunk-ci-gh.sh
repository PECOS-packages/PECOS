#!/bin/bash
# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
# Exact-argv fixture lookup using Bash builtins only: no Python or hash
# subprocess per request. Keys contain byte lengths, so argument boundaries,
# empty strings, newlines and shell metacharacters remain distinct.
set -euo pipefail
export LC_ALL=C

printf '%s\0' "$#" "$@" >> "$FAKE_GH_DIR/calls.bin"
if [[ "${1:-}" == issue && "${2:-}" =~ ^(create|close|comment)$ ]]; then
  printf '%s\0' "$#" "$@" >> "$FAKE_GH_DIR/mutations.bin"
fi

key=""
for arg in "$@"; do
  key+="${#arg}:${arg}"
done
# Length buckets keep each generated case table small. The table still tests
# the complete key, so equal-length requests cannot alias each other.
index=""
lookup_file="$FAKE_GH_DIR/requests/${#key}.bash"
if [[ -f "$lookup_file" ]]; then
  # Generated per test; it contains only literal case patterns and numeric IDs.
  # shellcheck source=/dev/null
  source "$lookup_file"
fi
if [[ -n "$index" ]]; then
  # A terminating NUL lets read preserve trailing newlines and empty output.
  {
    IFS= read -r code
    IFS= read -r -d '' response
  } < "$FAKE_GH_DIR/responses/$index"
  printf '%s' "$response"
  if [[ "$code" != 0 ]]; then
    printf 'Fixture API failure:' >&2
    printf ' %q' "$@" >&2
    printf '\n' >&2
  fi
  exit "$code"
fi

unexpected() {
  printf 'Unexpected gh argv:'
  for arg in "$@"; do
    printf ' %q' "$arg"
  done
  printf '\n'
}
unexpected "$@" >&2
unexpected "$@" >> "$FAKE_GH_DIR/unexpected.log"
exit 97
