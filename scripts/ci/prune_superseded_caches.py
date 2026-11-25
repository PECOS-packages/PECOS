#!/usr/bin/env python3
# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Delete Rust caches on the main branches that a newer save has superseded.

Usage: python scripts/ci/prune_superseded_caches.py [--dry-run]

Requires authenticated gh with actions write access and GH_REPO set.

Swatinem/rust-cache keys are ``<prefix-key>-<job>-<os>-<arch>-<env hash>-<lockfile
hash>``, and a run restores only entries with its own env hash. Each Cargo.lock or
manifest change saves a new entry under a new lockfile hash and nothing removes the
old one, so the old one keeps its gigabytes until the repository cache limit evicts
the least recently used entries, which are often the caches pull requests restore.

An entry is deleted only when all of these hold:

- its key starts with one of this repository's rust-cache prefix keys and ends in
  two 8-digit hex hashes;
- it belongs to a main branch, never to a pull request;
- a newer entry exists on the same branch with the same key apart from the
  lockfile hash, and the same cache version (paths and compression);
- nothing has restored it for ``IDLE``, judged by GitHub's ``last_accessed_at``;
  an entry without one is kept.

A newer upload can come from an older commit whose run finished late, so the idle
rule, not upload order, decides what is dead: an entry that runs keep restoring
survives, and one deleted by mistake is saved again by the next run that needs it.
A pull request still on an older lockfile can lose its exact entry once that entry
has been idle for ``IDLE``; it then restores the newest entry by prefix, a partial
hit rather than a miss. That is the accepted cost: without pruning, the cache limit
evicts whole caches that pull requests restore.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime, timedelta

TRUSTED_REFS = {f"refs/heads/{branch}" for branch in ("main", "master", "development", "dev")}
# Every prefix-key passed to Swatinem/rust-cache in .github/workflows; a test keeps
# this in step with the workflows.
PREFIX_KEYS = ("v1-rust-no-bin", "v1-selene-general-noise-semantics")
RUST_CACHE_KEY = re.compile(
    rf"^(?P<group>(?:{'|'.join(map(re.escape, PREFIX_KEYS))})-.+-[0-9a-f]{{8}})-[0-9a-f]{{8}}$",
)
IDLE = timedelta(hours=2)


def timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def idle(cache: dict, now: datetime) -> bool:
    accessed = cache.get("last_accessed_at")
    return accessed is not None and now - timestamp(accessed) >= IDLE


def superseded(caches: list[dict], now: datetime) -> list[dict]:
    """Return the entries of ``caches`` that the rules in the module docstring delete."""
    groups: dict[tuple[str, str, str], list[dict]] = {}
    unique = {cache["id"]: cache for cache in caches}
    for cache in unique.values():
        match = RUST_CACHE_KEY.match(cache["key"])
        if cache["ref"] in TRUSTED_REFS and match is not None:
            groups.setdefault((cache["ref"], match["group"], cache["version"]), []).append(cache)
    stale = []
    for entries in groups.values():
        entries.sort(key=lambda cache: timestamp(cache["created_at"]))
        stale.extend(cache for cache in entries[:-1] if idle(cache, now))
    return stale


def gh(*args: str) -> str:
    executable = shutil.which("gh")
    if executable is None:
        message = "gh is required to list and delete Actions caches"
        raise OSError(message)
    return subprocess.run([executable, *args], check=True, capture_output=True, text=True).stdout


def list_caches() -> list[dict]:
    # Sorting by creation keeps entries from moving between pages while runs restore
    # caches; superseded() also drops any entry listed twice.
    pages = gh(
        "api",
        "--paginate",
        "repos/{owner}/{repo}/actions/caches?per_page=100&sort=created_at&direction=asc",
        "--jq",
        ".actions_caches[]",
    )
    return [json.loads(line) for line in pages.splitlines() if line]


def delete(cache: dict) -> bool:
    """Delete one cache; return False when GitHub has already evicted it."""
    try:
        gh("api", "--method", "DELETE", f"repos/{{owner}}/{{repo}}/actions/caches/{cache['id']}")
    except subprocess.CalledProcessError as error:
        if "HTTP 404" in error.stderr:
            return False
        raise
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="report what would be deleted without deleting it")
    args = parser.parse_args(argv)

    freed = 0
    deleted = 0
    try:
        stale = superseded(list_caches(), datetime.now(UTC))
        for cache in stale:
            label = f"{cache['key']} ({cache['ref']}, {cache['size_in_bytes'] / 1e9:.2f} GB)"
            if args.dry_run:
                print(f"Would delete {label}")
            elif delete(cache):
                print(f"Deleted {label}")
            else:
                print(f"Already gone: {label}")
                continue
            deleted += 1
            freed += cache["size_in_bytes"]
    except (OSError, json.JSONDecodeError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as error:
        print(f"error: {' '.join(error.cmd)} failed: {error.stderr.strip()}", file=sys.stderr)
        print(f"{deleted} caches deleted before the failure, {freed / 1e9:.2f} GB", file=sys.stderr)
        return 1
    verb = "would delete" if args.dry_run else "deleted"
    print(f"{verb} {deleted} superseded caches, {freed / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
