# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Regression suite for scripts/ci/prune_superseded_caches.py."""

from __future__ import annotations

import itertools
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import prune_superseded_caches as prune
import yaml

if TYPE_CHECKING:
    import pytest

DEV = "refs/heads/dev"
NOW = datetime(2026, 11, 24, 12, tzinfo=UTC)
IDLE_SINCE = "2026-11-24T09:00:00Z"
# main() reads the real clock, so its fixtures must be idle at any later date.
LONG_IDLE = {"last_accessed_at": "2026-11-22T00:00:00Z"}
WORKFLOWS = Path(__file__).resolve().parents[2] / ".github/workflows"
LINUX = "v1-rust-no-bin-rust-test-Linux-x64-993ed683"
_ids = itertools.count()


def cache(key: str, created_at: str, ref: str = DEV, last_accessed_at: str = IDLE_SINCE, version: str = "v") -> dict:
    return {
        "id": next(_ids),
        "key": key,
        "ref": ref,
        "created_at": created_at,
        "last_accessed_at": last_accessed_at,
        "version": version,
        "size_in_bytes": 1,
    }


def ids(caches: list[dict]) -> list[int]:
    return sorted(entry["id"] for entry in caches)


def test_older_lockfile_hashes_are_superseded_in_any_order() -> None:
    old = cache(f"{LINUX}-cbf9e437", "2026-11-24T00:26:00Z")
    older = cache(f"{LINUX}-9f12d428", "2026-11-20T03:13:00Z")
    new = cache(f"{LINUX}-e98e8b33", "2026-11-24T00:39:00Z")
    for order in itertools.permutations([old, older, new]):
        assert ids(prune.superseded(list(order), NOW)) == ids([old, older])


def test_recently_restored_entries_survive_a_newer_upload() -> None:
    # A late run of an older commit can upload after the current lockfile's entry,
    # and a pull request on an older lockfile restores its exact entry.
    in_use = cache(f"{LINUX}-cbf9e437", "2026-11-24T00:26:00Z", last_accessed_at="2026-11-24T11:00:00Z")
    newer = cache(f"{LINUX}-e98e8b33", "2026-11-24T00:39:00Z")
    assert prune.superseded([in_use, newer], NOW) == []


def test_entries_without_an_access_time_are_kept() -> None:
    unknown = cache(f"{LINUX}-9f12d428", "2026-11-20T00:00:00Z")
    del unknown["last_accessed_at"]
    null = cache(f"{LINUX}-cbf9e437", "2026-11-21T00:00:00Z", last_accessed_at=None)
    stale = cache(f"{LINUX}-aaaaaaaa", "2026-11-22T00:00:00Z")
    newest = cache(f"{LINUX}-e98e8b33", "2026-11-24T00:00:00Z")
    assert ids(prune.superseded([unknown, null, stale, newest], NOW)) == ids([stale])


def test_an_entry_listed_twice_is_not_its_own_newer_copy() -> None:
    # Page-number pagination can return the same entry on two pages.
    only = cache(f"{LINUX}-e98e8b33", "2026-11-24T00:39:00Z")
    assert prune.superseded([only, dict(only)], NOW) == []


def test_each_env_hash_os_job_and_version_keeps_its_newest_entry() -> None:
    caches = [
        cache(f"{LINUX}-e98e8b33", "2026-11-24T00:39:00Z"),
        cache(f"{LINUX}-cbf9e437", "2026-11-21T00:00:00Z", version="other paths"),
        cache("v1-rust-no-bin-rust-test-Linux-x64-aaaaaaaa-11111111", "2026-11-20T00:00:00Z"),
        cache("v1-rust-no-bin-rust-test-Darwin-arm64-28c06b73-cbf9e437", "2026-11-21T00:00:00Z"),
        cache("v1-rust-no-bin-python-core-Linux-x64-993ed683-cbf9e437", "2026-11-21T00:00:00Z"),
    ]
    assert prune.superseded(caches, NOW) == []


def test_pull_request_and_untrusted_refs_are_never_touched() -> None:
    caches = [
        cache(f"{LINUX}-{lock}", created, ref)
        for ref in ("refs/pull/1728/merge", "refs/heads/feature")
        for lock, created in (("cbf9e437", "2026-11-21T00:00:00Z"), ("e98e8b33", "2026-11-24T00:00:00Z"))
    ]
    assert prune.superseded(caches, NOW) == []


def test_each_trusted_branch_is_pruned_separately() -> None:
    main_old = cache(f"{LINUX}-cbf9e437", "2026-11-21T00:00:00Z", "refs/heads/main")
    dev_new = cache(f"{LINUX}-e98e8b33", "2026-11-24T00:00:00Z")
    assert prune.superseded([main_old, dev_new], NOW) == []


def test_only_rust_cache_namespaces_are_touched() -> None:
    uv = "setup-uv-2-x86_64-unknown-linux-gnu-ubuntu-26.04"
    caches = [
        cache("dataset-deadbeef-11111111", "2026-11-20T00:00:00Z"),
        cache("dataset-deadbeef-22222222", "2026-11-24T00:00:00Z"),
        cache("llvm-21.1.8-win22-X64-v3", "2026-11-20T00:00:00Z"),
        cache(f"{uv}-3.12.16-{'1' * 64}", "2026-11-20T00:00:00Z"),
        cache(f"{uv}-unknown-{'1' * 64}", "2026-11-24T00:00:00Z"),
    ]
    assert prune.superseded(caches, NOW) == []
    selene = [
        cache(f"v1-selene-general-noise-semantics-semantics-Linux-x64-1234abcd-{lock}", created)
        for lock, created in (("cbf9e437", "2026-11-21T00:00:00Z"), ("e98e8b33", "2026-11-24T00:00:00Z"))
    ]
    assert ids(prune.superseded(selene, NOW)) == ids(selene[:1])


def test_prefix_keys_match_the_workflows() -> None:
    used = set()
    for path in WORKFLOWS.glob("*.yml"):
        for job in (yaml.safe_load(path.read_text()).get("jobs") or {}).values():
            for step in job.get("steps") or []:
                if str(step.get("uses", "")).startswith("Swatinem/rust-cache@"):
                    used.add((step.get("with") or {}).get("prefix-key"))
    assert used == set(prune.PREFIX_KEYS)


def test_an_already_evicted_cache_does_not_stop_the_run(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    gone = cache(f"{LINUX}-9f12d428", "2026-11-20T00:00:00Z", **LONG_IDLE)
    stale = cache(f"{LINUX}-cbf9e437", "2026-11-21T00:00:00Z", **LONG_IDLE)
    new = cache(f"{LINUX}-e98e8b33", "2026-11-24T00:00:00Z")
    deleted = []

    def fake_gh(*args: str) -> str:
        cache_id = int(args[-1].rsplit("/", 1)[1])
        if cache_id == gone["id"]:
            raise subprocess.CalledProcessError(1, ["gh", *args], stderr="gh: Not Found (HTTP 404)")
        deleted.append(cache_id)
        return ""

    monkeypatch.setattr(prune, "list_caches", lambda: [gone, stale, new])
    monkeypatch.setattr(prune, "gh", fake_gh)
    assert prune.main([]) == 0
    assert deleted == [stale["id"]]
    assert "Already gone" in capsys.readouterr().out


def test_other_delete_failures_stop_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    caches = [
        cache(f"{LINUX}-cbf9e437", "2026-11-21T00:00:00Z", **LONG_IDLE),
        cache(f"{LINUX}-e98e8b33", "2026-11-24T00:00:00Z"),
    ]

    def fake_gh(*args: str) -> str:
        raise subprocess.CalledProcessError(1, ["gh", *args], stderr="gh: Resource not accessible (HTTP 403)")

    monkeypatch.setattr(prune, "list_caches", lambda: caches)
    monkeypatch.setattr(prune, "gh", fake_gh)
    assert prune.main([]) == 1
