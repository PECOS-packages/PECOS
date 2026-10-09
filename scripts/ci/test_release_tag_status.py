# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Regression suite for release-tag workflow expectations."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import release_tag_status
from release_tag_status import expected_workflows

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github/workflows"
EXPECTED = {
    "cargo-deny.yml",
    "codeql.yml",
    "cuda-build-check.yml",
    "dependency-integrity-check.yml",
    "github-actions-security.yml",
    "julia-release.yml",
    "julia-test.yml",
    "julia-version-consistency.yml",
    "nightly.yml",
    "osv-scanner.yml",
    "pre-commit.yml",
    "python-release.yml",
    "python-test.yml",
    "python-version-consistency.yml",
    "rust-test.yml",
    "rust-version-consistency.yml",
    "selene-general-noise-semantics.yml",
    "selene-plugins.yml",
    "test-docs-examples.yml",
}


@pytest.mark.parametrize("tag", ["py-0.11.0.dev0", "jl-0.11.0", "rs-0.11.0"])
def test_real_workflows_expect_every_release_family(tag: str) -> None:
    assert set(expected_workflows(tag, WORKFLOWS)) == EXPECTED
    assert expected_workflows(tag) == expected_workflows(tag, WORKFLOWS)


@pytest.mark.parametrize(
    ("trigger", "tag", "expected"),
    [
        ("on:\n  push:\n    tags: ['py-*']", "py-1", True),
        ("on:\n  push:\n    tags: ['py-*']", "jl-1", False),
        ("on:\n  push:\n    tags: ['py-*']", "py-1/nested", False),
        ("on:\n  push:\n    tags: ['py-*']", "py-", True),
        ("on:\n  push:\n    tags: ['py-1.0*']", "py-1x0", False),
        ("on:\n  push:\n    tags: ['py-1!*']", "py-1!rc", True),
        ("on:\n  push:\n    paths: ['src/**']", "jl-1", True),
        ("on:\n  push:\n    paths-ignore: ['docs/**']", "rs-1", True),
        ("on:\n  push:", "rs-1", True),
        ("on: push", "jl-1", True),
        ("on: [push, pull_request]", "py-1", True),
        ("'on':\n  push:\n    tags: ['jl-*']", "jl-1", True),
        ("on:\n  pull_request:", "py-1", False),
        ("on:\n  push:\n    branches: ['dev']", "py-1", False),
        ("on:\n  push:\n    branches-ignore: ['dev']", "py-1", False),
        ("on:\n  push:\n    branches: ['dev']\n    tags: ['py-*']", "py-1", True),
    ],
)
def test_synthetic_push_filters(tmp_path: Path, trigger: str, tag: str, expected: bool) -> None:
    path = tmp_path / "validation.yml"
    path.write_text(f"name: Synthetic validation\n{trigger}\njobs: {{}}\n")
    assert expected_workflows(tag, tmp_path) == ({path.name: "Synthetic validation"} if expected else {})


def test_non_validation_exclusion_applies_even_with_tag_trigger(tmp_path: Path) -> None:
    text = "name: Non-validation\non:\n  push:\n    tags: ['jl-*']\njobs: {}\n"
    (tmp_path / "julia-update-hash.yml").write_text(text)
    assert expected_workflows("jl-1", tmp_path) == {}


@pytest.mark.parametrize("pattern", ["v[0-9]*", "v0-9]*", "py-**", "py-?", "py-+", r"py-\*", "!py-*"])
def test_unsupported_pattern_names_workflow_and_fails_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    pattern: str,
) -> None:
    path = tmp_path / "unsupported.yml"
    # The first pattern matches, but every pattern must still be validated.
    path.write_text(f"on:\n  push:\n    tags: ['py-*', '{pattern}']\njobs: {{}}\n")
    with pytest.raises(ValueError, match="unsupported tag pattern") as error:
        expected_workflows("py-1", tmp_path)
    assert path.name in str(error.value)
    assert repr(pattern) in str(error.value)

    monkeypatch.setattr(release_tag_status, "expected_workflows", lambda tag: expected_workflows(tag, tmp_path))
    assert release_tag_status.main(["py-1"]) == 1
    stderr = capsys.readouterr().err
    assert path.name in stderr
    assert repr(pattern) in stderr


def test_tags_ignore_names_workflow_and_pattern(tmp_path: Path) -> None:
    path = tmp_path / "unsupported.yml"
    path.write_text("on:\n  push:\n    tags-ignore: ['jl-*']\njobs: {}\n")
    with pytest.raises(ValueError, match="unsupported tags-ignore") as error:
        expected_workflows("py-1", tmp_path)
    assert path.name in str(error.value)
    assert "jl-*" in str(error.value)


def test_newest_run_must_match_current_tag_sha(monkeypatch: pytest.MonkeyPatch) -> None:
    wrong_sha = {"id": 3, "created_at": "2026-10-09T10:00:00Z", "head_sha": "old-sha"}
    older_match = {"id": 1, "created_at": "2026-10-09T08:00:00Z", "head_sha": "current-sha"}
    newest_match = {"id": 2, "created_at": "2026-10-09T09:00:00Z", "head_sha": "current-sha"}
    pages = [
        {"workflow_runs": [wrong_sha, older_match]},
        {"workflow_runs": [newest_match]},
    ]
    monkeypatch.setattr(release_tag_status, "gh_api", lambda *_args: json.dumps(pages))
    assert release_tag_status.newest_tag_run("python-test.yml", "py-1", "current-sha")["id"] == 2
    assert release_tag_status.newest_tag_run("python-test.yml", "py-1", "missing-sha") is None


@pytest.mark.parametrize(
    ("status", "conclusion", "exit_code"),
    [
        ("completed", "success", 0),
        ("completed", "failure", 1),
        ("completed", "skipped", 1),
        ("completed", "cancelled", 1),
        ("in_progress", None, 1),
        ("queued", None, 1),
        (None, None, 1),
    ],
)
def test_release_exit_requires_every_workflow_success(
    monkeypatch: pytest.MonkeyPatch,
    status: str | None,
    conclusion: str | None,
    exit_code: int,
) -> None:
    monkeypatch.setattr(release_tag_status, "expected_workflows", lambda _tag: {"one.yml": "One", "two.yml": "Two"})
    monkeypatch.setattr(release_tag_status, "gh_api", lambda *_args: "current-sha\n")

    def fake_run(workflow: str, tag: str, sha: str) -> dict | None:
        assert tag == "py-1"
        assert sha == "current-sha"
        if workflow == "one.yml":
            return {"status": "completed", "conclusion": "success", "html_url": "https://example.com/one"}
        return (
            None
            if status is None
            else {
                "status": status,
                "conclusion": conclusion,
                "html_url": "https://example.com/two",
            }
        )

    monkeypatch.setattr(release_tag_status, "newest_tag_run", fake_run)
    assert release_tag_status.main(["py-1"]) == exit_code
