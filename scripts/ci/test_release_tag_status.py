# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Regression suite for release-tag workflow expectations."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import release_tag_status
import yaml
from release_tag_status import NON_VALIDATION, checkout_workflows, expected_workflows

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


def tag_matches(pattern: str, tag: str, filename: str) -> bool:
    """Support only the slash-aware * syntax used by this repository."""
    if "**" in pattern or any(char in pattern for char in "?+[]\\") or pattern.startswith("!"):
        message = f"{filename}: unsupported tag pattern {pattern!r}"
        raise ValueError(message)
    return re.fullmatch("[^/]*".join(re.escape(part) for part in pattern.split("*")), tag) is not None


def push_runs_for_tag(workflow: dict, tag: str, filename: str) -> bool:
    """Check tag admission for the structural owner-rule test, not expectations."""
    events = workflow.get("on", workflow.get(True, {}))
    if isinstance(events, str):
        return events == "push"
    if isinstance(events, list):
        return "push" in events
    if not isinstance(events, dict) or "push" not in events:
        return False
    push = events["push"] or {}
    if "tags-ignore" in push:
        message = f"{filename}: unsupported tags-ignore patterns {push['tags-ignore']!r}"
        raise ValueError(message)
    if "tags" in push:
        matches = [tag_matches(pattern, tag, filename) for pattern in push["tags"]]
        return any(matches)
    return "branches" not in push and "branches-ignore" not in push


def assert_validation_structure(directory: Path) -> None:
    validation = checkout_workflows(directory)
    assert validation, "No validation workflows found"
    problems = []
    for filename in validation:
        workflow = yaml.safe_load((directory / filename).read_text())
        problems.extend(
            f"{filename}: push does not admit {tag}"
            for tag in ("py-*", "jl-*", "rs-*")
            if not push_runs_for_tag(workflow, tag, filename)
        )
        events = workflow.get("on", workflow.get(True, {}))
        schedules = events.get("schedule", []) if isinstance(events, dict) else []
        if len(schedules) != 1:
            problems.append(f"{filename}: expected exactly one daily cron")
        else:
            fields = schedules[0]["cron"].split()
            if len(fields) != 5 or fields[2:] != ["*", "*", "*"]:
                problems.append(f"{filename}: cron must be daily")
    assert not problems, "\n".join(problems)


def stub_tag_tree(monkeypatch: pytest.MonkeyPatch, contents: dict[str, str]) -> list[tuple[str, ...]]:
    """Stub git reads without creating commits or modifying repository history."""
    calls = []

    def fake_git(*args: str) -> str:
        calls.append(args)
        if args[0] == "rev-parse":
            return "tag-sha\n"
        if args[0] == "ls-tree":
            return "\0".join(f".github/workflows/{filename}" for filename in contents) + "\0"
        assert args[0] == "show"
        return contents[args[1].removeprefix("tag-sha:.github/workflows/")]

    monkeypatch.setattr(release_tag_status, "git_output", fake_git)
    return calls


@pytest.mark.parametrize("tag", ["py-0.11.0.dev0", "jl-0.11.0", "rs-0.11.0"])
def test_real_workflows_expect_every_release_family(monkeypatch: pytest.MonkeyPatch, tag: str) -> None:
    stub_tag_tree(monkeypatch, {})
    assert set(checkout_workflows(WORKFLOWS)) == EXPECTED
    assert set(expected_workflows(tag, WORKFLOWS)) == EXPECTED
    assert expected_workflows(tag) == expected_workflows(tag, WORKFLOWS)


def test_real_validation_workflows_admit_release_tags_and_one_daily_cron() -> None:
    assert_validation_structure(WORKFLOWS)


@pytest.mark.parametrize(
    ("push", "schedules", "problem"),
    [
        (None, [], "push does not admit"),
        ({"tags": ["py-*", "jl-*", "rs-*"]}, [], "exactly one daily cron"),
        ({"tags": ["py-*", "jl-*", "rs-*"]}, [{"cron": "7 19 * * 1"}], "cron must be daily"),
        ({"tags": ["py-*", "jl-*", "rs-*"]}, [{"cron": "7 19 * * *"}] * 2, "exactly one daily cron"),
    ],
)
def test_owner_rule_rejects_scratch_workflow(
    tmp_path: Path,
    push: dict | None,
    schedules: list[dict],
    problem: str,
) -> None:
    directory = tmp_path / "workflows"
    shutil.copytree(WORKFLOWS, directory)
    events = {"workflow_dispatch": None, "schedule": schedules}
    if push is not None:
        events["push"] = push
    (directory / "new-validation.yaml").write_text(yaml.safe_dump({"on": events, "jobs": {}}))
    with pytest.raises(AssertionError, match=r"new-validation\.yaml") as error:
        assert_validation_structure(directory)
    assert problem in str(error.value)
    if push is None:
        assert "exactly one daily cron" in str(error.value)


def test_expected_workflows_union_includes_files_without_tag_triggers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = stub_tag_tree(
        monkeypatch,
        {
            "old-only.yaml": "name: Old only\non: pull_request\njobs: {}\n",
            "shared.yml": "name: Old name\non: pull_request\njobs: {}\n",
            "notes.txt": "not a workflow",
            **dict.fromkeys(NON_VALIDATION, "not parsed"),
        },
    )
    (tmp_path / "new-only.yml").write_text("name: New only\non: pull_request\njobs: {}\n")
    (tmp_path / "shared.yml").write_text("name: Current name\non: pull_request\njobs: {}\n")
    (tmp_path / "current.yaml").write_text("jobs: {}\n")
    assert expected_workflows("py-1", tmp_path) == {
        "current.yaml": "current.yaml",
        "new-only.yml": "New only",
        "old-only.yaml": "Old only",
        "shared.yml": "Current name",
    }
    assert calls[0] == ("rev-parse", "--verify", "refs/tags/py-1^{commit}")
    assert calls[1] == ("ls-tree", "--name-only", "-z", "tag-sha", ".github/workflows/")


def test_missing_local_tag_reports_fetch_instruction(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def missing_tag(*args: str) -> str:
        raise subprocess.CalledProcessError(128, args)

    monkeypatch.setattr(release_tag_status, "git_output", missing_tag)
    assert release_tag_status.main(["py-missing"]) == 1
    stderr = capsys.readouterr().err
    assert "Local tag 'py-missing'" in stderr
    assert "fetch" in stderr


def test_stale_local_tag_cannot_check_another_github_commit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(release_tag_status, "expected_workflows", lambda _tag: {"validation.yml": "Validation"})
    monkeypatch.setattr(release_tag_status, "local_tag_commit", lambda _tag: "old-local-sha")
    monkeypatch.setattr(release_tag_status, "gh_api", lambda *_args: "new-github-sha\n")
    assert release_tag_status.main(["py-1"]) == 1
    stderr = capsys.readouterr().err
    assert "Local tag 'py-1' differs from GitHub" in stderr
    assert "refresh" in stderr


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
    workflow = yaml.safe_load(path.read_text())
    assert push_runs_for_tag(workflow, tag, path.name) == expected


def test_non_validation_exclusion_applies_even_with_tag_trigger(tmp_path: Path) -> None:
    text = "name: Non-validation\non:\n  push:\n    tags: ['jl-*']\njobs: {}\n"
    for filename in NON_VALIDATION:
        (tmp_path / filename).write_text(text)
    assert checkout_workflows(tmp_path) == {}


@pytest.mark.parametrize("pattern", ["v[0-9]*", "v0-9]*", "py-**", "py-?", "py-+", r"py-\*", "!py-*"])
def test_unsupported_pattern_names_workflow(
    tmp_path: Path,
    pattern: str,
) -> None:
    path = tmp_path / "unsupported.yml"
    # The first pattern matches, but every pattern must still be validated.
    path.write_text(f"on:\n  push:\n    tags: ['py-*', '{pattern}']\njobs: {{}}\n")
    with pytest.raises(ValueError, match="unsupported tag pattern") as error:
        push_runs_for_tag(yaml.safe_load(path.read_text()), "py-1", path.name)
    assert path.name in str(error.value)
    assert repr(pattern) in str(error.value)


def test_tags_ignore_names_workflow_and_pattern(tmp_path: Path) -> None:
    path = tmp_path / "unsupported.yml"
    path.write_text("on:\n  push:\n    tags-ignore: ['jl-*']\njobs: {}\n")
    with pytest.raises(ValueError, match="unsupported tags-ignore") as error:
        push_runs_for_tag(yaml.safe_load(path.read_text()), "py-1", path.name)
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
    calls = []

    def fake_gh(*args: str) -> str:
        calls.append(args)
        return json.dumps(pages)

    monkeypatch.setattr(release_tag_status, "gh_api", fake_gh)
    assert release_tag_status.newest_tag_run("python-test.yml", "py-1", "current-sha")["id"] == 2
    assert release_tag_status.newest_tag_run("python-test.yml", "py-1", "missing-sha") is None
    for args in calls:
        assert args[args.index("branch=py-1") - 1] == "-f"
        assert args[args.index("event=push") - 1] == "-f"


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
    monkeypatch.setattr(release_tag_status, "local_tag_commit", lambda _tag: "current-sha")

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
