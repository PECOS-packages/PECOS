#!/usr/bin/env python3
# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Report validation for a release tag; exit 0 only when every workflow succeeds.

Usage: uv run --frozen python scripts/ci/release_tag_status.py <tag>

Requires authenticated gh, a git checkout, and the existing dev dependency
PyYAML. Fetch the release tag before running; fetching is the operator's job.
Every validation workflow in either the tag's commit or this checkout is
expected, regardless of its triggers, so older tags can report missing runs.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

import yaml

REPOSITORY = Path(__file__).resolve().parents[2]
WORKFLOWS = REPOSITORY / ".github/workflows"
NON_VALIDATION = {
    "dependency-review.yml": "Reviews dependency diffs rather than validating the whole commit.",
    "julia-update-hash.yml": "Opens a build-hash update PR rather than validating the tag.",
    "trunk-ci-issues.yml": "Writes CI tracking issues rather than validating the commit.",
    "pr-core-gate.yml": "PR-only duplicate of rust-test's Ubuntu leg and python-core.",
}


def git_output(*args: str) -> str:
    """Read local git objects without fetching or changing the checkout."""
    git = shutil.which("git")
    if git is None:
        message = "git is required to inspect the release tag's workflows"
        raise FileNotFoundError(message)
    return subprocess.run(
        [git, *args],
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def local_tag_commit(tag: str) -> str:
    """Resolve an actual local tag, including annotated tags, to its commit."""
    try:
        return git_output("rev-parse", "--verify", f"refs/tags/{tag}^{{commit}}").strip()
    except subprocess.CalledProcessError as error:
        message = f"Local tag {tag!r} is unavailable or does not point to a commit; fetch it before running this check"
        raise ValueError(message) from error


def checkout_workflows(directory: Path = WORKFLOWS) -> dict[str, str]:
    """Read every validation workflow, including files without tag triggers."""
    expected = {}
    for path in sorted([*directory.glob("*.yml"), *directory.glob("*.yaml")]):
        if path.name not in NON_VALIDATION:
            workflow = yaml.safe_load(path.read_text())
            expected[path.name] = workflow.get("name", path.name)
    return expected


def expected_workflows(tag: str, directory: Path = WORKFLOWS) -> dict[str, str]:
    """Expect the union of validation workflows in the local tag and checkout."""
    commit = local_tag_commit(tag)
    paths = git_output("ls-tree", "--name-only", "-z", commit, ".github/workflows/").split("\0")
    expected = {}
    for filename in paths:
        path = Path(filename)
        if path.suffix in {".yml", ".yaml"} and path.name not in NON_VALIDATION:
            workflow = yaml.safe_load(git_output("show", f"{commit}:{filename}"))
            expected[path.name] = workflow.get("name", path.name)
    expected.update(checkout_workflows(directory))
    return dict(sorted(expected.items()))


def gh_api(*args: str) -> str:
    """Query the local repository using gh's owner/repo placeholders."""
    gh = shutil.which("gh")
    if gh is None:
        message = "gh is required; install it and authenticate before checking release status"
        raise FileNotFoundError(message)
    # Run from this checkout so {owner}/{repo} names the repository the git reads use.
    return subprocess.run([gh, "api", *args], check=True, stdout=subprocess.PIPE, text=True, cwd=REPOSITORY).stdout


def github_tag_commit(tag: str) -> str:
    """Require the local workflow inventory to describe GitHub's tag commit."""
    sha = gh_api(f"repos/{{owner}}/{{repo}}/commits/tags/{quote(tag, safe='')}", "--jq", ".sha").strip()
    if sha != local_tag_commit(tag):
        message = f"Local tag {tag!r} differs from GitHub; refresh the local tag before running this check"
        raise ValueError(message)
    return sha


def newest_tag_run(workflow: str, tag: str, sha: str) -> dict | None:
    """Read all pages and select the newest push run for the tag's current SHA."""
    pages = json.loads(
        gh_api(
            f"repos/{{owner}}/{{repo}}/actions/workflows/{quote(workflow, safe='')}/runs",
            "--method",
            "GET",
            "-f",
            f"branch={tag}",
            "-f",
            "event=push",
            "-F",
            "per_page=100",
            "--paginate",
            "--slurp",
        ),
    )
    runs = (run for page in pages for run in page["workflow_runs"] if run["head_sha"] == sha)
    return max(runs, key=lambda run: (run["created_at"], run["id"]), default=None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    args = parser.parse_args(argv)
    try:
        expected = expected_workflows(args.tag)
        if not expected:
            print(f"No validation workflows expect tag {args.tag!r}", file=sys.stderr)
            return 1
        sha = github_tag_commit(args.tag)
    except (OSError, subprocess.CalledProcessError, yaml.YAMLError, ValueError) as error:
        print(f"Could not resolve release validation: {error}", file=sys.stderr)
        return 1

    success = True
    for filename, name in expected.items():
        try:
            run = newest_tag_run(filename, args.tag, sha)
        except (OSError, subprocess.CalledProcessError, ValueError) as error:
            print(f"{name} ({filename}): ERROR -")
            print(f"{filename}: {error}", file=sys.stderr)
            success = False
            continue
        if run is None:
            print(f"{name} ({filename}): MISSING -")
            success = False
        else:
            print(f"{name} ({filename}): {run['status']}/{run['conclusion'] or '-'} {run['html_url']}")
            success = success and run["status"] == "completed" and run["conclusion"] == "success"
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
