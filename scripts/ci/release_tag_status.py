#!/usr/bin/env python3
# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Report validation for a release tag; exit 0 only when every workflow succeeds.

Usage: python3 scripts/ci/release_tag_status.py <tag>

Requires authenticated gh and the existing dev dependency PyYAML. Expectations
come from this checkout's workflows, so older tags can report missing runs.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

import yaml

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github/workflows"
NON_VALIDATION = {
    "julia-update-hash.yml": "Opens a build-hash update PR rather than validating the tag.",
}


def tag_matches(pattern: str, tag: str, filename: str) -> bool:
    """Match literal text and slash-aware *; reject all other ref-glob syntax."""
    if "**" in pattern or any(char in pattern for char in "?+[]\\") or pattern.startswith("!"):
        message = f"{filename}: unsupported tag pattern {pattern!r}; only literal text and * are supported"
        raise ValueError(message)
    expression = "[^/]*".join(re.escape(part) for part in pattern.split("*"))
    return re.fullmatch(expression, tag) is not None


def push_runs_for_tag(workflow: dict, tag: str, filename: str) -> bool:
    """Return whether on.push admits this tag; path filters do not affect tags."""
    # PyYAML's YAML 1.1 loader reads an unquoted on key as boolean True.
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
        # Evaluate every pattern so an earlier match cannot hide unsupported syntax.
        matches = [tag_matches(pattern, tag, filename) for pattern in push["tags"]]
        return any(matches)
    return "branches" not in push and "branches-ignore" not in push


def expected_workflows(tag: str, directory: Path = WORKFLOWS) -> dict[str, str]:
    """Derive filename -> display name from every validation workflow in the tree."""
    expected = {}
    for path in sorted(directory.glob("*.yml")):
        workflow = yaml.safe_load(path.read_text())
        if path.name not in NON_VALIDATION and push_runs_for_tag(workflow, tag, path.name):
            expected[path.name] = workflow.get("name", path.name)
    return expected


def gh_api(*args: str) -> str:
    """Query the local repository using gh's owner/repo placeholders."""
    gh = shutil.which("gh")
    if gh is None:
        message = "gh is required; install it and authenticate before checking release status"
        raise FileNotFoundError(message)
    return subprocess.run([gh, "api", *args], check=True, stdout=subprocess.PIPE, text=True).stdout


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
        sha = gh_api(f"repos/{{owner}}/{{repo}}/commits/{quote(args.tag, safe='')}", "--jq", ".sha").strip()
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
