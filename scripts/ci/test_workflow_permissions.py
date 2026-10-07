# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Regression suite for scripts/ci/workflow_permissions.py.

The write-permission allowlist in scripts/dependency-integrity-check.sh trusts
this parser to report every write grant, so each ``reported`` case below is a
spelling that a line matcher missed, and each ``rejected`` case is a value that
must fail the check rather than read as "no write".
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PARSER = Path(__file__).with_name("workflow_permissions.py")


def run(tmp_path: Path, text: str) -> tuple[int, list[str]]:
    path = tmp_path / "wf.yml"
    path.write_text(text)
    done = subprocess.run([sys.executable, str(PARSER), str(path)], capture_output=True, text=True, check=False)
    grants = [line.split("\t", 1)[1] for line in done.stdout.splitlines()]
    return done.returncode, grants


def workflow(top: str = "", job: str = "") -> str:
    return f"name: t\non: push\n{top}jobs:\n  build:\n    runs-on: ubuntu-latest\n{job}    steps:\n      - run: true\n"


@pytest.mark.parametrize(
    ("top", "job", "expected"),
    [
        ("permissions:\n  contents: write\n", "", ["-\tcontents"]),
        ('permissions:\n  contents: "write"\n', "", ["-\tcontents"]),
        ("permissions:\n  contents: 'write'\n", "", ["-\tcontents"]),
        ("permissions:\n  contents: write # publish\n", "", ["-\tcontents"]),
        ("permissions:\n  contents:   write\n", "", ["-\tcontents"]),
        ("permissions: {contents: write}\n", "", ["-\tcontents"]),
        ("permissions:\n  issues: write\n", "", ["-\tissues"]),
        ("permissions: write-all\n", "", ["-\twrite-all"]),
        ("", "    permissions: write-all\n", ["build\twrite-all"]),
        ("", "    permissions:\n      contents: read\n      pull-requests: write\n", ["build\tpull-requests"]),
        (
            "permissions:\n  security-events: write\n",
            "    permissions:\n      contents: write\n",
            ["-\tsecurity-events", "build\tcontents"],
        ),
    ],
)
def test_reported(tmp_path: Path, top: str, job: str, expected: list[str]) -> None:
    assert run(tmp_path, workflow(top, job)) == (0, expected)


@pytest.mark.parametrize(
    ("top", "job"),
    [
        ("", ""),
        ("permissions:\n  contents: read\n", ""),
        ("permissions: read-all\n", ""),
        ("permissions: {}\n", "    permissions:\n      contents: none\n"),
    ],
)
def test_read_only(tmp_path: Path, top: str, job: str) -> None:
    assert run(tmp_path, workflow(top, job)) == (0, [])


@pytest.mark.parametrize(
    "text",
    [
        workflow("permissions: write\n"),
        workflow("permissions:\n  contents: admin\n"),
        workflow("permissions:\n  - contents\n"),
        workflow("", "    permissions: true\n"),
        "name: t\non: push\njobs:\n  build: run\n",
        "name: t\non: push\n",
        "jobs: [\n",
    ],
)
def test_rejected(tmp_path: Path, text: str) -> None:
    assert run(tmp_path, text)[0] == 1
