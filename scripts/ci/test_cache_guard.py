# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Regression suite for scripts/ci/cache_guard.py.

Every case here is a real or attempted bypass of the cache-write policy in
scripts/dependency-integrity-check.sh, so a change that makes any of the
``fail`` cases pass is weakening the gate, not fixing a false positive.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

GUARD = Path(__file__).with_name("cache_guard.py")
USES = "uses: astral-sh/setup-uv@20cfd1bf945f4377ade1205e4dbc17946fc9a30d # v10.0.1"
GATED = (
    "${{ github.event_name == 'push'"
    ' && contains(fromJSON(\'["main", "master", "development", "dev"]\'), github.ref_name) }}'
)
GATED_ORS = (
    "${{ github.event_name == 'push' && (github.ref_name == 'main' || github.ref_name == 'master'"
    " || github.ref_name == 'development' || github.ref_name == 'dev') }}"
)


def workflow(steps: str) -> str:
    return "name: t\non: push\njobs:\n  j:\n    runs-on: ubuntu-latest\n    steps:\n" + steps


def run(tmp_path: Path, text: str, key: str, *, value: bool = False, line: int | None = None) -> tuple[int, str]:
    path = tmp_path / "wf.yml"
    path.write_text(text)
    if line is None:
        line = next(i for i, ln in enumerate(text.split("\n"), 1) if "uses: " in ln)
    args = [sys.executable, str(GUARD), str(path), str(line), key] + (["--value"] if value else [])
    done = subprocess.run(args, capture_output=True, text=True, check=False)
    return done.returncode, done.stdout.strip()


def step(with_lines: str, *, name: str = "s", extra: str = "") -> str:
    return f"      - name: {name}\n        {USES}\n{extra}        with:\n{with_lines}"


# --- gate check -------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    [
        GATED,
        GATED_ORS,
        "false",
        f"{GATED[:-3]} && runner.os != 'Linux' }}}}",
        "${{ contains('a#b', 'a') && " + GATED[4:],
    ],
)
def test_gate_accepts_trusted_push_or_literal_false(tmp_path: Path, value: str) -> None:
    code, _ = run(tmp_path, workflow(step(f"          save-if: {value}\n")), "save-if")
    assert code == 0


def test_gate_accepts_trailing_yaml_comment(tmp_path: Path) -> None:
    code, _ = run(tmp_path, workflow(step(f"          save-if: {GATED} # restore on PRs\n")), "save-if")
    assert code == 0


@pytest.mark.parametrize(
    "value",
    [
        "true",
        "${{ true }}",
        "${{ github.event_name == 'push' }}",
        "${{ " + GATED[4:-3] + " || true }}",
        "${{ !github.event.pull_request && " + GATED[4:],
        "${{ github.event_name == 'push' && contains(fromJSON('[\"main\", \"feature\"]'), github.ref_name) }}",
    ],
)
def test_gate_rejects_wider_predicates(tmp_path: Path, value: str) -> None:
    code, _ = run(tmp_path, workflow(step(f"          save-if: {value}\n")), "save-if")
    assert code == 1


def test_gate_rejects_folded_scalar_hiding_or_true_behind_a_hash(tmp_path: Path) -> None:
    # In a folded block scalar `#` is content, not a comment, so GitHub sees the
    # whole expression including `|| true`. A line-based comment strip used to
    # truncate at ` #` and accept the two conjuncts that remained.
    folded = (
        "          save-if: >-\n"
        "            ${{ true && github.event_name == 'push'"
        ' && contains(fromJSON(\'["main", "master", "development", "dev"]\'), github.ref_name)'
        " && contains('x # y', 'x') || true }}\n"
    )
    code, _ = run(tmp_path, workflow(step(folded)), "save-if")
    assert code == 1


def test_gate_reads_step_level_if_without_wrapper(tmp_path: Path) -> None:
    text = workflow(f"      - name: save\n        if: {GATED[4:-3]}\n        uses: actions/cache/save@abc # v5\n")
    code, _ = run(tmp_path, text, "if")
    assert code == 0


def test_gate_absent_input_exits_two(tmp_path: Path) -> None:
    code, _ = run(tmp_path, workflow(step("          path: x\n")), "save-if")
    assert code == 2


# --- value lookup ------------------------------------------------------------


def test_value_reads_only_the_with_mapping(tmp_path: Path) -> None:
    # A block scalar in `name:` that happens to contain the input text must
    # not be mistaken for the input.
    text = workflow(
        "      - name: |\n          enable-cache: false\n        "
        + USES
        + "\n        with:\n          enable-cache: true\n",
    )
    code, out = run(tmp_path, text, "enable-cache", value=True)
    assert (code, out) == (0, "true")


def test_value_ignores_neighbouring_step(tmp_path: Path) -> None:
    text = workflow(
        step('          version-file: ".github/uv.toml"\n') + step("          enable-cache: false\n", name="other"),
    )
    code, _ = run(tmp_path, text, "enable-cache", value=True)
    assert code == 2


@pytest.mark.parametrize(
    ("literal", "expected"),
    [("false", "false"), ('"false"', "false"), ("${{ false }}", "${{ false }}")],
)
def test_value_prints_the_scalar(tmp_path: Path, literal: str, expected: str) -> None:
    code, out = run(tmp_path, workflow(step(f"          enable-cache: {literal}\n")), "enable-cache", value=True)
    assert (code, out) == (0, expected)


def test_value_strips_trailing_comment_only_on_plain_scalars(tmp_path: Path) -> None:
    code, out = run(
        tmp_path,
        workflow(step("          enable-cache: true # enable-cache: false\n")),
        "enable-cache",
        value=True,
    )
    assert (code, out) == (0, "true")


def test_step_is_found_from_list_item_uses_and_with_before_uses(tmp_path: Path) -> None:
    listed = f"      - {USES}\n        with:\n          enable-cache: false\n"
    code, out = run(tmp_path, workflow(listed), "enable-cache", value=True)
    assert (code, out) == (0, "false")
    reordered = f"      - name: s\n        with:\n          enable-cache: false\n        {USES}\n"
    code, out = run(tmp_path, workflow(reordered), "enable-cache", value=True)
    assert (code, out) == (0, "false")
