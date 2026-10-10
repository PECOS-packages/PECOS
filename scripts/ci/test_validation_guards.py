# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Reject branch-only validation admission in workflow job and step ``if:`` gates.

Shell ``run:`` bodies are out of scope: this structural check inspects YAML
conditions only. Cache writes retain their trusted-branch policy, and release
publication has an explicit exception because it must only run for jl-* tags.
"""

from __future__ import annotations

import re
import shutil
from typing import TYPE_CHECKING

import pytest
import yaml
from cache_guard import gate_is_trusted_push, input_node, mapping_get, norm
from release_tag_status import WORKFLOWS, checkout_workflows

if TYPE_CHECKING:
    from pathlib import Path

BRANCH_GATE = re.compile(
    r"github\s*\.\s*(?:ref_name|head_ref)\b|github\s*\.\s*ref\s*==|refs/heads/"
    r"|contains\s*\(\s*fromJSON\s*\(\s*'\s*\[\s*\"(?:main|master|development|dev)\"",
    re.IGNORECASE,
)
# Keyed by (workflow, job, step index); None denotes a job-level condition.
ALLOWED_GUARDS = {
    ("julia-release.yml", "publish_release", None): (
        "github.event_name == 'push' && startsWith(github.ref, 'refs/tags/jl-')",
        "Only jl-* tag pushes may publish the GitHub release.",
    ),
}
OLD_CORE_GUARD = (
    'github.event_name == \'push\' && contains(fromJSON(\'["main", "master", "development", "dev"]\'), github.ref_name)'
)


def is_trusted_cache_write(step: yaml.MappingNode, condition: str) -> bool:
    """Identify cache writers and apply cache_guard's existing gate predicate."""
    # cache_guard exposes input lookup and gate validation, not writer detection.
    uses = mapping_get(step, "uses")
    cache_save = isinstance(uses, yaml.ScalarNode) and uses.value.split("@")[0] == "actions/cache/save"
    cache_inputs = any(input_node(step, key) is not None for key in ("save-if", "save-cache"))
    return (cache_save or cache_inputs) and gate_is_trusted_push(condition)


def assert_validation_guards(directory: Path) -> None:
    # checkout_workflows applies the status script's single exclusion map.
    validation = checkout_workflows(directory)
    assert validation, "No validation workflows found"
    problems = []
    for filename in validation:
        root = yaml.compose((directory / filename).read_text())
        assert isinstance(root, yaml.MappingNode), filename
        jobs = mapping_get(root, "jobs")
        assert isinstance(jobs, yaml.MappingNode), f"{filename}: jobs must be a mapping"
        for job_key, job in jobs.value:
            assert isinstance(job, yaml.MappingNode), f"{filename}: {job_key.value} must be a mapping"
            conditions = [(None, job)]
            steps = mapping_get(job, "steps")
            if isinstance(steps, yaml.SequenceNode):
                conditions.extend(enumerate(steps.value))
            for index, owner in conditions:
                assert isinstance(owner, yaml.MappingNode), f"{filename}: step must be a mapping"
                condition = mapping_get(owner, "if")
                if condition is None:
                    continue
                location = f"{filename}:{condition.start_mark.line + 1}: {job_key.value}"
                if index is not None:
                    location += f" step {index + 1}"
                assert isinstance(condition, yaml.ScalarNode), f"{location}: if must be a scalar"
                allowed = ALLOWED_GUARDS.get((filename, job_key.value, index))
                if allowed is not None:
                    if norm(condition.value) != norm(allowed[0]):
                        problems.append(f"{location}: allowlisted guard changed ({allowed[1]})")
                elif BRANCH_GATE.search(condition.value) and not (
                    index is not None and is_trusted_cache_write(owner, condition.value)
                ):
                    problems.append(f"{location}: branch-ref gate can skip validation: {condition.value}")
    assert not problems, "\n".join(problems)


def test_real_validation_workflows_have_no_branch_only_guards() -> None:
    assert_validation_guards(WORKFLOWS)


def test_old_python_core_guard_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "workflows"
    shutil.copytree(WORKFLOWS, directory)
    path = directory / "python-test.yml"
    root = yaml.compose(path.read_text())
    core = mapping_get(mapping_get(root, "jobs"), "python-core")
    steps = mapping_get(core, "steps")
    step = next(
        step
        for step in steps.value
        if isinstance(name := mapping_get(step, "name"), yaml.ScalarNode) and name.value == "Run core Python tests"
    )
    condition = mapping_get(step, "if")
    lines = path.read_text().splitlines(keepends=True)
    line = lines[condition.start_mark.line]
    lines[condition.start_mark.line] = f"{line[: len(line) - len(line.lstrip())]}if: {OLD_CORE_GUARD}\n"
    path.write_text("".join(lines))
    with pytest.raises(AssertionError, match=r"python-test\.yml:\d+: python-core step .*branch-ref gate"):
        assert_validation_guards(directory)


@pytest.mark.parametrize("level", ["job", "step"])
@pytest.mark.parametrize(
    "condition",
    [
        OLD_CORE_GUARD,
        "github.ref_name != 'release'",
        "github.ref == 'refs/tags/py-1'",
        "startsWith(github.ref, 'refs/heads/')",
        "github.head_ref == 'dev'",
        'contains(fromJSON(\'["main", "dev"]\'), inputs.branch)',
    ],
)
def test_branch_ref_guards_are_rejected(tmp_path: Path, level: str, condition: str) -> None:
    job = {"steps": [{"run": "validate"}]}
    owner = job if level == "job" else job["steps"][0]
    owner["if"] = condition
    (tmp_path / "validation.yaml").write_text(yaml.safe_dump({"jobs": {"validate": job}}))
    with pytest.raises(AssertionError, match=r"validation\.yaml:.*branch-ref gate"):
        assert_validation_guards(tmp_path)


@pytest.mark.parametrize(
    "writer",
    [
        {"uses": "actions/cache/save@pinned"},
        {"uses": "Swatinem/rust-cache@pinned", "with": {"save-if": OLD_CORE_GUARD}},
        {"uses": "astral-sh/setup-uv@pinned", "with": {"save-cache": OLD_CORE_GUARD}},
    ],
)
def test_cache_write_exception_is_limited_to_trusted_step_guards(tmp_path: Path, writer: dict) -> None:
    step = {**writer, "if": OLD_CORE_GUARD}
    workflow = {"jobs": {"validate": {"steps": [step]}}}
    path = tmp_path / "validation.yml"
    path.write_text(yaml.safe_dump(workflow))
    assert_validation_guards(tmp_path)
    workflow["jobs"]["validate"]["if"] = OLD_CORE_GUARD
    path.write_text(yaml.safe_dump(workflow))
    with pytest.raises(AssertionError, match="branch-ref gate"):
        assert_validation_guards(tmp_path)


def test_publication_exception_cannot_hide_a_branch_guard(tmp_path: Path) -> None:
    path = tmp_path / "julia-release.yml"
    job = {"if": ALLOWED_GUARDS[(path.name, "publish_release", None)][0], "steps": []}
    path.write_text(yaml.safe_dump({"jobs": {"publish_release": job}}))
    assert_validation_guards(tmp_path)
    job["if"] = OLD_CORE_GUARD
    path.write_text(yaml.safe_dump({"jobs": {"publish_release": job}}))
    with pytest.raises(AssertionError, match="allowlisted guard changed"):
        assert_validation_guards(tmp_path)
