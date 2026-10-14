#!/usr/bin/env python3
# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""List the write permissions GitHub Actions workflows grant.

Usage::

    workflow_permissions.py FILE...

Prints one tab-separated line per write grant: ``FILE JOB SCOPE``. JOB is ``-``
for the workflow-level ``permissions:`` and SCOPE is ``write-all`` for that
shorthand. FILE is printed with forward slashes on every platform.

The workflow is parsed as YAML, so quoting, comments and spacing cannot hide a
grant, and every scope is reported rather than a fixed list of names. Mapping
keys stay as written, as GitHub reads them: YAML 1.1 would turn job ids such as
``on`` and ``yes`` into the same boolean and drop one job. Duplicate keys, which
GitHub rejects, exit 1, as does a ``permissions:`` value the GitHub schema does
not allow, so the check fails closed instead of reading an unknown form as
"no write".
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

LEVELS = {"read", "write", "none"}
SHORTHANDS = {"read-all", "write-all"}


class WorkflowLoader(yaml.SafeLoader):
    """SafeLoader that keeps scalar mapping keys as written and rejects duplicates."""


def construct_mapping(loader: WorkflowLoader, node: yaml.MappingNode) -> dict:
    loader.flatten_mapping(node)
    mapping = {}
    for key_node, value_node in node.value:
        scalar = isinstance(key_node, yaml.ScalarNode)
        key = key_node.value if scalar else loader.construct_object(key_node)
        if key in mapping:
            msg = f"duplicate key {key!r}"
            raise yaml.constructor.ConstructorError(None, None, msg, key_node.start_mark)
        mapping[key] = loader.construct_object(value_node)
    return mapping


WorkflowLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, construct_mapping)


def write_scopes(permissions: object, where: str) -> list[str]:
    """Return the scopes ``permissions`` grants write access to."""
    if permissions is None:
        return []
    if isinstance(permissions, str):
        if permissions not in SHORTHANDS:
            msg = f"{where}: unknown permissions shorthand {permissions!r}"
            raise ValueError(msg)
        return ["write-all"] if permissions == "write-all" else []
    if isinstance(permissions, dict):
        scopes = []
        for scope, level in permissions.items():
            if level not in LEVELS:
                msg = f"{where}: unknown level {level!r} for {scope!r}"
                raise ValueError(msg)
            if level == "write":
                scopes.append(str(scope))
        return scopes
    msg = f"{where}: permissions must be a mapping or shorthand, got {permissions!r}"
    raise TypeError(msg)


def grants(path: Path) -> list[tuple[str, str]]:
    """Return ``(job, scope)`` for every write grant in the workflow at ``path``."""
    loader = WorkflowLoader(path.read_text(encoding="utf-8"))
    try:
        workflow = loader.get_single_data()
    finally:
        loader.dispose()
    if not isinstance(workflow, dict):
        msg = f"{path}: not a workflow mapping"
        raise TypeError(msg)
    found = [("-", scope) for scope in write_scopes(workflow.get("permissions"), f"{path}: workflow")]
    jobs = workflow.get("jobs")
    if not isinstance(jobs, dict):
        msg = f"{path}: jobs must be a mapping"
        raise TypeError(msg)
    for job_id, job in jobs.items():
        if not isinstance(job, dict):
            msg = f"{path}: job {job_id!r} must be a mapping"
            raise TypeError(msg)
        found += [(str(job_id), scope) for scope in write_scopes(job.get("permissions"), f"{path}: job {job_id}")]
    return found


def main(argv: list[str]) -> int:
    # The integrity check splits this output in bash; Windows text mode would
    # otherwise end each line with CRLF and leave a CR on the last field.
    sys.stdout.reconfigure(newline="\n")
    try:
        for name in argv[1:]:
            for job, scope in grants(Path(name)):
                print(f"{Path(name).as_posix()}\t{job}\t{scope}")
    except (OSError, TypeError, ValueError, yaml.YAMLError) as err:
        print(f"workflow_permissions.py: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
