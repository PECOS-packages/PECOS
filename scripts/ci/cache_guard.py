#!/usr/bin/env python3
# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Validate the cache-write gate on one GitHub Actions workflow step.

Usage::

    cache_guard.py FILE LINE KEY          # gate check
    cache_guard.py FILE LINE KEY --value  # print the input's scalar

FILE and LINE (1-based) locate a line inside a step, normally its ``uses:``
line. KEY is ``if`` (a step-level condition) or the name of a ``with:`` input
such as ``save-if``, ``save-cache`` or ``enable-cache``.

Gate check exit codes: 0 when the value is a literal ``false`` (never saves)
or an expression that only saves on pushes to a trusted branch; 1 otherwise;
2 when the step has no such input at all. ``--value`` prints the scalar and
exits 0, or exits 2 when absent.

The workflow is parsed as YAML, so comments, block scalars, quoting and key
order are handled by the parser rather than by line matching, and the lookup
only ever sees the owning step's own ``with:`` mapping.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

TRUSTED = {"main", "master", "development", "dev"}


def step_containing(root: yaml.Node, line0: int) -> yaml.MappingNode | None:
    """Return the step mapping whose source span contains 0-based ``line0``."""
    if not isinstance(root, yaml.MappingNode):
        return None
    for key, jobs in root.value:
        if key.value != "jobs" or not isinstance(jobs, yaml.MappingNode):
            continue
        for _, job in jobs.value:
            if not isinstance(job, yaml.MappingNode):
                continue
            for jkey, steps in job.value:
                if jkey.value != "steps" or not isinstance(steps, yaml.SequenceNode):
                    continue
                for step in steps.value:
                    if isinstance(step, yaml.MappingNode) and step.start_mark.line <= line0 < step.end_mark.line:
                        return step
    return None


def mapping_get(node: yaml.MappingNode, name: str) -> yaml.Node | None:
    for key, value in node.value:
        if isinstance(key, yaml.ScalarNode) and key.value == name:
            return value
    return None


def input_node(step: yaml.MappingNode, key: str) -> yaml.Node | None:
    """The node for ``key``: a direct step key for ``if``, else a ``with:`` input."""
    if key == "if":
        return mapping_get(step, "if")
    with_node = mapping_get(step, "with")
    if not isinstance(with_node, yaml.MappingNode):
        return None
    return mapping_get(with_node, key)


def has_unary_not(text: str) -> bool:
    in_q = False
    for j, ch in enumerate(text):
        if ch == "'":
            in_q = not in_q
            continue
        if in_q:
            continue
        if ch == "!" and not (j + 1 < len(text) and text[j + 1] == "="):
            return True
    return False


def split_top(text: str, op: str) -> list[str]:
    parts = []
    buf = ""
    depth = 0
    in_q = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "'":
            in_q = not in_q
            buf += ch
            i += 1
            continue
        if in_q:
            buf += ch
            i += 1
            continue
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth == 0 and text[i : i + 2] == op:
            parts.append(buf)
            buf = ""
            i += 2
            continue
        buf += ch
        i += 1
    parts.append(buf)
    return parts


def strip_parens(text: str) -> str:
    text = text.strip()
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        ok = True
        for j, ch in enumerate(text):
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and j != len(text) - 1:
                    ok = False
                    break
        if not ok:
            break
        text = text[1:-1].strip()
    return text


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def is_event_push(conjunct: str) -> bool:
    return conjunct == "github.event_name == 'push'"


def is_trusted_ref(conjunct: str) -> bool:
    m1 = re.match(
        r"^contains\(\s*fromJSON\(\s*'\[(.*)\]'\s*\)\s*,\s*github\.ref_name\s*\)$",
        conjunct,
    )
    if m1:
        names = {it.strip().strip('"').strip("'").strip() for it in m1.group(1).split(",")}
        return names == TRUSTED
    names = set()
    for alt in (norm(strip_parens(o)) for o in split_top(conjunct, "||")):
        mo = re.match(r"^github\.ref_name == '([^']*)'$", alt)
        if not mo:
            return False
        names.add(mo.group(1))
    return names == TRUSTED


def gate_is_trusted_push(value: str) -> bool:
    """True when ``value`` is a literal false or a push-to-trusted-branch predicate.

    ``if:`` conditions may omit the ``${{ }}`` wrapper; ``with:`` inputs include
    it. A bare ``true`` fails both predicates below and is therefore rejected.
    """
    value = norm(value)
    if value == "false":
        return True
    wrapped = re.match(r"^\$\{\{(.*)\}\}$", value)
    expr = wrapped.group(1).strip() if wrapped else value
    if has_unary_not(expr) or len(split_top(expr, "||")) > 1:
        return False
    conjuncts = [norm(strip_parens(c)) for c in split_top(expr, "&&")]
    return any(is_event_push(c) for c in conjuncts) and any(is_trusted_ref(c) for c in conjuncts)


def main(argv: list[str]) -> int:
    if len(argv) not in (4, 5) or (len(argv) == 5 and argv[4] != "--value"):
        print(__doc__, file=sys.stderr)
        return 1
    path, line, key = Path(argv[1]), int(argv[2]), argv[3]
    value_mode = len(argv) == 5

    root = yaml.compose(path.read_text())
    step = step_containing(root, line - 1)
    if step is None:
        print(f"{path}:{line}: not inside a workflow step", file=sys.stderr)
        return 1
    node = input_node(step, key)
    if node is None:
        return 2
    if not isinstance(node, yaml.ScalarNode):
        print(f"{path}:{line}: `{key}` is not a scalar", file=sys.stderr)
        return 1
    if value_mode:
        print(node.value)
        return 0
    return 0 if gate_is_trusted_push(node.value) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
