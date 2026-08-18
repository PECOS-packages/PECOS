#!/usr/bin/env python3
# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Rewrite a version literal across a set of files, with rollback on failure.

Shared by the per-train bump commands (`scripts/bump_python_version.py`,
`scripts/bump_julia_version.py`). A bump that lands in some files and not others leaves a
train's members disagreeing, which is the state the version guards exist to catch, so the
rewrite is checked before it starts and unwound if a write fails partway.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

# Characters that may not sit against a version literal. `\w`, `.` and `-` keep a longer
# version from matching through this one; `+`, `!`, `:` and `/` keep local-version
# suffixes, PEP 440 epochs, and URL path segments from being rewritten in half.
BOUNDARY = r"[\w.!+:/-]"


def version_pattern(version: str) -> re.Pattern[str]:
    """Match `version` as a whole token.

    The boundaries let one pattern cover every place a version is written -- `version =
    "X"`, the `==X` inside a longer requirement string, Julia's `v"X"` -- while refusing to
    match a version or URL that merely contains this one.
    """
    return re.compile(rf"(?<!{BOUNDARY}){re.escape(version)}(?!{BOUNDARY})")


def rewrite_version(paths: Sequence[Path], old: str, new: str) -> tuple[list[Path], list[tuple[Path, int]]]:
    """Replace `old` with `new` in every path, or in none of them.

    Returns `(stale, changes)`. If any file does not carry `old` it is returned in `stale`
    and nothing is written; otherwise `changes` pairs each path with its replacement count.
    If a write fails, the files already written are restored before the error propagates.
    """
    pattern = version_pattern(old)
    contents = {path: path.read_text() for path in paths}

    stale = [path for path, text in contents.items() if not pattern.search(text)]
    if stale:
        return stale, []

    changes: list[tuple[Path, int]] = []
    try:
        for path, text in contents.items():
            updated, count = pattern.subn(new, text)
            path.write_text(updated)
            changes.append((path, count))
    except OSError:
        for path, _ in changes:
            path.write_text(contents[path])
        raise

    return [], changes
