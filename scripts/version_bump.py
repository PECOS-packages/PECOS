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

import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

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


def write_atomic(path: Path, text: str) -> None:
    """Replace `path`'s contents through a same-directory temporary file.

    `Path.write_text` truncates first, so a write that fails partway leaves the file
    damaged and outside the caller's record of what was written. Renaming a fully written
    temporary file over the original means a file is either wholly old or wholly new.
    """
    handle_fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(handle_fd, "w") as handle:
            handle.write(text)
        tmp.replace(path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


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
            write_atomic(path, updated)
            changes.append((path, count))
    except OSError as err:
        unrestored = []
        for path, _ in changes:
            try:
                write_atomic(path, contents[path])
            except OSError:
                unrestored.append(path)
        if unrestored:
            msg = f"{err}; and these files could not be restored: {', '.join(str(p) for p in unrestored)}"
            raise RuntimeError(msg) from err
        raise

    return [], changes
