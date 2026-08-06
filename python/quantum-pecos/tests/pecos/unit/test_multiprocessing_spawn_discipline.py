# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""Prevent platform-default or explicit-fork multiprocessing in PECOS source."""

from __future__ import annotations

import re
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src" / "pecos"
_ALLOWLIST: frozenset[Path] = frozenset()
_FORK_HAZARD_PATTERNS = {
    "get_context(fork)": re.compile(r"\bget_context\s*\(\s*(['\"])fork\1\s*\)"),
    "set_start_method(fork)": re.compile(r"\bset_start_method\s*\(\s*(['\"])fork\1\s*\)"),
    "multiprocessing.Pool": re.compile(r"\bmultiprocessing\.Pool\s*\("),
    "mp.Pool": re.compile(r"\bmp\.Pool\s*\("),
    "bare Pool": re.compile(r"(?<![\w.])Pool\s*\("),
}


def test_pecos_source_avoids_fork_based_multiprocessing() -> None:
    """Require explicit non-fork multiprocessing throughout PECOS Python source."""
    violations = []
    for source_path in sorted(_SOURCE_ROOT.rglob("*.py")):
        relative_path = source_path.relative_to(_SOURCE_ROOT)
        if relative_path in _ALLOWLIST:
            continue
        for line_number, line in enumerate(source_path.read_text(encoding="utf-8").splitlines(), start=1):
            for pattern_name, pattern in _FORK_HAZARD_PATTERNS.items():
                if pattern.search(line):
                    violations.append(f"{relative_path}:{line_number}: {pattern_name}: {line.strip()}")

    assert not violations, "Fork-unsafe multiprocessing usage found:\n" + "\n".join(violations)
