# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""LLVM-independent QIS text analysis and runtime helper normalization."""

from __future__ import annotations

PECOS_HELPER_ABIS: dict[str, str]

class ResultTagCertificationError(ValueError):
    """The lowered program is outside the supported certification subset."""


def extract_result_tag_measurements(qis: str) -> tuple[dict[str, list[int | None]], int]: ...
def normalize_pecos_helper_symbols(ir: str) -> str: ...
