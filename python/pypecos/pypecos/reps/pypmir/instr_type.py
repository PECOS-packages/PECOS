from __future__ import annotations


class Instr:
    """Base type for all PyMIR instructions including QOps, Blocks, MOps, etc."""

    def __init__(self, metadata: dict | None = None) -> None:
        self.metadata = metadata
