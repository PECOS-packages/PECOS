"""Program types for PECOS quantum simulation.

This module provides the Rust program types for the unified simulation API.
"""

# Import the Rust program types
from pecos_rslib._pecos_rslib import (
    HugrProgram,
    PhirJsonProgram,
    QasmProgram,
    QisProgram,
    WasmProgram,
    WatProgram,
)


__all__ = [
    "HugrProgram",
    "PhirJsonProgram",
    "QasmProgram",
    "QisProgram",
    "WasmProgram",
    "WatProgram",
]
