"""
LLVM IR generation API implemented in Rust via PyO3 and inkwell.

This module provides a drop-in replacement for llvmlite, enabling:
- Python 3.13+ support (llvmlite doesn't support it)
- Reduced Python dependencies
- High-performance LLVM IR generation using Rust

Usage:
    from pecos_rslib.llvm import ir, binding

This is compatible with:
    from llvmlite import ir, binding

But implemented entirely in Rust for better performance and compatibility.
"""

from pecos_rslib._pecos_rslib import binding, ir

__all__ = ["ir", "binding"]
