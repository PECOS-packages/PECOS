"""Guppy code generation for SLR programs.

This module provides the entry point for Guppy code generation.
The actual implementation is in the guppy/ subdirectory.
"""

from pecos.slr.gen_codes.guppy import IRGuppyGenerator

# Alias for convenience
GuppyGenerator = IRGuppyGenerator

__all__ = ["GuppyGenerator", "IRGuppyGenerator"]
