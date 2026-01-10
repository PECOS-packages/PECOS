# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""4.8.8 Triangular Color Code geometry.

This module provides pure geometry for the 4.8.8 color code,
a topological QEC code with transversal Clifford gates.

Classes:
    ColorCode488: Main color code class
    ColorCode488Geometry: Underlying geometry data
    ColorCodeStabilizer: Individual stabilizer data

Functions:
    generate_488_layout: Generate the 4.8.8 lattice layout
"""

from pecos.qec.color.code import (
    ColorCode488,
    ColorCode488Builder,
    ColorCode488Geometry,
    ColorCodeStabilizer,
)
from pecos.qec.color.geometry import generate_488_layout

__all__ = [
    # Code classes
    "ColorCode488",
    "ColorCode488Builder",
    "ColorCode488Geometry",
    "ColorCodeStabilizer",
    # Geometry
    "generate_488_layout",
]
