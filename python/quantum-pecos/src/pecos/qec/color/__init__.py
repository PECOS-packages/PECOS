# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Color-code geometry and cited 4.8.8/6.6.6 extraction constructions.

This module provides pure geometry for the 4.8.8 color code,
a topological QEC code with transversal Clifford gates, alongside explicit
paper code specifications, extraction implementations, and memory plans.

Classes:
    ColorCode488: Main color code class
    ColorCode488Geometry: Underlying geometry data
    ColorCodeStabilizer: Individual stabilizer data

Functions:
    generate_488_layout: Generate the 4.8.8 lattice layout
"""

from pecos.qec.color.clifford import (
    ColorCliffordExperiment,
    ColorFrame,
    LogicalState,
    color_state_memory,
    color_state_memory_program,
    logical_clifford_gadget,
)
from pecos.qec.color.code import (
    ColorCode488,
    ColorCode488Builder,
    ColorCode488Geometry,
    ColorCodeStabilizer,
)
from pecos.qec.color.geometry import generate_488_layout
from pecos.qec.color.lacroix import lacroix_memory, lacroix_memory_program
from pecos.qec.color.paper_codes import (
    ColorCodeImplementation,
    ColorExtraction,
    CSSCodeSpec,
    color_code_implementation,
    color_code_spec,
    extraction_gadget,
)
from pecos.qec.color.paper_memory import color_memory, color_memory_program
from pecos.qec.color.periodic488 import (
    Periodic488CodeSpec,
    Periodic488Implementation,
    periodic_488_code,
    periodic_488_implementation,
    periodic_488_memory,
    periodic_488_memory_program,
)

__all__ = [
    "CSSCodeSpec",
    "ColorCliffordExperiment",
    # Code classes
    "ColorCode488",
    "ColorCode488Builder",
    "ColorCode488Geometry",
    "ColorCodeImplementation",
    "ColorCodeStabilizer",
    "ColorExtraction",
    "ColorFrame",
    "LogicalState",
    "Periodic488CodeSpec",
    "Periodic488Implementation",
    "color_code_implementation",
    "color_code_spec",
    "color_memory",
    "color_memory_program",
    "color_state_memory",
    "color_state_memory_program",
    "extraction_gadget",
    # Geometry
    "generate_488_layout",
    "lacroix_memory",
    "lacroix_memory_program",
    "logical_clifford_gadget",
    "periodic_488_code",
    "periodic_488_implementation",
    "periodic_488_memory",
    "periodic_488_memory_program",
]
