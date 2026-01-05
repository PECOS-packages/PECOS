# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Direct Guppy code generation for quantum error correction.

This module provides code generation utilities for creating Guppy
quantum programs from QEC geometry definitions. It bypasses the SLR
intermediate representation for faster direct Guppy generation.

Submodules:
    surface: Surface code generation
    color: Color code generation
    transversal: Transversal operations (CNOT for CSS codes)

Example:
    >>> from pecos.guppy import make_surface_code, get_num_qubits
    >>> prog = make_surface_code(distance=3, num_rounds=3, basis="Z")
    >>> num_qubits = get_num_qubits(3)
    >>> result = prog.emulator(num_qubits=num_qubits).stabilizer_sim().run()
"""

from pecos.guppy.color import (
    generate_color_code_module,
    generate_color_code_source,
    get_color_code_module,
    get_num_qubits_color,
    make_color_code,
)
from pecos.guppy.surface import (
    generate_guppy_source,
    generate_memory_experiment,
    generate_surface_code_module,
    get_num_qubits,
    get_surface_code_module,
    make_surface_code,
)
from pecos.guppy.transversal import (
    CSSCodeType,
    get_transversal_num_qubits,
    make_color_transversal_cnot,
    make_color_transversal_cnot_d3,
    make_color_transversal_cnot_with_x,
    make_color_transversal_cnot_with_x_d3,
    make_css_transversal_cnot,
    make_css_transversal_cnot_with_x,
    make_surface_transversal_cnot,
    make_surface_transversal_cnot_with_x,
)

__all__ = [  # noqa: RUF022
    # Surface code
    "generate_guppy_source",
    "generate_memory_experiment",
    "generate_surface_code_module",
    "get_num_qubits",
    "get_surface_code_module",
    "make_surface_code",
    # Color code
    "generate_color_code_module",
    "generate_color_code_source",
    "get_color_code_module",
    "get_num_qubits_color",
    "make_color_code",
    # Generic CSS transversal operations
    "CSSCodeType",
    "get_transversal_num_qubits",
    "make_css_transversal_cnot",
    "make_css_transversal_cnot_with_x",
    "make_color_transversal_cnot",
    "make_color_transversal_cnot_d3",
    "make_color_transversal_cnot_with_x",
    "make_color_transversal_cnot_with_x_d3",
    "make_surface_transversal_cnot",
    "make_surface_transversal_cnot_with_x",
]
