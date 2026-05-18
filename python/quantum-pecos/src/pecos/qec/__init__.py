# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Pure QEC geometry and abstractions.

This module provides code-agnostic QEC geometry and data structures
with no SLR or runtime dependencies.

Submodules:
    analysis: Result analysis and post-processing utilities
    generic: Generic stabilizer check framework
    protocols: Protocol geometry (MSD, etc.)
    surface: Surface code geometry (square and rotated lattices)
    color: Color code geometry (4.8.8 triangular layout)

Example:
    >>> from pecos.qec.surface import compute_x_stabilizer_supports
    >>> stabs = compute_x_stabilizer_supports(d=3)
    >>> print(f"X stabilizers: {len(stabs)}")

    >>> from pecos.qec.color import ColorCode488
    >>> code = ColorCode488.create(distance=3)
    >>> print(f"Data qubits: {code.num_data}")
"""

from pecos_rslib.qec import (
    # Pauli constants
    PAULI_I,
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    DagFaultAnalyzer,
    DagFaultInfluenceMap,
    DemBuilder,
    DemSampler,
    DemSamplerBuilder,
    EquivalenceResult,
    FaultLocation,
    InfluenceBuilder,
    ParsedDem,
    assert_dems_equivalent,
    compare_dems_exact,
    compare_dems_statistical,
    verify_dem_equivalence,
)

from pecos.qec import analysis, color, protocols, surface
from pecos.qec.analysis import (
    build_adaptive_dem,
    compare_flip_matrices,
    compare_k_body_rates,
    detector_flip_matrices_by_round,
    detector_flip_matrix,
    detector_k_body_rates,
    detector_k_body_rates_by_round,
    empirical_correlation_table,
    fit_dem_from_simulation,
    logical_error_rate,
    logical_fidelity,
    logical_from_data,
    logical_x_from_data,
    logical_z_from_data,
    lower_bound_fidelity,
    syndrome_difference,
    syndrome_to_detection_events,
)
from pecos.qec.color import (
    ColorCode488,
    ColorCode488Builder,
    ColorCode488Geometry,
    ColorCodeStabilizer,
    generate_488_layout,
)

# DetectorErrorModel is re-exported from pecos.qec.dem: a thin Python subclass
# of the Rust class that adds the from_guppy convenience constructor (the
# Guppy/Selene trace pipeline is Python-only, so it cannot live in the Rust
# extension without a dependency cycle).
from pecos.qec.dem import DetectorErrorModel
from pecos.qec.generic import (
    CheckSchedule,
    PauliOperator,
    PauliType,
    StabilizerCheck,
)
from pecos.qec.protocols import (
    InnerCodeGeometry,
    MSDProtocol,
    OuterCodeGeometry,
    create_msd_protocol,
)
from pecos.qec.surface import (
    LogicalOperator,
    PatchGeometry,
    PatchOrientation,
    Stabilizer,
    StabilizerSupport,
    SurfacePatch,
    SurfacePatchBuilder,
    build_memory_circuit,
    compute_x_stabilizer_supports,
    compute_z_stabilizer_supports,
    generate_nonrotated_surface_layout,
    generate_surface_layout,
    parity_matrix_x,
    parity_matrix_z,
    surface_code_memory,
)

__all__ = [
    # Submodules
    "analysis",
    "color",
    "protocols",
    "surface",
    # DEM generation and analysis
    "DagFaultAnalyzer",
    "DagFaultInfluenceMap",
    "DemBuilder",
    "DemSampler",
    "DemSamplerBuilder",
    "DetectorErrorModel",
    "EquivalenceResult",
    "FaultLocation",
    "InfluenceBuilder",
    "ParsedDem",
    "assert_dems_equivalent",
    "compare_dems_exact",
    "compare_dems_statistical",
    "verify_dem_equivalence",
    # Pauli constants
    "PAULI_I",
    "PAULI_X",
    "PAULI_Y",
    "PAULI_Z",
    # Analysis utilities
    "build_adaptive_dem",
    "compare_flip_matrices",
    "compare_k_body_rates",
    "detector_flip_matrices_by_round",
    "detector_flip_matrix",
    "detector_k_body_rates",
    "detector_k_body_rates_by_round",
    "empirical_correlation_table",
    "fit_dem_from_simulation",
    "logical_error_rate",
    "logical_fidelity",
    "logical_from_data",
    "logical_x_from_data",
    "logical_z_from_data",
    "lower_bound_fidelity",
    "syndrome_difference",
    "syndrome_to_detection_events",
    # Generic
    "CheckSchedule",
    "PauliOperator",
    "PauliType",
    "StabilizerCheck",
    # Protocols - MSD
    "InnerCodeGeometry",
    "MSDProtocol",
    "OuterCodeGeometry",
    "create_msd_protocol",
    # Surface code - rotated (most common, default)
    "generate_surface_layout",
    # Surface code - non-rotated
    "compute_x_stabilizer_supports",
    "compute_z_stabilizer_supports",
    "generate_nonrotated_surface_layout",
    "parity_matrix_x",
    "parity_matrix_z",
    # Surface code - patch classes
    "LogicalOperator",
    "PatchGeometry",
    "PatchOrientation",
    "Stabilizer",
    "StabilizerSupport",
    "SurfacePatch",
    "SurfacePatchBuilder",
    "build_memory_circuit",
    "surface_code_memory",
    # Color code
    "ColorCode488",
    "ColorCode488Builder",
    "ColorCode488Geometry",
    "ColorCodeStabilizer",
    "generate_488_layout",
]
