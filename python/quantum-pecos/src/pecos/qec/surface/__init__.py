# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Surface code geometry and parity matrices.

This module provides pure geometry computations for surface codes,
including stabilizer support calculation and parity check matrices.

Layouts:
    square_lattice: Standard square lattice (d^2 data qubits)
    rotated_lattice: Rotated lattice (more efficient)

Functions:
    compute_x_stabilizer_supports: Get X stabilizer qubit indices
    compute_z_stabilizer_supports: Get Z stabilizer qubit indices
    parity_matrix_x: Generate X parity check matrix
    parity_matrix_z: Generate Z parity check matrix
"""

# Circuit generation from geometry (unified abstraction)
from pecos.qec.surface.circuit_builder import (
    DagCircuitRenderer,
    GuppyRenderer,
    OpType,
    QubitAllocation,
    StimRenderer,
    SurfaceCircuitStep,
    TickCircuitRenderer,
    build_surface_code_circuit,
    classify_stabilizer_boundary,
    describe_surface_memory_experiment,
    generate_dag_circuit_from_patch,
    generate_dem_from_tick_circuit,
    generate_dem_from_tick_circuit_via_autodetection,
    generate_dem_from_tick_circuit_via_pauli_frame,
    generate_dem_from_tick_circuit_via_stim,
    generate_guppy_from_patch,
    generate_stim_from_patch,
    generate_tick_circuit_from_patch,
    get_detector_descriptors_from_tick_circuit,
    get_measurement_order_from_tick_circuit,
    get_observable_descriptors_from_tick_circuit,
    get_stabilizer_region,
    get_stabilizer_schedule_entries,
    get_stabilizer_schedule_metadata,
    get_stabilizer_touch_label,
    tick_circuit_to_stim,
)
from pecos.qec.surface.circuit_builder import (
    generate_dem_from_patch as generate_dem_from_patch_stim,
)
from pecos.qec.surface.decode import (
    DecoderType,
    DecodingResult,
    NativeSampler,
    NoiseModel,
    SimulationResult,
    SurfaceDecoder,
    build_memory_circuit,
    build_native_sampler,
    build_stim_circuit_from_patch,
    generate_circuit_level_dem,
    generate_dem_from_patch,
    generate_repetition_code_dem,
    generate_surface_code_dem,
    run_noisy_memory_experiment,
    surface_code_memory,
    syndromes_to_detection_events,
)
from pecos.qec.surface.layouts import (
    StabilizerSupport,
    compute_rotated_x_stabilizers,
    compute_rotated_z_stabilizers,
    compute_x_stabilizer_supports,
    compute_z_stabilizer_supports,
    generate_nonrotated_surface_layout,
    generate_surface_layout,
    get_rotated_logical_x,
    get_rotated_logical_z,
)
from pecos.qec.surface.logical_circuit import (
    LogicalCircuitBuilder,
    LogicalGateType,
    LogicalOp,
    PatchState,
)
from pecos.qec.surface.parity import (
    parity_matrix_x,
    parity_matrix_z,
)
from pecos.qec.surface.patch import (
    LogicalDescriptor,
    LogicalOperator,
    PatchGeometry,
    PatchOrientation,
    Stabilizer,
    StabilizerDescriptor,
    StabilizerScheduleEntry,
    SurfacePatch,
    SurfacePatchBuilder,
    SurfacePatchDescriptor,
)
from pecos.qec.surface.plot import plot_patch, plot_surface_code
from pecos.qec.surface.schedule import (
    compute_cnot_schedule,
    get_stab_schedule,
)

__all__ = [
    # Rotated lattice (most common, default)
    "compute_rotated_x_stabilizers",
    "compute_rotated_z_stabilizers",
    "generate_surface_layout",
    "get_rotated_logical_x",
    "get_rotated_logical_z",
    # Non-rotated lattice
    "StabilizerSupport",
    "compute_x_stabilizer_supports",
    "compute_z_stabilizer_supports",
    "generate_nonrotated_surface_layout",
    # CNOT schedule
    "compute_cnot_schedule",
    "get_stab_schedule",
    # Parity matrices
    "parity_matrix_x",
    "parity_matrix_z",
    # Patch classes
    "LogicalOperator",
    "LogicalDescriptor",
    "PatchGeometry",
    "PatchOrientation",
    "Stabilizer",
    "StabilizerDescriptor",
    "StabilizerScheduleEntry",
    "SurfacePatchDescriptor",
    "SurfacePatch",
    "SurfacePatchBuilder",
    # Decoding
    "DecoderType",
    "DecodingResult",
    "NativeSampler",
    "NoiseModel",
    "SimulationResult",
    "SurfaceDecoder",
    "build_memory_circuit",
    "build_native_sampler",
    "build_stim_circuit_from_patch",
    "generate_circuit_level_dem",
    "generate_dem_from_patch",
    "generate_repetition_code_dem",
    "generate_surface_code_dem",
    "run_noisy_memory_experiment",
    "surface_code_memory",
    "syndromes_to_detection_events",
    # Visualization
    "plot_patch",
    "plot_surface_code",
    # Circuit generation (unified abstraction)
    "SurfaceCircuitStep",
    "DagCircuitRenderer",
    "GuppyRenderer",
    "OpType",
    "QubitAllocation",
    "StimRenderer",
    "TickCircuitRenderer",
    "build_surface_code_circuit",
    "classify_stabilizer_boundary",
    "describe_surface_memory_experiment",
    "generate_dag_circuit_from_patch",
    "generate_dem_from_tick_circuit",
    "generate_dem_from_tick_circuit_via_autodetection",
    "generate_dem_from_tick_circuit_via_pauli_frame",
    "generate_dem_from_tick_circuit_via_stim",
    "generate_guppy_from_patch",
    "generate_stim_from_patch",
    "generate_tick_circuit_from_patch",
    "get_detector_descriptors_from_tick_circuit",
    "get_measurement_order_from_tick_circuit",
    "get_observable_descriptors_from_tick_circuit",
    "get_stabilizer_region",
    "get_stabilizer_schedule_entries",
    "get_stabilizer_schedule_metadata",
    "get_stabilizer_touch_label",
    # Logical circuit builder (transversal gates)
    "LogicalCircuitBuilder",
    "LogicalGateType",
    "LogicalOp",
    "PatchState",
    "tick_circuit_to_stim",
]
