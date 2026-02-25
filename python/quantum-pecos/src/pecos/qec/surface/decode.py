# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Decoding for surface codes using various Rust-wrapped decoders.

This module provides decoders for surface code memory experiments,
supporting multiple decoder backends:

MWPM Decoders (space-time matching):
- PyMatching: Fast C++ MWPM (default)
- FusionBlossom: Pure Rust MWPM

LDPC Decoders (belief propagation):
- BP+OSD: Belief Propagation with Ordered Statistics Decoding
- BP+LSD: Belief Propagation with Localized Statistics Decoding
- UnionFind: Cluster-based decoder

Search-based Decoders:
- Tesseract: A* search with pruning heuristics (requires DEM)

DEM Generation:
The default DEM generation uses PECOS native fault propagation via Rust:
- TickCircuit -> DagCircuit -> DagFaultAnalyzer -> DemBuilder
- Same CNOT schedule as Guppy code
- Proper circuit-level error propagation through gates
- No external dependencies (pure PECOS pipeline)

- generate_circuit_level_dem_from_builder: Circuit-level DEM via native PECOS
  - Uses DagFaultAnalyzer for backward fault propagation
  - DemBuilder constructs DEM with proper probability combination
  - Matches the circuits actually executed via Selene

- generate_surface_code_dem: Phenomenological noise model (code-capacity style)
  - One error per data qubit per round
  - Simple measurement errors
  - Fast but doesn't model circuit-level error propagation

For circuit-level decoding with MWPM:
1. Raw syndromes are converted to detection events (differences between rounds)
2. A space-time matching graph connects detectors across rounds
3. The decoder finds minimum-weight corrections
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    import stim
    from numpy.typing import NDArray
    from pecos_rslib.qec import MeasurementNoiseModel

    from pecos.qec.surface.patch import Stabilizer, SurfacePatch


class DecoderType(str, Enum):
    """Available decoder backends."""

    PYMATCHING = "pymatching"
    FUSION_BLOSSOM = "fusion_blossom"
    BP_OSD = "bp_osd"
    BP_LSD = "bp_lsd"
    UNION_FIND = "union_find"
    TESSERACT = "tesseract"


@dataclass
class NoiseModel:
    """Depolarizing noise parameters for surface code simulation.

    These parameters match the DepolarizingErrorModel in selene_sim.

    Attributes:
        p1: Single-qubit gate error rate
        p2: Two-qubit gate error rate
        p_meas: Measurement error rate
        p_init: Initialization error rate
    """

    p1: float = 0.0  # Single-qubit gate error rate
    p2: float = 0.0  # Two-qubit gate error rate
    p_meas: float = 0.0  # Measurement error rate
    p_init: float = 0.0  # Initialization error rate

    @property
    def is_noiseless(self) -> bool:
        """True if all error rates are zero."""
        return self.p1 == 0.0 and self.p2 == 0.0 and self.p_meas == 0.0 and self.p_init == 0.0

    @property
    def physical_error_rate(self) -> float:
        """Approximate combined physical error rate (for DEM weights)."""
        # Use maximum as a conservative estimate
        return max(self.p1, self.p2, self.p_meas, self.p_init)


@dataclass
class DecodingResult:
    """Result from decoding a single shot."""

    x_correction: NDArray[np.uint8]  # X corrections to apply to data qubits
    z_correction: NDArray[np.uint8]  # Z corrections to apply to data qubits
    logical_x_flip: bool  # True if logical X was flipped by correction
    logical_z_flip: bool  # True if logical Z was flipped by correction
    decoding_weight: float  # Weight of the matching solution


def syndromes_to_detection_events(
    syndromes: NDArray[np.uint8],
    num_rounds: int,
    num_detectors_per_round: int,
) -> NDArray[np.uint8]:
    """Convert raw syndromes to detection events.

    Detection events are the XOR between consecutive syndrome rounds.
    For circuit-level noise, this is required because measurement errors
    flip syndromes in both the current and next round.

    Args:
        syndromes: Raw syndrome array of shape (num_rounds, num_detectors_per_round)
                   or flat array of length num_rounds * num_detectors_per_round
        num_rounds: Number of syndrome extraction rounds
        num_detectors_per_round: Number of detectors per round

    Returns:
        Detection events array of shape (num_rounds, num_detectors_per_round)
    """
    # Reshape to (rounds, detectors) if flat
    if syndromes.ndim == 1:
        syndromes = syndromes.reshape(num_rounds, num_detectors_per_round)

    # First round: compare to expected zero syndrome
    events = np.zeros_like(syndromes)
    events[0] = syndromes[0]

    # Subsequent rounds: XOR with previous round
    for r in range(1, num_rounds):
        events[r] = syndromes[r] ^ syndromes[r - 1]

    return events


def generate_repetition_code_dem(
    num_checks: int,
    num_rounds: int,
    p_data: float = 0.01,
    p_meas: float = 0.01,
) -> str:
    """Generate a DEM for a repetition code (for testing).

    Args:
        num_checks: Number of parity checks (distance - 1)
        num_rounds: Number of syndrome rounds
        p_data: Data qubit error probability
        p_meas: Measurement error probability

    Returns:
        DEM string for PyMatching
    """
    lines = []
    lines.append("# Repetition code DEM")
    lines.append(f"# num_checks={num_checks}, num_rounds={num_rounds}")
    lines.append("")

    # Detector indices: round * num_checks + check_index
    def det_id(round_: int, check: int) -> int:
        return round_ * num_checks + check

    # Spacelike edges (data qubit errors)
    for r in range(num_rounds):
        # First boundary
        lines.append(f"error({p_data:.6f}) D{det_id(r, 0)} L0")

        # Internal edges
        lines.extend(f"error({p_data:.6f}) D{det_id(r, c)} D{det_id(r, c + 1)}" for c in range(num_checks - 1))

        # Last boundary
        lines.append(f"error({p_data:.6f}) D{det_id(r, num_checks - 1)} L0")

    # Timelike edges (measurement errors)
    if num_rounds > 1:
        lines.extend(
            f"error({p_meas:.6f}) D{det_id(r, c)} D{det_id(r + 1, c)}"
            for r in range(num_rounds - 1)
            for c in range(num_checks)
        )

    # Detector coordinates
    lines.extend(f"detector({c}, 0, {r}) D{det_id(r, c)}" for r in range(num_rounds) for c in range(num_checks))

    lines.append("logical_observable L0")

    return "\n".join(lines)


def generate_surface_code_dem(
    patch: SurfacePatch,
    num_rounds: int,
    noise: NoiseModel,
    stab_type: str = "Z",
) -> str:
    """Generate a phenomenological DEM for surface code decoding.

    This creates a simplified "code-capacity" style noise model with:
    - One error mechanism per data qubit per round (spacelike edges)
    - One measurement error per stabilizer between rounds (timelike edges)
    - Boundary edges for logical operator detection

    NOTE: This is a phenomenological model that does NOT account for:
    - Error propagation through CNOT gates
    - Hook errors from the syndrome extraction circuit
    - Correlated errors from multi-qubit gates

    For circuit-level noise modeling, use generate_circuit_level_dem() instead.

    Args:
        patch: Surface code patch with geometry
        num_rounds: Number of syndrome extraction rounds
        noise: Noise model parameters (p2 used for data errors, p_meas for measurement)
        stab_type: Which stabilizer type to decode ('X' or 'Z')
                   X stabilizers detect Z errors, Z stabilizers detect X errors

    Returns:
        DEM string in Stim format
    """
    geom = patch.geometry
    lines = []

    # Get stabilizers based on type
    # For Z-basis memory: Z stabilizers detect X errors, which flip Z measurements
    # The logical observable is the logical Z parity (sum of Z measurements on logical_Z qubits)
    # So X errors on logical_Z qubits flip both the stabilizers AND the logical observable
    #
    # For X-basis memory: X stabilizers detect Z errors, which flip X measurements
    # The logical observable is the logical X parity
    # So Z errors on logical_X qubits flip both the stabilizers AND the logical observable
    if stab_type == "X":
        stabilizers = geom.x_stabilizers
        logical_op = geom.logical_x  # X checks detect Z errors; Z errors on logical_X flip logical
    else:
        stabilizers = geom.z_stabilizers
        logical_op = geom.logical_z  # Z checks detect X errors; X errors on logical_Z flip logical

    num_stab = len(stabilizers)

    # Use noise parameters for error probabilities
    p_data = noise.p2 if noise.p2 > 0 else 0.01  # Default for phenomenological
    p_meas = noise.p_meas if noise.p_meas > 0 else 0.01

    # Detector indices: round * num_stab + stab_index
    def det_id(round_: int, stab_idx: int) -> int:
        return round_ * num_stab + stab_idx

    lines.append(f"# Surface code d={patch.distance} {stab_type}-stabilizer DEM")
    lines.append(f"# rounds={num_rounds}, p_data={p_data:.4f}, p_meas={p_meas:.4f}")
    lines.append("")

    # Build adjacency: which stabilizers share data qubits
    stab_to_data: dict[int, set[int]] = {}
    data_to_stabs: dict[int, list[int]] = {}

    for stab in stabilizers:
        stab_to_data[stab.index] = set(stab.data_qubits)
        for dq in stab.data_qubits:
            if dq not in data_to_stabs:
                data_to_stabs[dq] = []
            data_to_stabs[dq].append(stab.index)

    # Track logical operator data qubits
    logical_qubits = set(logical_op.data_qubits) if logical_op else set()

    # For each round, add spacelike edges
    for r in range(num_rounds):
        # Data qubit errors create edges between adjacent stabilizers
        for dq, stab_indices in data_to_stabs.items():
            affects_logical = dq in logical_qubits

            if len(stab_indices) == 1:
                # Boundary data qubit - edge to boundary
                stab_idx = stab_indices[0]
                if affects_logical:
                    lines.append(f"error({p_data:.6f}) D{det_id(r, stab_idx)} L0")
                else:
                    lines.append(f"error({p_data:.6f}) D{det_id(r, stab_idx)}")
            elif len(stab_indices) == 2:
                # Internal data qubit - edge between two stabilizers
                s1, s2 = stab_indices
                if affects_logical:
                    lines.append(
                        f"error({p_data:.6f}) D{det_id(r, s1)} D{det_id(r, s2)} L0",
                    )
                else:
                    lines.append(
                        f"error({p_data:.6f}) D{det_id(r, s1)} D{det_id(r, s2)}",
                    )

    # Timelike edges (measurement errors)
    # For multi-round: measurement errors create edges between same stabilizer in consecutive rounds
    # For single-round: measurement errors are boundary edges (flip one detector)
    if num_rounds > 1:
        lines.extend(
            f"error({p_meas:.6f}) D{det_id(r, stab.index)} D{det_id(r + 1, stab.index)}"
            for r in range(num_rounds - 1)
            for stab in stabilizers
        )
    else:
        # Single round: measurement errors are boundary edges
        lines.extend(f"error({p_meas:.6f}) D{det_id(0, stab.index)}" for stab in stabilizers)

    # Detector coordinates (x, y, t)
    # Use stabilizer index as spatial coordinate
    lines.extend(
        f"detector({stab.index}, 0, {r}) D{det_id(r, stab.index)}" for r in range(num_rounds) for stab in stabilizers
    )

    lines.append("logical_observable L0")

    return "\n".join(lines)


def generate_circuit_level_dem_from_builder(
    patch: SurfacePatch,
    num_rounds: int,
    noise: NoiseModel,
    basis: str = "Z",
) -> str:
    """Generate circuit-level DEM using PECOS native fault propagation.

    This is the preferred method for DEM generation. It uses:
    - TickCircuit generated with same CNOT schedule as Guppy code
    - DagFaultAnalyzer for Rust-based backward fault propagation
    - DemBuilder to construct the detector error model

    This ensures the DEM exactly matches the circuit that would be executed
    via the Guppy -> HUGR -> Selene pipeline, using native PECOS analysis
    without external dependencies.

    Args:
        patch: Surface code patch with geometry
        num_rounds: Number of syndrome extraction rounds
        noise: Noise model parameters
        basis: Memory basis ('X' or 'Z')

    Returns:
        DEM string in standard format

    Example:
        >>> from pecos.qec.surface import SurfacePatch, NoiseModel
        >>> from pecos.qec.surface.decode import generate_circuit_level_dem_from_builder
        >>> patch = SurfacePatch.create(distance=3)
        >>> noise = NoiseModel(p1=0.001, p2=0.01, p_meas=0.01)
        >>> dem = generate_circuit_level_dem_from_builder(patch, num_rounds=3, noise=noise)
    """
    from pecos.qec import DagFaultAnalyzer, DemBuilder
    from pecos.qec.surface.circuit_builder import (
        _extract_measurement_order,
        generate_tick_circuit_from_patch,
    )

    # Generate TickCircuit (source of truth for circuit structure)
    tc = generate_tick_circuit_from_patch(patch, num_rounds, basis)

    # Convert to DAG and build influence map via Rust fault propagation
    dag = tc.to_dag_circuit()
    analyzer = DagFaultAnalyzer(dag)
    influence_map = analyzer.build_influence_map()

    # Extract metadata from TickCircuit
    detectors_json = tc.get_meta("detectors")
    observables_json = tc.get_meta("observables")
    num_measurements = int(tc.get_meta("num_measurements") or "0")
    measurement_order = _extract_measurement_order(tc)

    # Build DEM using native PECOS builder
    builder = DemBuilder(influence_map)
    builder.with_noise(noise.p1, noise.p2, noise.p_meas, noise.p_init)
    builder.with_num_measurements(num_measurements)
    builder.with_measurement_order(measurement_order)
    builder.with_detectors_json(detectors_json)
    if observables_json:
        builder.with_observables_json(observables_json)

    dem = builder.build()
    return dem.to_string()


def generate_circuit_level_dem(
    distance: int,
    num_rounds: int,
    noise: NoiseModel,
    basis: str = "Z",
) -> str:
    """Generate a circuit-level DEM using Stim's surface code generator.

    This generates a proper circuit-level noise model that accounts for:
    - Error propagation through CNOT gates
    - Hook errors from the measurement circuit
    - Correlated errors from multi-qubit gates
    - Idle errors during the syndrome extraction rounds

    Uses Stim's built-in rotated surface code circuit generator, which has
    a similar structure to the Guppy-generated circuits (4-round CNOT schedule).

    Args:
        distance: Code distance (must be odd, >= 3)
        num_rounds: Number of syndrome extraction rounds
        noise: Noise model parameters
        basis: Memory basis ('X' or 'Z')

    Returns:
        DEM string in Stim format

    Example:
        >>> from pecos.qec.surface import generate_circuit_level_dem, NoiseModel
        >>> noise = NoiseModel(p1=0.001, p2=0.01, p_meas=0.01)
        >>> dem = generate_circuit_level_dem(distance=3, num_rounds=3, noise=noise, basis="Z")
    """
    import stim

    # Map basis to Stim's circuit type
    circuit_type = "surface_code:rotated_memory_x" if basis.upper() == "X" else "surface_code:rotated_memory_z"

    # Generate circuit with noise
    # Stim uses:
    # - after_clifford_depolarization: depolarizing noise after each Clifford gate
    # - before_measure_flip_probability: bit-flip before measurement
    # - after_reset_flip_probability: bit-flip after reset
    circuit = stim.Circuit.generated(
        circuit_type,
        distance=distance,
        rounds=num_rounds,
        after_clifford_depolarization=noise.p2 if noise.p2 > 0 else 0.0,
        before_measure_flip_probability=noise.p_meas if noise.p_meas > 0 else 0.0,
        after_reset_flip_probability=noise.p_init if noise.p_init > 0 else 0.0,
    )

    # Generate DEM from circuit
    dem = circuit.detector_error_model(decompose_errors=True)

    return str(dem)


def build_stim_circuit_from_patch(
    patch: SurfacePatch,
    num_rounds: int,
    noise: NoiseModel | None = None,
    basis: str = "Z",
) -> stim.Circuit:
    """Build a Stim circuit from our patch geometry and CNOT schedule.

    This converts our Guppy-style surface code circuit to Stim format,
    adding proper DETECTOR and OBSERVABLE_INCLUDE annotations.

    The circuit structure matches what Guppy generates:
    - State preparation (R for Z-basis, R+H for X-basis)
    - For each syndrome round:
      - H on X ancillas
      - 4 rounds of CX gates (from compute_cnot_schedule)
      - H on X ancillas
      - Measure ancillas
    - Final data qubit measurement
    - DETECTOR annotations comparing consecutive measurements
    - OBSERVABLE_INCLUDE for logical operator

    Args:
        patch: Surface code patch with geometry
        num_rounds: Number of syndrome extraction rounds
        noise: Optional noise model (if None, noiseless circuit)
        basis: Memory basis ('X' or 'Z')

    Returns:
        stim.Circuit object with DETECTOR and OBSERVABLE_INCLUDE annotations

    Example:
        >>> from pecos.qec.surface import (
        ...     SurfacePatch,
        ...     NoiseModel,
        ...     build_stim_circuit_from_patch,
        ... )
        >>> patch = SurfacePatch.create(distance=3)
        >>> noise = NoiseModel(p2=0.01, p_meas=0.01)
        >>> circuit = build_stim_circuit_from_patch(patch, num_rounds=3, noise=noise)
        >>> dem = circuit.detector_error_model()
    """
    import stim

    from pecos.qec.surface.schedule import compute_cnot_schedule

    geom = patch.geometry
    d = patch.distance
    num_data = geom.num_data
    num_x_anc = len(geom.x_stabilizers)
    num_z_anc = len(geom.z_stabilizers)

    # Qubit layout: [data qubits] [X ancillas] [Z ancillas]
    def data_qubit(idx: int) -> int:
        return idx

    def x_ancilla(stab_idx: int) -> int:
        return num_data + stab_idx

    def z_ancilla(stab_idx: int) -> int:
        return num_data + num_x_anc + stab_idx

    # Compute stabilizer positions from data qubits (center of support)
    def stab_coords(stab: Stabilizer) -> tuple[float, float]:
        """Compute stabilizer coordinates as center of its data qubits."""
        rows = [dq // d for dq in stab.data_qubits]
        cols = [dq % d for dq in stab.data_qubits]
        return (sum(cols) / len(cols), sum(rows) / len(rows))

    # Get CNOT schedule
    cnot_schedule = compute_cnot_schedule(patch)

    # Get logical operator qubits
    if basis.upper() == "Z":
        logical_qubits = list(geom.logical_z.data_qubits) if geom.logical_z else []
    else:
        logical_qubits = list(geom.logical_x.data_qubits) if geom.logical_x else []

    circuit = stim.Circuit()

    # Add qubit coordinates for data qubits
    for i in range(num_data):
        row, col = i // d, i % d
        circuit.append("QUBIT_COORDS", [i], [col, row])

    # Add qubit coordinates for ancillas (at stabilizer centers)
    for stab in geom.x_stabilizers:
        cx, cy = stab_coords(stab)
        circuit.append("QUBIT_COORDS", [x_ancilla(stab.index)], [cx, cy])
    for stab in geom.z_stabilizers:
        cx, cy = stab_coords(stab)
        circuit.append("QUBIT_COORDS", [z_ancilla(stab.index)], [cx, cy])

    # === State Preparation ===
    all_data = list(range(num_data))
    all_x_anc = [x_ancilla(s.index) for s in geom.x_stabilizers]
    all_z_anc = [z_ancilla(s.index) for s in geom.z_stabilizers]
    all_ancillas = all_x_anc + all_z_anc

    # Reset all qubits
    circuit.append("R", all_data + all_ancillas)

    # For X-basis memory, apply H to data qubits
    if basis.upper() == "X":
        circuit.append("TICK")
        circuit.append("H", all_data)
        if noise and noise.p1 > 0:
            circuit.append("DEPOLARIZE1", all_data, noise.p1)

    circuit.append("TICK")

    # === Syndrome Extraction Rounds ===
    for rnd in range(num_rounds):
        # H on X ancillas (before CNOTs)
        circuit.append("H", all_x_anc)
        if noise and noise.p1 > 0:
            circuit.append("DEPOLARIZE1", all_x_anc, noise.p1)
        circuit.append("TICK")

        # 4 rounds of CX gates
        for cx_round in cnot_schedule:
            cx_pairs = []
            for stab_type, stab_idx, data_q in cx_round:
                if stab_type == "X":
                    # X stabilizer: ancilla is control, data is target
                    cx_pairs.extend([x_ancilla(stab_idx), data_qubit(data_q)])
                else:
                    # Z stabilizer: data is control, ancilla is target
                    cx_pairs.extend([data_qubit(data_q), z_ancilla(stab_idx)])

            if cx_pairs:
                circuit.append("CX", cx_pairs)
                if noise and noise.p2 > 0:
                    circuit.append("DEPOLARIZE2", cx_pairs, noise.p2)
            circuit.append("TICK")

        # H on X ancillas (after CNOTs)
        circuit.append("H", all_x_anc)
        if noise and noise.p1 > 0:
            circuit.append("DEPOLARIZE1", all_x_anc, noise.p1)
        circuit.append("TICK")

        # Measure ancillas
        if noise and noise.p_meas > 0:
            circuit.append("X_ERROR", all_ancillas, noise.p_meas)

        # Use MR (measure and reset) for all rounds
        circuit.append("MR", all_ancillas)

        # Add DETECTOR annotations
        # For Z-basis memory: only Z stabilizers are deterministic in round 0
        # For X-basis memory: only X stabilizers are deterministic in round 0
        num_stab = num_x_anc + num_z_anc
        if rnd == 0:
            # First round: only add detectors for stabilizers that are deterministic
            if basis.upper() == "Z":
                # Z-basis: Z stabilizers are deterministic (Z parity of |0⟩ states)
                for i, stab in enumerate(geom.z_stabilizers):
                    cx, cy = stab_coords(stab)
                    circuit.append(
                        "DETECTOR",
                        [stim.target_rec(-num_stab + num_x_anc + i)],
                        [cx, cy, rnd],
                    )
            else:
                # X-basis: X stabilizers are deterministic (X parity of |+⟩ states)
                for i, stab in enumerate(geom.x_stabilizers):
                    cx, cy = stab_coords(stab)
                    circuit.append(
                        "DETECTOR",
                        [stim.target_rec(-num_stab + i)],
                        [cx, cy, rnd],
                    )
        else:
            # Subsequent rounds: XOR with previous round (both X and Z stabilizers)
            for i, stab in enumerate(geom.x_stabilizers):
                cx, cy = stab_coords(stab)
                circuit.append(
                    "DETECTOR",
                    [
                        stim.target_rec(-num_stab + i),
                        stim.target_rec(-2 * num_stab + i),
                    ],
                    [cx, cy, rnd],
                )
            for i, stab in enumerate(geom.z_stabilizers):
                cx, cy = stab_coords(stab)
                circuit.append(
                    "DETECTOR",
                    [
                        stim.target_rec(-num_stab + num_x_anc + i),
                        stim.target_rec(-2 * num_stab + num_x_anc + i),
                    ],
                    [cx, cy, rnd],
                )

        circuit.append("TICK")

    # === Final Data Measurement ===
    if basis.upper() == "X":
        circuit.append("H", all_data)
        if noise and noise.p1 > 0:
            circuit.append("DEPOLARIZE1", all_data, noise.p1)
        circuit.append("TICK")

    if noise and noise.p_meas > 0:
        circuit.append("X_ERROR", all_data, noise.p_meas)

    circuit.append("M", all_data)

    # Final detectors: compare last syndrome to parity of final data measurements
    # For Z-basis memory: Z stabilizers can be reconstructed from final Z measurements
    # For X-basis memory: X stabilizers can be reconstructed from final X measurements
    num_stab = num_x_anc + num_z_anc
    if basis.upper() == "Z":
        # Z stabilizers: check parity of Z measurements matches last syndrome
        for i, stab in enumerate(geom.z_stabilizers):
            cx, cy = stab_coords(stab)
            # Last Z ancilla measurement + final data measurements
            rec_targets = [
                stim.target_rec(-num_data - num_stab + num_x_anc + i),
                *[stim.target_rec(-num_data + dq) for dq in stab.data_qubits],
            ]
            circuit.append("DETECTOR", rec_targets, [cx, cy, num_rounds])
    else:
        # X stabilizers: check parity of X measurements (after H) matches last syndrome
        for i, stab in enumerate(geom.x_stabilizers):
            cx, cy = stab_coords(stab)
            # Last X ancilla measurement + final data measurements
            rec_targets = [
                stim.target_rec(-num_data - num_stab + i),
                *[stim.target_rec(-num_data + dq) for dq in stab.data_qubits],
            ]
            circuit.append("DETECTOR", rec_targets, [cx, cy, num_rounds])

    # OBSERVABLE_INCLUDE: logical operator parity from final measurements
    obs_targets = [stim.target_rec(-num_data + q) for q in logical_qubits]
    circuit.append("OBSERVABLE_INCLUDE", obs_targets, 0)

    return circuit


def generate_dem_from_patch(
    patch: SurfacePatch,
    num_rounds: int,
    noise: NoiseModel,
    basis: str = "Z",
) -> str:
    """Generate a circuit-level DEM from our patch geometry.

    This is the "Guppy → Stim → DEM" route:
    1. Build a Stim circuit matching our Guppy circuit structure
    2. Add noise operations
    3. Use Stim's detector_error_model() to compute the DEM

    Args:
        patch: Surface code patch with geometry
        num_rounds: Number of syndrome extraction rounds
        noise: Noise model parameters
        basis: Memory basis ('X' or 'Z')

    Returns:
        DEM string in Stim format

    Example:
        >>> from pecos.qec.surface import (
        ...     SurfacePatch,
        ...     NoiseModel,
        ...     generate_dem_from_patch,
        ... )
        >>> patch = SurfacePatch.create(distance=3)
        >>> noise = NoiseModel(p2=0.01, p_meas=0.01)
        >>> dem = generate_dem_from_patch(patch, num_rounds=3, noise=noise)
    """
    circuit = build_stim_circuit_from_patch(patch, num_rounds, noise, basis)
    dem = circuit.detector_error_model(decompose_errors=True)
    return str(dem)


class SurfaceDecoder:
    """Decoder for surface codes supporting multiple backends.

    Supports MWPM decoders (PyMatching, FusionBlossom) with space-time matching
    and LDPC decoders (BP+OSD, BP+LSD, UnionFind) with per-qubit error estimation.

    Example:
        >>> from pecos.qec.surface import SurfacePatch, SurfaceDecoder
        >>> patch = SurfacePatch.create(distance=3)
        >>> # Default: PyMatching MWPM
        >>> decoder = SurfaceDecoder(patch, num_rounds=3, noise=NoiseModel(p2=0.01, p_meas=0.01))
        >>> # Alternative: FusionBlossom MWPM
        >>> decoder = SurfaceDecoder(patch, num_rounds=3, decoder_type="fusion_blossom")
        >>> # Alternative: BP+OSD (LDPC)
        >>> decoder = SurfaceDecoder(patch, num_rounds=3, decoder_type="bp_osd")
        >>> is_error, result = decoder.decode_memory_z(synx_list, synz_list, final)
    """

    def __init__(
        self,
        patch: SurfacePatch,
        num_rounds: int = 1,
        noise: NoiseModel | None = None,
        decoder_type: Literal[
            "pymatching",
            "fusion_blossom",
            "bp_osd",
            "bp_lsd",
            "union_find",
            "tesseract",
        ] = "pymatching",
        *,
        use_circuit_level_dem: bool = True,
    ) -> None:
        """Initialize decoder from surface code patch.

        Args:
            patch: Surface code patch with geometry
            num_rounds: Number of syndrome extraction rounds
            noise: Noise model for edge weights (defaults to uniform)
            decoder_type: Decoder backend to use:
                - "pymatching": Fast C++ MWPM decoder (default)
                - "fusion_blossom": Pure Rust MWPM decoder
                - "bp_osd": Belief Propagation + OSD
                - "bp_lsd": Belief Propagation + LSD
                - "union_find": Union-Find decoder
                - "tesseract": A* search-based decoder
            use_circuit_level_dem: If True (default), use circuit-level DEMs from
                our abstracted circuit builder for PyMatching and Tesseract.
                This provides proper error propagation through gates matching
                the actual Guppy/Selene circuits. If False, use phenomenological
                DEMs or check matrices.
        """
        self.patch = patch
        self.num_rounds = num_rounds
        self.noise = noise or NoiseModel(p2=0.01, p_meas=0.01)
        self.decoder_type = DecoderType(decoder_type)
        self.use_circuit_level_dem = use_circuit_level_dem

        # Lazily create decoders
        self._x_decoder = None
        self._z_decoder = None
        self._x_check_matrix = None
        self._z_check_matrix = None
        self._z_dem = None  # DEM string for Z-basis decoding
        self._x_dem = None  # DEM string for X-basis decoding

    def _compute_weight(self, p: float) -> float:
        """Compute MWPM edge weight from error probability."""
        import math

        if p <= 0:
            return 100.0  # Very high weight for impossible errors
        if p >= 1:
            return 0.0  # Zero weight for certain errors
        return -math.log(p / (1 - p))

    def _get_circuit_level_dem(self, basis: str) -> str:
        """Get circuit-level DEM from our abstracted circuit builder.

        Args:
            basis: 'Z' or 'X' basis

        Returns:
            DEM string in Stim format
        """
        return generate_circuit_level_dem_from_builder(
            self.patch,
            self.num_rounds,
            self.noise,
            basis=basis,
        )

    def _get_z_check_matrix(self) -> NDArray[np.uint8]:
        """Get Z stabilizer parity check matrix."""
        if self._z_check_matrix is None:
            geom = self.patch.geometry
            num_stab = len(geom.z_stabilizers)
            num_data = geom.num_data

            # H is standard notation for parity check matrix in coding theory
            H = np.zeros((num_stab, num_data), dtype=np.uint8)
            for stab in geom.z_stabilizers:
                for q in stab.data_qubits:
                    H[stab.index, q] = 1

            self._z_check_matrix = H
        return self._z_check_matrix

    def _get_x_check_matrix(self) -> NDArray[np.uint8]:
        """Get X stabilizer parity check matrix."""
        if self._x_check_matrix is None:
            geom = self.patch.geometry
            num_stab = len(geom.x_stabilizers)
            num_data = geom.num_data

            # H is standard notation for parity check matrix in coding theory
            H = np.zeros((num_stab, num_data), dtype=np.uint8)
            for stab in geom.x_stabilizers:
                for q in stab.data_qubits:
                    H[stab.index, q] = 1

            self._x_check_matrix = H
        return self._x_check_matrix

    def _get_z_decoder(self) -> Any:
        """Get or create decoder for Z-basis memory (decodes Z syndromes for X errors)."""
        if self._z_decoder is None:
            # For PyMatching and Tesseract with circuit-level DEMs, use DEM directly
            if self.use_circuit_level_dem and self.decoder_type in (
                DecoderType.PYMATCHING,
                DecoderType.TESSERACT,
            ):
                self._z_decoder = self._create_decoder_from_dem("Z")
            else:
                self._z_decoder = self._create_decoder(self._get_z_check_matrix())
        return self._z_decoder

    def _create_decoder(self, H: NDArray[np.uint8]) -> Any:
        """Create decoder instance based on decoder_type."""
        num_data = H.shape[1]
        num_stab = H.shape[0]

        # Compute weights from noise model
        p_data = self.noise.p2 if self.noise.p2 > 0 else 0.01
        p_meas = self.noise.p_meas if self.noise.p_meas > 0 else 0.01

        data_weight = self._compute_weight(p_data)
        meas_weight = self._compute_weight(p_meas)

        if self.decoder_type == DecoderType.PYMATCHING:
            from pecos_rslib.decoders import CheckMatrix, PyMatchingDecoder

            weights = [data_weight] * num_data
            check_matrix = CheckMatrix.from_dense(H.tolist()).with_weights(weights)
            timelike_weights = [meas_weight] * num_stab

            return PyMatchingDecoder.from_check_matrix_with_repetitions(
                check_matrix,
                repetitions=self.num_rounds,
                timelike_weights=timelike_weights,
                use_virtual_boundary=True,
            )

        if self.decoder_type == DecoderType.FUSION_BLOSSOM:
            from pecos_rslib.decoders import FusionBlossomDecoder

            # FusionBlossom uses check matrix directly
            # For multi-round, we need to construct the space-time graph manually
            if self.num_rounds == 1:
                weights = [data_weight] * num_data
                return FusionBlossomDecoder.from_check_matrix(
                    H.tolist(),
                    weights=weights,
                    num_observables=num_data,
                )
            # For multi-round, build space-time graph
            return self._create_fusion_blossom_spacetime(H, data_weight, meas_weight)

        if self.decoder_type in (
            DecoderType.BP_OSD,
            DecoderType.BP_LSD,
            DecoderType.UNION_FIND,
        ):
            # LDPC decoders work per-round, not on space-time graph
            return self._create_ldpc_decoder(H, p_data)

        if self.decoder_type == DecoderType.TESSERACT:
            # Tesseract requires a DEM string
            return self._create_tesseract_decoder(H, p_data, p_meas)

        msg = f"Unknown decoder type: {self.decoder_type}"
        raise ValueError(msg)

    def _create_decoder_from_dem(self, basis: str) -> Any:
        """Create decoder from circuit-level DEM.

        Uses our abstracted circuit builder to generate a Stim circuit with
        proper DETECTOR and OBSERVABLE_INCLUDE annotations, then extracts
        the DEM for decoder initialization.

        Args:
            basis: 'Z' or 'X' basis for the memory experiment

        Returns:
            Decoder instance initialized from circuit-level DEM
        """
        # Get circuit-level DEM from our circuit builder
        dem = self._get_circuit_level_dem(basis)

        # Cache the DEM for get_dem() calls
        if basis.upper() == "Z":
            self._z_dem = dem
        else:
            self._x_dem = dem

        if self.decoder_type == DecoderType.PYMATCHING:
            from pecos_rslib.decoders import PyMatchingDecoder

            return PyMatchingDecoder.from_dem(dem)

        if self.decoder_type == DecoderType.TESSERACT:
            from pecos_rslib.decoders import TesseractDecoder

            # Tesseract's remove_zero_probability_errors() doesn't handle
            # DEM_LOGICAL_OBSERVABLE instructions. Filter them out - the
            # observable info is encoded in error edges via L0 references.
            dem_filtered = "\n".join(line for line in dem.split("\n") if not line.startswith("logical_observable"))
            return TesseractDecoder.from_dem(dem_filtered, preset="fast")

        msg = f"Decoder type {self.decoder_type} does not support DEM initialization"
        raise ValueError(msg)

    def _create_fusion_blossom_spacetime(
        self,
        H: NDArray[np.uint8],
        data_weight: float,
        meas_weight: float,
    ) -> Any:
        """Create FusionBlossom decoder with space-time matching graph."""
        from pecos_rslib.decoders import FusionBlossomDecoder

        num_stab = H.shape[0]
        num_data = H.shape[1]
        num_rounds = self.num_rounds

        # Total nodes: num_stab * num_rounds
        total_nodes = num_stab * num_rounds

        decoder = FusionBlossomDecoder(
            num_nodes=total_nodes,
            num_observables=num_data,
        )

        # Build data-to-stabilizer adjacency
        data_to_stabs: dict[int, list[int]] = {}
        for stab_idx in range(num_stab):
            for data_idx in range(num_data):
                if H[stab_idx, data_idx] == 1:
                    if data_idx not in data_to_stabs:
                        data_to_stabs[data_idx] = []
                    data_to_stabs[data_idx].append(stab_idx)

        # Add spacelike edges for each round
        for r in range(num_rounds):
            for data_idx, stab_indices in data_to_stabs.items():
                if len(stab_indices) == 1:
                    # Boundary edge
                    node = r * num_stab + stab_indices[0]
                    decoder.add_boundary_edge(
                        node,
                        observables=[data_idx],
                        weight=data_weight,
                    )
                elif len(stab_indices) == 2:
                    # Internal edge
                    node1 = r * num_stab + stab_indices[0]
                    node2 = r * num_stab + stab_indices[1]
                    decoder.add_edge(
                        node1,
                        node2,
                        observables=[data_idx],
                        weight=data_weight,
                    )

        # Add timelike edges (measurement errors)
        for r in range(num_rounds - 1):
            for stab_idx in range(num_stab):
                node1 = r * num_stab + stab_idx
                node2 = (r + 1) * num_stab + stab_idx
                decoder.add_edge(node1, node2, observables=[], weight=meas_weight)

        return decoder

    def _create_ldpc_decoder(
        self,
        H: NDArray[np.uint8],
        p_data: float,
    ) -> Any:
        """Create LDPC decoder (BP+OSD, BP+LSD, or UnionFind)."""
        from pecos_rslib.decoders import SparseMatrix

        sparse_H = SparseMatrix(H.tolist())

        if self.decoder_type == DecoderType.BP_OSD:
            from pecos_rslib.decoders import BpOsdBuilder

            return (
                BpOsdBuilder(sparse_H, error_rate=p_data)
                .max_iter(100)
                .bp_method("product_sum")
                .osd_method("osd0")
                .osd_order(0)
                .build()
            )

        if self.decoder_type == DecoderType.BP_LSD:
            from pecos_rslib.decoders import BpLsdBuilder

            return BpLsdBuilder(sparse_H, error_rate=p_data).max_iter(100).bp_method("product_sum").lsd_order(0).build()

        if self.decoder_type == DecoderType.UNION_FIND:
            from pecos_rslib.decoders import UnionFindBuilder

            return UnionFindBuilder(sparse_H).method("inversion").build()

        msg = f"Unknown LDPC decoder type: {self.decoder_type}"
        raise ValueError(msg)

    def _create_tesseract_decoder(
        self,
        H: NDArray[np.uint8],
        _p_data: float,
        _p_meas: float,
    ) -> Any:
        """Create Tesseract decoder from check matrix by generating DEM."""
        from pecos_rslib.decoders import TesseractDecoder

        # Determine stabilizer type based on check matrix shape
        z_check = self._get_z_check_matrix()
        stab_type = "Z" if H.shape == z_check.shape and np.array_equal(H, z_check) else "X"

        # Generate DEM using the full surface code DEM generator
        dem = generate_surface_code_dem(
            self.patch,
            self.num_rounds,
            self.noise,
            stab_type,
        )

        # Tesseract's remove_zero_probability_errors() function doesn't handle
        # DEM_LOGICAL_OBSERVABLE instructions - it only supports DEM_ERROR, DEM_DETECTOR,
        # and DEM_SHIFT_DETECTORS. See tesseract/src/common.cc line 104-106.
        # The logical observable info is encoded in the error edges via L0 references,
        # so the standalone 'logical_observable L0' declaration is redundant for Tesseract.
        dem_lines = [line for line in dem.split("\n") if not line.startswith("logical_observable")]
        dem = "\n".join(dem_lines)

        return TesseractDecoder.from_dem(dem, preset="fast")

    def get_dem(self, basis: str = "Z", *, circuit_level: bool | None = None) -> str:
        """Get the Detector Error Model (DEM) string for this decoder configuration.

        This can be used with external decoders or for analysis.

        Args:
            basis: "Z" or "X" basis for the memory experiment
            circuit_level: If True, use circuit-level DEM from our circuit builder.
                          If False, use phenomenological DEM.
                          If None (default), use self.use_circuit_level_dem setting.

        Returns:
            DEM string in Stim format
        """
        use_circuit = circuit_level if circuit_level is not None else self.use_circuit_level_dem

        if use_circuit:
            # Return cached DEM if available
            if basis.upper() == "Z" and self._z_dem is not None:
                return self._z_dem
            if basis.upper() == "X" and self._x_dem is not None:
                return self._x_dem

            # Generate circuit-level DEM from our circuit builder
            return self._get_circuit_level_dem(basis)

        # Phenomenological DEM (backward compatible)
        # Map basis to stabilizer type for phenomenological model:
        # Z-basis memory -> Z stabilizers detect X errors
        # X-basis memory -> X stabilizers detect Z errors
        stab_type = basis.upper()
        return generate_surface_code_dem(
            self.patch,
            self.num_rounds,
            self.noise,
            stab_type,
        )

    def _get_x_decoder(self) -> Any:
        """Get or create decoder for X-basis memory (decodes X syndromes for Z errors)."""
        if self._x_decoder is None:
            # For PyMatching and Tesseract with circuit-level DEMs, use DEM directly
            if self.use_circuit_level_dem and self.decoder_type in (
                DecoderType.PYMATCHING,
                DecoderType.TESSERACT,
            ):
                self._x_decoder = self._create_decoder_from_dem("X")
            else:
                self._x_decoder = self._create_decoder(self._get_x_check_matrix())
        return self._x_decoder

    def _is_mwpm_decoder(self) -> bool:
        """Check if using an MWPM or Tesseract decoder (vs LDPC)."""
        return self.decoder_type in (
            DecoderType.PYMATCHING,
            DecoderType.FUSION_BLOSSOM,
            DecoderType.TESSERACT,
        )

    def decode_z_syndrome(
        self,
        detection_events: NDArray[np.uint8],
        raw_syndrome: NDArray[np.uint8] | None = None,
    ) -> tuple[NDArray[np.uint8], float]:
        """Decode Z stabilizer syndrome to get X corrections.

        For MWPM decoders: uses detection_events (differences between rounds)
        For LDPC decoders: uses raw_syndrome (last round or combined)

        Args:
            detection_events: Detection events array (flat or 2D) for MWPM
            raw_syndrome: Raw syndrome for LDPC decoders (optional)

        Returns:
            (x_correction, weight) - correction is per-qubit
        """
        decoder = self._get_z_decoder()

        if self._is_mwpm_decoder():
            # MWPM/Tesseract: use detection events
            events_flat = detection_events.ravel().astype(np.uint8)

            if self.decoder_type == DecoderType.TESSERACT:
                # Tesseract takes sparse detection indices
                detection_indices = [i for i, v in enumerate(events_flat) if v != 0]
                result = decoder.decode(detection_indices)
                # Tesseract returns observables_mask, not per-qubit correction
                # We return a dummy correction and encode logical flip in first element
                num_data = self._get_z_check_matrix().shape[1]
                correction = np.zeros(num_data, dtype=np.uint8)
                if result.observables_mask & 1:  # L0 flipped
                    correction[0] = 1  # Mark that logical was predicted flipped
                weight = result.cost
            else:
                result = decoder.decode(events_flat.tolist())

                # For FusionBlossom, need to clear state for next decode
                if self.decoder_type == DecoderType.FUSION_BLOSSOM:
                    decoder.clear()

                correction = np.array(result.correction, dtype=np.uint8)
                weight = result.weight
        else:
            # LDPC: use raw syndrome (last round)
            if raw_syndrome is None:
                # Use last round of detection events as approximation
                num_stab = self._get_z_check_matrix().shape[0]
                if detection_events.size >= num_stab:
                    raw_syndrome = detection_events.ravel()[-num_stab:]
                else:
                    raw_syndrome = detection_events.ravel()

            result = decoder.decode(raw_syndrome.astype(np.uint8).tolist())
            correction = np.array(result.decoding, dtype=np.uint8)
            weight = 0.0 if result.converged else 1.0  # LDPC doesn't have weight

        return correction, weight

    def decode_x_syndrome(
        self,
        detection_events: NDArray[np.uint8],
        raw_syndrome: NDArray[np.uint8] | None = None,
    ) -> tuple[NDArray[np.uint8], float]:
        """Decode X stabilizer syndrome to get Z corrections.

        For MWPM decoders: uses detection_events (differences between rounds)
        For LDPC decoders: uses raw_syndrome (last round or combined)

        Args:
            detection_events: Detection events array (flat or 2D) for MWPM
            raw_syndrome: Raw syndrome for LDPC decoders (optional)

        Returns:
            (z_correction, weight) - correction is per-qubit
        """
        decoder = self._get_x_decoder()

        if self._is_mwpm_decoder():
            # MWPM/Tesseract: use detection events
            events_flat = detection_events.ravel().astype(np.uint8)

            if self.decoder_type == DecoderType.TESSERACT:
                # Tesseract takes sparse detection indices
                detection_indices = [i for i, v in enumerate(events_flat) if v != 0]
                result = decoder.decode(detection_indices)
                # Tesseract returns observables_mask, not per-qubit correction
                num_data = self._get_x_check_matrix().shape[1]
                correction = np.zeros(num_data, dtype=np.uint8)
                if result.observables_mask & 1:  # L0 flipped
                    correction[0] = 1  # Mark that logical was predicted flipped
                weight = result.cost
            else:
                result = decoder.decode(events_flat.tolist())

                # For FusionBlossom, need to clear state for next decode
                if self.decoder_type == DecoderType.FUSION_BLOSSOM:
                    decoder.clear()

                correction = np.array(result.correction, dtype=np.uint8)
                weight = result.weight
        else:
            # LDPC: use raw syndrome (last round)
            if raw_syndrome is None:
                # Use last round of detection events as approximation
                num_stab = self._get_x_check_matrix().shape[0]
                if detection_events.size >= num_stab:
                    raw_syndrome = detection_events.ravel()[-num_stab:]
                else:
                    raw_syndrome = detection_events.ravel()

            result = decoder.decode(raw_syndrome.astype(np.uint8).tolist())
            correction = np.array(result.decoding, dtype=np.uint8)
            weight = 0.0 if result.converged else 1.0  # LDPC doesn't have weight

        return correction, weight

    def decode_memory_z(
        self,
        _synx_list: list[NDArray[np.uint8]],
        synz_list: list[NDArray[np.uint8]],
        final: NDArray[np.uint8],
    ) -> tuple[bool, DecodingResult]:
        """Decode a Z-basis memory experiment.

        For Z-basis memory:
        - Z stabilizers detect X errors (which flip Z measurements)
        - We decode Z syndromes to find X corrections
        - Apply corrections to final measurements to get corrected logical Z parity

        Args:
            synx_list: List of X syndrome arrays, one per round
            synz_list: List of Z syndrome arrays, one per round
            final: Final data qubit measurements

        Returns:
            (is_logical_error, decoding_result)
        """
        geom = self.patch.geometry
        num_z_stab = len(geom.z_stabilizers)

        # Stack syndromes into 2D array
        synz = np.array(synz_list, dtype=np.uint8)

        # Convert to detection events
        events = syndromes_to_detection_events(synz, self.num_rounds, num_z_stab)

        # Get raw syndrome (last round) for LDPC decoders
        raw_syn = synz[-1] if len(synz_list) > 0 else None

        # Decode to get per-qubit X corrections
        x_correction, weight = self.decode_z_syndrome(events, raw_syndrome=raw_syn)

        # Compute logical Z parity from corrected final measurements
        # X corrections flip Z measurements, so we XOR
        logical_z_qubits = geom.logical_z.data_qubits if geom.logical_z else ()

        # Compute parity of final measurements on logical Z qubits
        final_parity = sum(final[q] for q in logical_z_qubits) % 2

        if self.decoder_type == DecoderType.TESSERACT:
            # Tesseract returns logical prediction directly in correction[0]
            logical_prediction = x_correction[0] if len(x_correction) > 0 else 0
            # Logical error if raw parity XOR prediction != 0
            corrected_parity = (final_parity + logical_prediction) % 2
            correction_parity = int(logical_prediction)
        else:
            # Compute parity of corrections on logical Z qubits
            # (corrections flip the measurement values)
            if len(x_correction) >= self.patch.num_data:
                correction_parity = sum(x_correction[q] for q in logical_z_qubits) % 2
            else:
                correction_parity = 0

            # Corrected parity: XOR of final measurements and corrections
            corrected_parity = (final_parity + correction_parity) % 2

        # Logical error if corrected parity is not 0 (expected for |0_L>)
        is_logical_error = corrected_parity != 0

        # Compute logical X flip (did we apply an odd number of corrections on logical Z?)
        logical_x_flip = correction_parity != 0

        result = DecodingResult(
            x_correction=(
                x_correction
                if len(x_correction) == self.patch.num_data
                else np.zeros(self.patch.num_data, dtype=np.uint8)
            ),
            z_correction=np.zeros(self.patch.num_data, dtype=np.uint8),
            logical_x_flip=logical_x_flip,
            logical_z_flip=False,
            decoding_weight=weight,
        )

        return is_logical_error, result

    def decode_memory_x(
        self,
        synx_list: list[NDArray[np.uint8]],
        _synz_list: list[NDArray[np.uint8]],
        final: NDArray[np.uint8],
    ) -> tuple[bool, DecodingResult]:
        """Decode an X-basis memory experiment.

        For X-basis memory:
        - X stabilizers detect Z errors (which flip X measurements)
        - We decode X syndromes to find Z corrections
        - Apply corrections to final measurements to get corrected logical X parity

        Args:
            synx_list: List of X syndrome arrays, one per round
            synz_list: List of Z syndrome arrays, one per round
            final: Final data qubit measurements

        Returns:
            (is_logical_error, decoding_result)
        """
        geom = self.patch.geometry
        num_x_stab = len(geom.x_stabilizers)

        # Stack syndromes into 2D array
        synx = np.array(synx_list, dtype=np.uint8)

        # Convert to detection events
        events = syndromes_to_detection_events(synx, self.num_rounds, num_x_stab)

        # Get raw syndrome (last round) for LDPC decoders
        raw_syn = synx[-1] if len(synx_list) > 0 else None

        # Decode to get per-qubit Z corrections
        z_correction, weight = self.decode_x_syndrome(events, raw_syndrome=raw_syn)

        # Compute logical X parity from corrected final measurements
        # Z corrections flip X measurements, so we XOR
        logical_x_qubits = geom.logical_x.data_qubits if geom.logical_x else ()

        # Compute parity of final measurements on logical X qubits
        final_parity = sum(final[q] for q in logical_x_qubits) % 2

        if self.decoder_type == DecoderType.TESSERACT:
            # Tesseract returns logical prediction directly in correction[0]
            logical_prediction = z_correction[0] if len(z_correction) > 0 else 0
            # Logical error if raw parity XOR prediction != 0
            corrected_parity = (final_parity + logical_prediction) % 2
            correction_parity = int(logical_prediction)
        else:
            # Compute parity of corrections on logical X qubits
            if len(z_correction) >= self.patch.num_data:
                correction_parity = sum(z_correction[q] for q in logical_x_qubits) % 2
            else:
                correction_parity = 0

            # Corrected parity: XOR of final measurements and corrections
            corrected_parity = (final_parity + correction_parity) % 2

        # Logical error if corrected parity is not 0 (expected for |+_L>)
        is_logical_error = corrected_parity != 0

        # Compute logical Z flip
        logical_z_flip = correction_parity != 0

        result = DecodingResult(
            x_correction=np.zeros(self.patch.num_data, dtype=np.uint8),
            z_correction=(
                z_correction
                if len(z_correction) == self.patch.num_data
                else np.zeros(self.patch.num_data, dtype=np.uint8)
            ),
            logical_x_flip=False,
            logical_z_flip=logical_z_flip,
            decoding_weight=weight,
        )

        return is_logical_error, result


@dataclass
class SimulationResult:
    """Results from a noisy memory experiment.

    Attributes:
        distance: Code distance
        num_shots: Number of shots run
        num_rounds: Number of syndrome extraction rounds
        basis: Memory basis ('Z' or 'X')
        num_logical_errors: Number of logical errors after decoding
        num_raw_errors: Number of raw errors (before decoding)
        logical_error_rate: Decoded logical error rate
        raw_error_rate: Raw error rate (no decoding)
        decoded: Whether decoding was applied
        decoder_type: Decoder backend used (if decoded)
    """

    distance: int
    num_shots: int
    num_rounds: int
    basis: str
    num_logical_errors: int
    num_raw_errors: int
    logical_error_rate: float
    raw_error_rate: float
    decoded: bool
    decoder_type: str | None = None


def run_noisy_memory_experiment(
    distance: int,
    num_rounds: int,
    num_shots: int,
    basis: str,
    noise: NoiseModel,
    *,
    decode: bool = True,
    decoder_type: str = "pymatching",
) -> SimulationResult:
    """Run a noisy surface code memory experiment with optional decoding.

    This function:
    1. Creates a surface code patch and Guppy circuit
    2. Compiles to HUGR and runs with Selene using depolarizing noise
    3. Collects syndromes and final measurements
    4. Optionally decodes and computes logical error rate

    Args:
        distance: Code distance (must be odd >= 3)
        num_rounds: Number of syndrome extraction rounds
        num_shots: Number of shots to run
        basis: Memory basis ('Z' or 'X')
        noise: Noise model parameters
        decode: If True, use decoding to correct errors
        decoder_type: Decoder backend (pymatching, fusion_blossom, bp_osd, etc.)

    Returns:
        SimulationResult with error rate statistics

    Example:
        >>> from pecos.qec.surface import run_noisy_memory_experiment, NoiseModel
        >>> noise = NoiseModel(p1=0.001, p2=0.01, p_meas=0.01, p_init=0.001)
        >>> result = run_noisy_memory_experiment(
        ...     distance=3,
        ...     num_rounds=3,
        ...     num_shots=1000,
        ...     basis="Z",
        ...     noise=noise,
        ...     decode=True,
        ... )
        >>> print(f"Logical error rate: {result.logical_error_rate:.4f}")
    """
    from selene_sim import DepolarizingErrorModel, SimpleRuntime, Stim, build

    from pecos.compilation_pipeline import compile_guppy_to_hugr
    from pecos.guppy.surface import get_num_qubits, make_surface_code
    from pecos.qec.surface import SurfacePatch

    # Create patch and decoder
    patch = SurfacePatch.create(distance=distance)
    geom = patch.geometry

    # Get logical operator qubits
    if basis.upper() == "Z":
        logical_qubits = geom.logical_z.data_qubits if geom.logical_z else ()
    else:
        logical_qubits = geom.logical_x.data_qubits if geom.logical_x else ()

    # Create decoder if needed
    decoder = None
    if decode:
        decoder = SurfaceDecoder(
            patch,
            num_rounds=num_rounds,
            noise=noise,
            decoder_type=decoder_type,
        )

    # Build and compile circuit
    num_qubits = get_num_qubits(distance)
    prog = make_surface_code(distance=distance, num_rounds=num_rounds, basis=basis)
    hugr_bytes = compile_guppy_to_hugr(prog)
    instance = build(hugr_bytes, name=f"surface_d{distance}")

    # Create error model
    error_model = DepolarizingErrorModel(
        p_1q=noise.p1,
        p_2q=noise.p2,
        p_meas=noise.p_meas,
        p_init=noise.p_init,
    )

    # Run shots
    num_logical_errors = 0
    num_raw_errors = 0

    for shot_results in instance.run_shots(
        simulator=Stim(),
        n_qubits=num_qubits,
        n_shots=num_shots,
        error_model=error_model,
        runtime=SimpleRuntime(),
        n_processes=1,
    ):
        # Collect syndromes
        synx_list = []
        synz_list = []
        final = None

        for name, values in shot_results:
            vals = list(values)
            if name == "synx":
                synx_list.append(np.array(vals, dtype=np.uint8))
            elif name == "synz":
                synz_list.append(np.array(vals, dtype=np.uint8))
            elif name == "final":
                final = vals

        if final is None:
            continue

        # Raw parity check
        raw_parity = sum(final[q] for q in logical_qubits) % 2
        if raw_parity != 0:
            num_raw_errors += 1

        if decode and decoder is not None:
            final_arr = np.array(final, dtype=np.uint8)

            # Decode based on basis
            if basis.upper() == "Z":
                is_error, _ = decoder.decode_memory_z(synx_list, synz_list, final_arr)
            else:
                is_error, _ = decoder.decode_memory_x(synx_list, synz_list, final_arr)

            if is_error:
                num_logical_errors += 1
        else:
            # No decoding - use raw parity
            if raw_parity != 0:
                num_logical_errors += 1

    return SimulationResult(
        distance=distance,
        num_shots=num_shots,
        num_rounds=num_rounds,
        basis=basis,
        num_logical_errors=num_logical_errors,
        num_raw_errors=num_raw_errors,
        logical_error_rate=num_logical_errors / num_shots if num_shots > 0 else 0.0,
        raw_error_rate=num_raw_errors / num_shots if num_shots > 0 else 0.0,
        decoded=decode,
        decoder_type=decoder_type if decode else None,
    )


# =============================================================================
# PECOS Native Sampling
# =============================================================================


@dataclass
class NativeSampler:
    """PECOS native sampler for threshold estimation.

    This provides a pure-PECOS alternative to Stim's DEM sampler,
    using the MeasurementNoiseModel (MNM) for efficient sampling.

    The sampler uses explicit detector and observable definitions from
    TickCircuit metadata, matching Stim's output format closely (~98%
    per-detector correlation in testing).

    Two sampling backends are available:
    - MNM (default): Samples measurement outcomes, computes events from definitions
    - NoisySampler: Samples fault locations directly (faster for statistics)

    Attributes:
        mnm: The MeasurementNoiseModel for sampling
        detectors_json: JSON string with detector definitions
        observables_json: JSON string with observable definitions
        num_detectors: Number of detectors
        num_observables: Number of observables
    """

    mnm: MeasurementNoiseModel
    detectors_json: str
    observables_json: str
    num_detectors: int
    num_observables: int

    def sample(
        self,
        num_shots: int,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample detection events and observable flips.

        This matches Stim's DEM sampler output format.

        Args:
            num_shots: Number of shots to sample
            seed: Optional random seed for reproducibility

        Returns:
            Tuple of (detection_events, observable_flips) as numpy arrays.
            - detection_events: shape (num_shots, num_detectors)
            - observable_flips: shape (num_shots, num_observables)
        """
        det_events, obs_flips = self.mnm.sample_batch_for_decoding(
            num_shots,
            self.detectors_json,
            self.observables_json,
            seed,
        )
        return np.array(det_events, dtype=bool), np.array(obs_flips, dtype=bool)


def build_native_sampler(
    patch: SurfacePatch,
    num_rounds: int,
    noise: NoiseModel,
    basis: str = "Z",
) -> NativeSampler:
    """Build a PECOS native sampler for threshold estimation.

    This creates a sampler that can generate (detection_events, observable_flips)
    pairs using PECOS native fault propagation, providing an alternative to
    Stim's DEM sampler.

    The pipeline is:
    TickCircuit -> DagCircuit -> DagFaultAnalyzer -> InfluenceMap -> MNM -> Sampler

    Args:
        patch: Surface code patch with geometry
        num_rounds: Number of syndrome extraction rounds
        noise: Noise model parameters
        basis: Memory basis ('X' or 'Z')

    Returns:
        NativeSampler that can generate samples for threshold estimation

    Example:
        >>> from pecos.qec.surface import SurfacePatch, NoiseModel, build_native_sampler
        >>> patch = SurfacePatch.create(distance=5)
        >>> noise = NoiseModel(p1=0.001, p2=0.001, p_meas=0.001)
        >>> sampler = build_native_sampler(patch, num_rounds=5, noise=noise)
        >>> detection_events, observable_flips = sampler.sample(num_shots=10000)
    """
    import json

    from pecos.qec import DagFaultAnalyzer, MemBuilder
    from pecos.qec.surface.circuit_builder import (
        _extract_measurement_order,
        generate_tick_circuit_from_patch,
    )

    # Generate TickCircuit (source of truth for circuit structure)
    tc = generate_tick_circuit_from_patch(patch, num_rounds, basis)

    # Convert to DAG and build influence map via Rust fault propagation
    dag = tc.to_dag_circuit()
    analyzer = DagFaultAnalyzer(dag)
    influence_map = analyzer.build_influence_map()

    # Extract metadata from TickCircuit
    detectors_json = tc.get_meta("detectors") or "[]"
    observables_json = tc.get_meta("observables") or "[]"
    measurement_order = _extract_measurement_order(tc)

    # Build MNM for sampling
    builder = MemBuilder(influence_map)
    builder.with_noise(noise.p1, noise.p2, noise.p_meas, noise.p_init)
    builder.with_measurement_order(measurement_order)
    mnm = builder.build()

    # Parse to count detectors/observables
    num_detectors = len(json.loads(detectors_json)) if detectors_json else 0
    num_observables = len(json.loads(observables_json)) if observables_json else 0

    return NativeSampler(
        mnm=mnm,
        detectors_json=detectors_json,
        observables_json=observables_json,
        num_detectors=num_detectors,
        num_observables=num_observables,
    )
