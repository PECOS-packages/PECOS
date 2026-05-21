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
from functools import cache
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    import stim
    from numpy.typing import NDArray

    from pecos.qec.surface.patch import Stabilizer, SurfacePatch


def _validate_probability(name: str, value: float) -> float:
    """Return ``value`` as a float after validating it is a probability."""
    probability = float(value)
    if not 0.0 <= probability <= 1.0:
        msg = f"{name} must be a probability in [0, 1], got {value!r}"
        raise ValueError(msg)
    return probability


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
    """Circuit-level noise parameters for QEC simulation.

    Matches the Rust ``NoiseConfig`` type. All parameters are optional
    beyond the four base rates.

    Attributes:
        p1: Single-qubit gate error rate.
        p2: Two-qubit gate error rate.
        p_meas: Measurement error rate.
        p_prep: Initialization error rate.
        p_idle: Idle noise rate per time unit (uniform depolarizing).
        t1: T1 relaxation time for idle noise (same units as idle duration).
        t2: T2 dephasing time (must satisfy t2 <= 2*t1).
    """

    p1: float = 0.0
    p2: float = 0.0
    p_meas: float = 0.0
    p_prep: float = 0.0
    p_idle: float | None = None
    t1: float | None = None
    t2: float | None = None

    @staticmethod
    def uniform(physical_error_rate: float) -> NoiseModel:
        """Create a uniform circuit-level noise model from one physical error rate."""
        p = _validate_probability("physical_error_rate", physical_error_rate)
        return NoiseModel(p1=p, p2=p, p_meas=p, p_prep=p)

    @property
    def is_noiseless(self) -> bool:
        """True if all error rates are zero."""
        return (
            self.p1 == 0.0
            and self.p2 == 0.0
            and self.p_meas == 0.0
            and self.p_prep == 0.0
            and (self.p_idle is None or self.p_idle == 0.0)
        )

    @property
    def physical_error_rate(self) -> float:
        """Approximate combined physical error rate."""
        rates = [self.p1, self.p2, self.p_meas, self.p_prep]
        if self.p_idle is not None:
            rates.append(self.p_idle)
        return max(rates)


@dataclass
class DecodingResult:
    """Result from decoding a single shot."""

    x_correction: NDArray[np.uint8]  # X corrections to apply to data qubits
    z_correction: NDArray[np.uint8]  # Z corrections to apply to data qubits
    logical_x_flip: bool  # True if logical X was flipped by correction
    logical_z_flip: bool  # True if logical Z was flipped by correction
    decoding_weight: float  # Weight of the matching solution


@dataclass(frozen=True)
class _CachedNativeSurfaceTopology:
    """Topology-only native model data reused across noise configurations."""

    influence_map: Any
    detectors_json: str
    observables_json: str
    measurement_order: tuple[int, ...]
    num_measurements: int
    num_detectors: int
    num_observables: int


def _surface_patch_cache_key(patch: SurfacePatch) -> tuple[int, int, str, bool]:
    """Create a stable cache key for surface-patch topology."""
    return (
        patch.dx,
        patch.dz,
        patch.geometry.orientation.name,
        patch.geometry.rotated,
    )


@cache
def _cached_surface_patch(patch_key: tuple[int, int, str, bool]) -> SurfacePatch:
    """Recreate a canonical patch from a geometry cache key."""
    from pecos.qec.surface.patch import PatchOrientation, SurfacePatch

    dx, dz, orientation_name, rotated = patch_key
    return SurfacePatch.create(
        dx=dx,
        dz=dz,
        orientation=PatchOrientation[orientation_name],
        rotated=rotated,
    )


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


def _copy_surface_tick_circuit_metadata(source_tc: Any, target_tc: Any) -> None:
    """Copy the surface-level metadata needed by the native DEM/sampler builders."""
    for key in (
        "basis",
        "detectors",
        "observables",
        "num_measurements",
        "num_detectors",
        "detector_descriptors",
        "observable_descriptors",
        "ancilla_budget",
    ):
        value = source_tc.get_meta(key)
        if value is not None:
            target_tc.set_meta(key, value)


def _replay_qis_trace_into_tick_circuit(operations: list[dict[str, Any]]) -> Any:
    """Replay traced QIS operations into a PECOS TickCircuit."""
    import heapq

    from pecos_rslib.quantum import TickCircuit

    tick_circuit = TickCircuit()
    active_slots: dict[int, int] = {}
    free_slots: list[int] = []
    next_slot = 0

    def allocate_slot(program_id: int) -> int:
        nonlocal next_slot
        if program_id in active_slots:
            return active_slots[program_id]
        if free_slots:
            slot = heapq.heappop(free_slots)
        else:
            slot = next_slot
            next_slot += 1
        active_slots[program_id] = slot
        return slot

    def release_slot(program_id: int) -> None:
        slot = active_slots.pop(program_id, None)
        if slot is not None:
            heapq.heappush(free_slots, slot)

    def mapped_slot(program_id: int, op_name: str) -> int:
        if program_id not in active_slots:
            msg = f"Traced QIS op {op_name!r} referenced unmapped program qubit {program_id}"
            raise ValueError(msg)
        return active_slots[program_id]

    def scalar_arg(payload: Any, op_name: str) -> int:
        if isinstance(payload, list):
            msg = f"Expected scalar payload for {op_name}, got {payload!r}"
            raise TypeError(msg)
        return int(payload)

    def tuple_args(payload: Any, op_name: str, arity: int) -> tuple[Any, ...]:
        if not isinstance(payload, list) or len(payload) != arity:
            msg = f"Expected {arity} arguments for {op_name}, got {payload!r}"
            raise ValueError(msg)
        return tuple(payload)

    for operation in operations:
        if "AllocateQubit" in operation:
            program_id = int(operation["AllocateQubit"]["id"])
            slot = allocate_slot(program_id)
            tick_circuit.tick().pz([slot])
            continue

        if "ReleaseQubit" in operation:
            release_slot(int(operation["ReleaseQubit"]["id"]))
            continue

        if "AllocateResult" in operation or "RecordOutput" in operation or "Barrier" in operation:
            continue

        quantum = operation.get("Quantum")
        if quantum is None or len(quantum) != 1:
            msg = f"Unsupported traced operation payload: {operation!r}"
            raise ValueError(msg)

        op_name, payload = next(iter(quantum.items()))
        tick = tick_circuit.tick()

        if op_name == "H":
            tick.h([mapped_slot(scalar_arg(payload, op_name), op_name)])
        elif op_name == "X":
            tick.x([mapped_slot(scalar_arg(payload, op_name), op_name)])
        elif op_name == "Y":
            tick.y([mapped_slot(scalar_arg(payload, op_name), op_name)])
        elif op_name == "Z":
            tick.z([mapped_slot(scalar_arg(payload, op_name), op_name)])
        elif op_name == "S":
            tick.sz([mapped_slot(scalar_arg(payload, op_name), op_name)])
        elif op_name == "Sdg":
            tick.szdg([mapped_slot(scalar_arg(payload, op_name), op_name)])
        elif op_name == "T":
            tick.t([mapped_slot(scalar_arg(payload, op_name), op_name)])
        elif op_name == "Tdg":
            tick.tdg([mapped_slot(scalar_arg(payload, op_name), op_name)])
        elif op_name == "RX":
            theta, program_id = tuple_args(payload, op_name, 2)
            tick.rx(float(theta), [mapped_slot(int(program_id), op_name)])
        elif op_name == "RY":
            theta, program_id = tuple_args(payload, op_name, 2)
            tick.ry(float(theta), [mapped_slot(int(program_id), op_name)])
        elif op_name == "RZ":
            theta, program_id = tuple_args(payload, op_name, 2)
            tick.rz(float(theta), [mapped_slot(int(program_id), op_name)])
        elif op_name == "RXY":
            theta, phi, program_id = tuple_args(payload, op_name, 3)
            tick.r1xy(float(theta), float(phi), [mapped_slot(int(program_id), op_name)])
        elif op_name == "CX":
            control, target = tuple_args(payload, op_name, 2)
            tick.cx([(mapped_slot(int(control), op_name), mapped_slot(int(target), op_name))])
        elif op_name == "CY":
            control, target = tuple_args(payload, op_name, 2)
            tick.cy([(mapped_slot(int(control), op_name), mapped_slot(int(target), op_name))])
        elif op_name == "CZ":
            control, target = tuple_args(payload, op_name, 2)
            tick.cz([(mapped_slot(int(control), op_name), mapped_slot(int(target), op_name))])
        elif op_name == "CH":
            control, target = tuple_args(payload, op_name, 2)
            tick.ch([(mapped_slot(int(control), op_name), mapped_slot(int(target), op_name))])
        elif op_name == "CRZ":
            theta, control, target = tuple_args(payload, op_name, 3)
            tick.crz(
                float(theta),
                [(mapped_slot(int(control), op_name), mapped_slot(int(target), op_name))],
            )
        elif op_name == "CCX":
            control_a, control_b, target = tuple_args(payload, op_name, 3)
            tick.ccx(
                [
                    (
                        mapped_slot(int(control_a), op_name),
                        mapped_slot(int(control_b), op_name),
                        mapped_slot(int(target), op_name),
                    ),
                ],
            )
        elif op_name == "ZZ":
            qubit_a, qubit_b = tuple_args(payload, op_name, 2)
            tick.szz([(mapped_slot(int(qubit_a), op_name), mapped_slot(int(qubit_b), op_name))])
        elif op_name == "RZZ":
            theta, qubit_a, qubit_b = tuple_args(payload, op_name, 3)
            tick.rzz(
                float(theta),
                [(mapped_slot(int(qubit_a), op_name), mapped_slot(int(qubit_b), op_name))],
            )
        elif op_name == "Measure":
            program_id, result_id = tuple_args(payload, op_name, 2)
            # Stamp the QIS-provided result_id as the MeasId rather than
            # discarding it and letting assign_missing_meas_ids() invent
            # sequential ids (which would be wrong for non-sequential ids).
            tick.mz_with_ids(
                [mapped_slot(int(program_id), op_name)],
                [int(result_id)],
            )
        elif op_name == "Reset":
            tick.pz([mapped_slot(scalar_arg(payload, op_name), op_name)])
        else:
            msg = f"Unsupported traced QIS quantum op {op_name!r}"
            raise ValueError(msg)

    # Compact: ASAP-schedule gates into minimal ticks
    tick_circuit.compact_ticks()

    return tick_circuit


def _gate_pairs(qubits: list[int], gate_type: str) -> list[tuple[int, int]]:
    """Convert a flattened qubit list into disjoint qubit pairs."""
    if len(qubits) % 2 != 0:
        msg = f"Lowered gate {gate_type!r} expected an even number of qubits, got {qubits!r}"
        raise ValueError(msg)
    return list(zip(qubits[::2], qubits[1::2], strict=True))


def _gate_triples(qubits: list[int], gate_type: str) -> list[tuple[int, int, int]]:
    """Convert a flattened qubit list into disjoint qubit triples."""
    if len(qubits) % 3 != 0:
        msg = f"Lowered gate {gate_type!r} expected qubits in triples, got {qubits!r}"
        raise ValueError(msg)
    return [(qubits[i], qubits[i + 1], qubits[i + 2]) for i in range(0, len(qubits), 3)]


def _replay_lowered_qis_trace_into_tick_circuit(chunks: list[dict[str, Any]]) -> Any:
    """Replay lowered post-Selene ByteMessage gate batches into a TickCircuit.

    The lowered trace emits gates one at a time. We replay each into its own
    tick, then compact (ASAP schedule) so that gates on disjoint qubits share
    a tick --- matching the parallel structure of the abstract circuit.

    MeasIds flow from the QIS measurement result slot: Quantum.Measure carries
    ``[qubit, result_id]``, and those IDs are stamped on MZ gates via
    mz_with_ids().
    """
    from pecos_rslib.quantum import TickCircuit

    tick_circuit = TickCircuit()

    # Pass 1: the ordered MeasIds, read directly from each Measure op. A
    # ``Quantum.Measure`` op carries ``[qubit, result_id]`` where ``result_id``
    # is the QIS result slot the runtime allocated for it (== the MeasId we
    # stamp). Using it directly needs no AllocateResult/Measure pairing
    # heuristic and no interleave assumption -- batched
    # allocate-allocate-measure-measure (a valid QIS pattern) works the same
    # as interleaved. (The order of Measure ops here matches the order of MZ
    # gates in ``lowered_quantum_ops``, consumed in pass 2.)
    meas_ids_in_order: list[int] = []
    for chunk in chunks:
        for op in chunk.get("operations") or []:
            quantum = dict(op).get("Quantum")
            if isinstance(quantum, dict) and "Measure" in quantum:
                meas_ids_in_order.append(int(quantum["Measure"][1]))

    # Pass 2: replay gates, stamping MeasIds on MZ gates in global trace order.
    meas_cursor = 0
    for chunk in chunks:
        for gate in chunk.get("lowered_quantum_ops") or []:
            gate_type = str(gate["gate_type"])
            qubits = [int(q) for q in gate.get("qubits", [])]
            angles = [float(theta) for theta in gate.get("angles", [])]
            tick = tick_circuit.tick()

            if gate_type == "H":
                tick.h(qubits)
            elif gate_type == "X":
                tick.x(qubits)
            elif gate_type == "Y":
                tick.y(qubits)
            elif gate_type == "Z":
                tick.z(qubits)
            elif gate_type == "SZ":
                tick.sz(qubits)
            elif gate_type == "SZdg":
                tick.szdg(qubits)
            elif gate_type == "T":
                tick.t(qubits)
            elif gate_type == "Tdg":
                tick.tdg(qubits)
            elif gate_type == "PZ":
                tick.pz(qubits)
            elif gate_type == "MZ":
                end = meas_cursor + len(qubits)
                if end > len(meas_ids_in_order):
                    msg = (
                        "More measured qubits than result(...)-anchored "
                        "MeasIds in the traced program; a measurement is "
                        "missing its result(...) call."
                    )
                    raise ValueError(msg)
                tick.mz_with_ids(qubits, meas_ids_in_order[meas_cursor:end])
                meas_cursor = end
            elif gate_type == "RX":
                tick.rx(angles[0], qubits)
            elif gate_type == "RY":
                tick.ry(angles[0], qubits)
            elif gate_type == "RZ":
                tick.rz(angles[0], qubits)
            elif gate_type == "R1XY":
                tick.r1xy(angles[0], angles[1], qubits)
            elif gate_type == "CX":
                tick.cx(_gate_pairs(qubits, gate_type))
            elif gate_type == "CY":
                tick.cy(_gate_pairs(qubits, gate_type))
            elif gate_type == "CZ":
                tick.cz(_gate_pairs(qubits, gate_type))
            elif gate_type == "CH":
                tick.ch(_gate_pairs(qubits, gate_type))
            elif gate_type == "CRZ":
                tick.crz(angles[0], _gate_pairs(qubits, gate_type))
            elif gate_type == "SZZ":
                tick.szz(_gate_pairs(qubits, gate_type))
            elif gate_type == "SZZdg":
                tick.szzdg(_gate_pairs(qubits, gate_type))
            elif gate_type == "RZZ":
                tick.rzz(angles[0], _gate_pairs(qubits, gate_type))
            elif gate_type == "CCX":
                tick.ccx(_gate_triples(qubits, gate_type))
            else:
                msg = f"Unsupported lowered traced gate {gate_type!r}"
                raise ValueError(msg)

    if meas_cursor != len(meas_ids_in_order):
        msg = (
            f"Traced program has {len(meas_ids_in_order)} result(...)-anchored "
            f"measurements but only {meas_cursor} measured qubit(s) in the "
            "lowered gate stream; result()/measurement mismatch."
        )
        raise ValueError(msg)

    # Compact: ASAP-schedule gates into minimal ticks
    tick_circuit.compact_ticks()

    return tick_circuit


def _chunk_has_lowerable_op(chunk: dict[str, Any]) -> bool:
    """True if a chunk carries an operation that lowers to a TickCircuit gate.

    A raw ``Quantum`` op (gate / measure / reset) lowers to a gate, and an
    ``AllocateQubit`` lowers to a prep (``PZ``) -- both appear in
    ``lowered_quantum_ops`` after Selene lowering, and both are emitted as
    gates by the raw replay (see :func:`_replay_qis_trace_into_tick_circuit`).
    ``AllocateResult``, ``RecordOutput``, ``Barrier``, and ``ReleaseQubit``
    emit no gate and are pass-through bookkeeping, so a chunk containing only
    those legitimately has no lowered ops.
    """
    return any(
        isinstance(op, dict) and ("Quantum" in op or "AllocateQubit" in op) for op in (chunk.get("operations") or [])
    )


def _reject_partially_lowered_trace(chunks: list[dict[str, Any]]) -> None:
    """Fail loud on a mixed/partially-lowered trace.

    The lowered replay consumes a chunk's gates from ``lowered_quantum_ops``
    only (it reads ``operations`` solely for measurement result ids). So once
    *any* chunk is lowered, a chunk that carries a lowerable operation (a raw
    ``Quantum`` gate/measure/reset, or an ``AllocateQubit`` prep) but an empty
    ``lowered_quantum_ops`` would have those gates silently dropped -- the
    resulting TickCircuit would be missing operations with no error. A dropped
    *measurement* is already caught downstream by the meas-count guard in
    :func:`_replay_lowered_qis_trace_into_tick_circuit`, but a dropped prep or
    non-measurement gate (H, CX, ...) would pass silently. Reject the
    incomplete trace here instead of building from a partial gate stream.

    This is the explicit trace-format contract for live
    ``capture_operation_trace()`` output: lowered and raw forms must not be
    mixed across chunks. (Per-chunk completeness of lowering is assumed and is
    exercised end-to-end by the byte-identical surface DEM regressions.)
    """
    for idx, chunk in enumerate(chunks):
        if _chunk_has_lowerable_op(chunk) and not chunk.get("lowered_quantum_ops"):
            msg = (
                f"Traced chunk {idx} carries lowerable operations (a quantum "
                "gate/measure/reset or an AllocateQubit prep) but no "
                "lowered_quantum_ops while other chunks are lowered. This "
                "mixed/partially-lowered trace would silently drop the chunk's "
                "gates in the lowered replay; refusing to build from an "
                "incomplete gate stream."
            )
            raise ValueError(msg)


def trace_guppy_into_tick_circuit(program: Any, num_qubits: int, *, seed: int = 0) -> Any:
    """Trace a Guppy/QIS program's lowered Selene op stream into a ``TickCircuit``.

    Runs ``program`` under the Selene QIS engine with operation tracing enabled
    and replays the captured (lowered) gate stream into a PECOS ``TickCircuit``.
    This is the generic core shared by the surface traced-QIS path and the
    general ``DetectorErrorModel.from_guppy`` entry point.

    Note: this traces ONE ideal execution. Measurement-dependent (dynamic)
    control flow is therefore *unsupported / undefined* for DEM construction --
    a single sampled branch is not a static circuit. No reliable runtime-trace
    heuristic distinguishes that from statically-scheduled post-measurement
    gates (the surface code legitimately has those), so no guard is attempted;
    callers must pass straight-line programs.

    Args:
        program: Anything ``pecos.sim`` accepts -- a ``@guppy`` function, a
            compiled Guppy program, or a program wrapper.
        num_qubits: Number of qubits to allocate. QIS/HUGR programs require an
            explicit qubit count for trace capture.
        seed: Seed for the (ideal) trace run.

    Returns:
        A ``TickCircuit`` with no detector/observable metadata attached; the
        caller supplies that.
    """
    import pecos

    sim_builder = (
        pecos.sim(program).classical(pecos.selene_engine()).quantum(pecos.stabilizer()).qubits(num_qubits).seed(seed)
    )
    chunks = list(sim_builder.capture_operation_trace())

    # Selene lowers QIS gates into per-chunk `lowered_quantum_ops` (the gate
    # shape actually executed; e.g. cx -> RZZ + rotations). When any chunk is
    # lowered we replay from those, but first reject a mixed/partially-lowered
    # trace that would silently drop a chunk's raw gates (see
    # `_reject_partially_lowered_trace`).
    if any(chunk.get("lowered_quantum_ops") for chunk in chunks):
        _reject_partially_lowered_trace(chunks)
        return _replay_lowered_qis_trace_into_tick_circuit(chunks)

    # No chunk was lowered: replay the uniformly-raw QIS operation stream.
    operations: list[dict[str, Any]] = []
    for chunk in chunks:
        operations.extend(list(chunk.get("operations", [])))
    return _replay_qis_trace_into_tick_circuit(operations)


def _generate_traced_surface_tick_circuit(
    patch: SurfacePatch,
    num_rounds: int,
    basis: str,
    *,
    ancilla_budget: int | None = None,
) -> Any:
    """Trace the lowered ideal Selene/QIS op stream and replay it into a TickCircuit.

    With ``ancilla_budget=None``, emits the unconstrained Guppy program
    (one ancilla per stabilizer, all measured at the end of one round).
    With a finite budget, emits the stabilizer-batched program; Selene's
    lowering reuses ancilla slots across batches so the traced TickCircuit
    uses only ``num_data + min(budget, total_ancilla)`` physical qubits
    simultaneously.

    The program and qubit count are derived from the **actual patch**, not
    its scalar distance, so a non-default patch (non-rotated, asymmetric) is
    traced faithfully rather than silently substituting the default rotated
    patch of the same distance.
    """
    from pecos.guppy import get_num_qubits
    from pecos.guppy.surface import generate_memory_experiment

    program = generate_memory_experiment(
        patch,
        num_rounds,
        basis,
        ancilla_budget=ancilla_budget,
    )
    return trace_guppy_into_tick_circuit(
        program,
        get_num_qubits(patch=patch, ancilla_budget=ancilla_budget),
        seed=0,
    )


def _build_surface_tick_circuit_for_native_model(
    patch: SurfacePatch,
    num_rounds: int,
    basis: str,
    *,
    ancilla_budget: int | None = None,
    circuit_source: Literal["abstract", "traced_qis"] = "abstract",
) -> Any:
    """Build the TickCircuit used by the native DEM and sampler paths."""
    from pecos.qec.surface.circuit_builder import (
        _extract_measurement_order,
        generate_tick_circuit_from_patch,
    )

    abstract_tc = generate_tick_circuit_from_patch(
        patch,
        num_rounds,
        basis,
        ancilla_budget=ancilla_budget,
    )

    if circuit_source == "abstract":
        return abstract_tc

    if circuit_source != "traced_qis":
        msg = f"Unknown circuit_source {circuit_source!r}"
        raise ValueError(msg)

    traced_tc = _generate_traced_surface_tick_circuit(
        patch,
        num_rounds,
        basis,
        ancilla_budget=ancilla_budget,
    )
    # Coarse sanity check: the traced and abstract circuits must agree on the
    # sequence of *measured qubit indices*. This catches gross drift (a dropped
    # or added measurement, a wrong-qubit measurement, a different schedule
    # shape). It is NOT an identity-level check: `_extract_measurement_order`
    # returns physical qubit indices, and under ancilla reuse the same physical
    # qubit appears in many measurements -- so two different stabilizer
    # orderings can produce an identical qubit-index sequence and pass here.
    # There is no independent stabilizer-identity oracle in the stack today:
    # the detector/observable record offsets are the production binding (not a
    # validator), and the byte-identical traced-vs-traced DEM regression shares
    # the same shared batching policy on both sides (so it cannot catch a
    # policy bug). The current safeguards against identity drift are the shared
    # `batched_stabilizers` source-of-truth and the source-level CX-emission
    # pins; a true identity check here would need stabilizer provenance the
    # replayed TickCircuit does not currently carry (future work).
    traced_measurement_order = _extract_measurement_order(traced_tc)
    abstract_measurement_order = _extract_measurement_order(abstract_tc)
    if traced_measurement_order != abstract_measurement_order:
        msg = (
            "Traced and abstract surface circuits disagree on the measured-qubit "
            "sequence (a dropped/added/wrong-qubit measurement or a different "
            "schedule shape); refusing to build a native DEM/sampler from a "
            "circuit that does not match the abstract detector/observable metadata"
        )
        raise ValueError(msg)

    _copy_surface_tick_circuit_metadata(abstract_tc, traced_tc)
    traced_tc.set_meta("circuit_source", circuit_source)
    return traced_tc


def build_memory_circuit(
    *,
    rounds: int,
    distance: int | None = None,
    patch: SurfacePatch | None = None,
    basis: str = "Z",
    ancilla_budget: int | None = None,
    circuit_source: Literal["abstract", "traced_qis"] = "abstract",
) -> Any:
    """Build the standard surface-code memory ``TickCircuit``.

    This is the public, friendly entry point for the circuit used by PECOS's
    native DEM, sampler, and decoder helpers.

    Args:
        rounds: Number of syndrome-extraction rounds.
        distance: Rotated surface-code distance. Provide either ``distance``
            or ``patch``.
        patch: Explicit surface-code patch. Provide either ``patch`` or
            ``distance``.
        basis: Memory basis, ``"Z"`` or ``"X"``.
        ancilla_budget: Optional cap on simultaneously live ancillas.
        circuit_source: ``"abstract"`` for the native surface builder or
            ``"traced_qis"`` for the lowered traced QIS gate stream.

    Returns:
        A Rust-backed ``TickCircuit`` with detector and observable metadata.

    Example:
        >>> from pecos.qec.surface import build_memory_circuit
        >>> tc = build_memory_circuit(distance=3, rounds=3, basis="Z")
        >>> int(tc.get_meta("num_measurements")) > 0
        True
    """
    from pecos.qec.surface.patch import SurfacePatch

    if rounds < 1:
        msg = f"rounds must be >= 1, got {rounds}"
        raise ValueError(msg)
    if patch is None:
        if distance is None:
            msg = "build_memory_circuit requires either distance=... or patch=..."
            raise ValueError(msg)
        patch = SurfacePatch.create(distance=distance)
    elif distance is not None:
        msg = "build_memory_circuit accepts either distance=... or patch=..., not both"
        raise ValueError(msg)

    return _build_surface_tick_circuit_for_native_model(
        patch,
        rounds,
        basis,
        ancilla_budget=ancilla_budget,
        circuit_source=circuit_source,
    )


def _canonical_ancilla_budget(patch: SurfacePatch, ancilla_budget: int | None) -> int | None:
    """Canonicalize an ancilla budget for the shared native topology cache.

    Collapses every "unconstrained" spelling -- ``None``, a budget equal to
    ``total_ancilla``, or any larger value -- to ``None`` so they share one
    cache entry and use the unconstrained codegen path; a genuine constraint
    (``< total_ancilla``) passes through unchanged. Routing through
    :func:`normalize_ancilla_budget` also validates type/range fail-loud at the
    cache boundary.

    All cache parameters (``ancilla_budget``, ``circuit_source``, idle-gate
    insertion) are independent keys on the cached functions, so constrained
    budgets cache correctly -- there is no correctness reason to bypass the
    cache for them. ``None``/``== total``/``>> total`` were verified to produce
    byte-identical DEMs for both circuit sources, so canonicalizing them
    together is behavior-preserving.
    """
    if ancilla_budget is None:
        return None
    from pecos.qec.surface._ancilla_batching import normalize_ancilla_budget

    geom = patch.geometry
    total_ancilla = len(geom.x_stabilizers) + len(geom.z_stabilizers)
    effective = normalize_ancilla_budget(total_ancilla, ancilla_budget)
    return None if effective >= total_ancilla else effective


def _uses_dedicated_idle_noise(
    *,
    p_idle: float | None,
    t1: float | None,
    t2: float | None,
) -> bool:
    """Return True when noise parameters require explicit idle locations."""
    return (p_idle is not None and p_idle > 0.0) or (t1 is not None and t2 is not None)


def _noise_uses_dedicated_idle_noise(noise: NoiseModel) -> bool:
    """Return True when this noise model requires explicit idle locations."""
    return _uses_dedicated_idle_noise(p_idle=noise.p_idle, t1=noise.t1, t2=noise.t2)


@cache
def _cached_surface_native_topology(
    patch_key: tuple[int, int, str, bool],
    num_rounds: int,
    basis: str,
    ancilla_budget: int | None,
    circuit_source: Literal["abstract", "traced_qis"],
    include_idle_gates: bool,
) -> _CachedNativeSurfaceTopology:
    """Cache topology-only native analysis shared across noise parameters."""
    import json

    from pecos.qec import DagFaultAnalyzer
    from pecos.qec.surface.circuit_builder import _extract_measurement_order

    patch = _cached_surface_patch(patch_key)
    tc = _build_surface_tick_circuit_for_native_model(
        patch,
        num_rounds,
        basis,
        ancilla_budget=ancilla_budget,
        circuit_source=circuit_source,
    )
    if include_idle_gates:
        # Insert idle gates only when the requested noise model includes a
        # dedicated idle channel. Otherwise inserted idle gates receive ordinary
        # one-qubit gate noise and change the explicit circuit-level DEM.
        tc.fill_idle_gates()

    dag = tc.to_dag_circuit()
    analyzer = DagFaultAnalyzer(dag)
    influence_map = analyzer.build_influence_map()

    detectors_json = tc.get_meta("detectors") or "[]"
    observables_json = tc.get_meta("observables") or "[]"
    measurement_order = tuple(_extract_measurement_order(tc))
    num_measurements = int(tc.get_meta("num_measurements") or str(len(measurement_order)))

    return _CachedNativeSurfaceTopology(
        influence_map=influence_map,
        detectors_json=detectors_json,
        observables_json=observables_json,
        measurement_order=measurement_order,
        num_measurements=num_measurements,
        num_detectors=len(json.loads(detectors_json)) if detectors_json else 0,
        num_observables=len(json.loads(observables_json)) if observables_json else 0,
    )


def _dem_string_from_cached_surface_topology(
    topology: _CachedNativeSurfaceTopology,
    noise: NoiseModel,
    *,
    decompose_errors: bool,
) -> str:
    """Build a DEM string from cached topology and fresh noise parameters."""
    from pecos.qec import DemBuilder

    dem = (
        DemBuilder(topology.influence_map)
        .with_noise(noise.p1, noise.p2, noise.p_meas, noise.p_prep, p_idle=noise.p_idle, t1=noise.t1, t2=noise.t2)
        .with_num_measurements(topology.num_measurements)
        .with_measurement_order(list(topology.measurement_order))
        .with_detectors_json(topology.detectors_json)
        .with_observables_json(topology.observables_json)
        .build_with_source_tracking()
    )
    return dem.to_string_decomposed() if decompose_errors else dem.to_string()


@cache
def _cached_surface_native_dem_string(
    patch_key: tuple[int, int, str, bool],
    num_rounds: int,
    basis: str,
    ancilla_budget: int | None,
    circuit_source: Literal["abstract", "traced_qis"],
    p1: float,
    p2: float,
    p_meas: float,
    p_prep: float,
    decompose_errors: bool,
    p_idle: float | None = None,
    t1: float | None = None,
    t2: float | None = None,
) -> str:
    """Cache native DEM strings across callers for one topology + noise tuple."""
    include_idle_gates = _uses_dedicated_idle_noise(p_idle=p_idle, t1=t1, t2=t2)
    topology = _cached_surface_native_topology(
        patch_key,
        num_rounds,
        basis,
        ancilla_budget,
        circuit_source,
        include_idle_gates,
    )
    return _dem_string_from_cached_surface_topology(
        topology,
        NoiseModel(p1=p1, p2=p2, p_meas=p_meas, p_prep=p_prep, p_idle=p_idle, t1=t1, t2=t2),
        decompose_errors=decompose_errors,
    )


@cache
def _cached_parsed_dem(dem_str: str) -> Any:
    """Cache parsed DEM objects so repeated sampler builds only instantiate the sampler."""
    from pecos.qec import ParsedDem

    return ParsedDem.from_string(dem_str)


def _build_native_sampler_from_cached_surface_topology(
    topology: _CachedNativeSurfaceTopology,
    noise: NoiseModel,
    *,
    sampling_model: Literal[
        "dem",
        "influence_dem",
        "mnm",
    ] = "dem",  # "mnm" accepted for compat, mapped to "influence_dem",
) -> NativeSampler:
    """Construct a native sampler from cached topology-only analysis."""
    from pecos.qec import DemSampler, ParsedDem

    if sampling_model == "dem":
        dem_str = _dem_string_from_cached_surface_topology(
            topology,
            noise,
            decompose_errors=True,
        )
        sampler = ParsedDem.from_string(dem_str).to_dem_sampler()
    elif sampling_model in ("influence_dem", "mnm"):
        import json

        det_records = [d["records"] for d in json.loads(topology.detectors_json)]
        obs_records = [o["records"] for o in json.loads(topology.observables_json)] if topology.observables_json else []
        sampler = DemSampler.with_detectors(
            topology.influence_map,
            det_records,
            obs_records,
            noise.p1,
            noise.p2,
            noise.p_meas,
            noise.p_prep,
            p_idle=noise.p_idle,
            t1=noise.t1,
            t2=noise.t2,
        )
        # Remap sampling_model for NativeSampler dispatch
        sampling_model = "influence_dem"
    else:
        msg = f"Unknown native sampling_model {sampling_model!r}"
        raise ValueError(msg)

    return NativeSampler(
        sampler=sampler,
        detectors_json=topology.detectors_json,
        observables_json=topology.observables_json,
        num_detectors=topology.num_detectors,
        num_observables=topology.num_observables,
        sampling_model=sampling_model,
    )


def generate_circuit_level_dem_from_builder(
    patch: SurfacePatch,
    num_rounds: int,
    noise: NoiseModel,
    basis: str = "Z",
    *,
    decompose_errors: bool = False,
    ancilla_budget: int | None = None,
    circuit_source: Literal["abstract", "traced_qis"] = "abstract",
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
        decompose_errors: If True, return PECOS's native decomposed DEM
            representation, which is more appropriate for graph-based
            decoders like PyMatching.
        ancilla_budget: Optional cap on simultaneously live ancillas. When
            provided below the total stabilizer count, the native DEM is built
            from the same batched ancilla-reuse circuit family used by Guppy.
        circuit_source: Which ideal circuit to analyze for the native DEM path.
            ``"abstract"`` uses the existing high-level surface TickCircuit.
            ``"traced_qis"`` traces the lowered ideal Selene/QIS gate stream
            and replays that exact gate list into a TickCircuit before running
            native PECOS fault analysis.

    Returns:
        DEM string in standard format

    Example:
        >>> from pecos.qec.surface import SurfacePatch, NoiseModel
        >>> from pecos.qec.surface.decode import generate_circuit_level_dem_from_builder
        >>> patch = SurfacePatch.create(distance=3)
        >>> noise = NoiseModel(p1=0.001, p2=0.01, p_meas=0.01)
        >>> dem = generate_circuit_level_dem_from_builder(patch, num_rounds=3, noise=noise)
    """
    ancilla_budget = _canonical_ancilla_budget(patch, ancilla_budget)
    patch_key = _surface_patch_cache_key(patch)
    return _cached_surface_native_dem_string(
        patch_key,
        num_rounds,
        basis.upper(),
        ancilla_budget,
        circuit_source,
        noise.p1,
        noise.p2,
        noise.p_meas,
        noise.p_prep,
        decompose_errors=decompose_errors,
        p_idle=noise.p_idle,
        t1=noise.t1,
        t2=noise.t2,
    )


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
        after_reset_flip_probability=noise.p_prep if noise.p_prep > 0 else 0.0,
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
    *,
    decompose_errors: bool = True,
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
        decompose_errors: If True, return Stim's decomposed graphlike DEM.
            If False, return the raw hypergraph DEM.

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
    dem = circuit.detector_error_model(decompose_errors=decompose_errors)
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
        circuit_level_dem_mode: Literal["native_full", "native_decomposed"] = "native_full",
        circuit_level_dem_source: Literal["abstract", "traced_qis"] = "abstract",
        ancilla_budget: int | None = None,
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
            circuit_level_dem_mode: Which PECOS-native DEM representation to use
                when circuit-level DEMs are enabled. ``"native_full"`` preserves
                the current non-decomposed DEM output. ``"native_decomposed"``
                returns PECOS's graphlike decomposed DEM output, which is often
                a better fit for graph decoders such as PyMatching.
            circuit_level_dem_source: Which ideal circuit to analyze when
                building native circuit-level DEMs. ``"abstract"`` uses the
                high-level surface TickCircuit, while ``"traced_qis"`` traces
                the lowered ideal Selene/QIS gate stream and analyzes that.
            ancilla_budget: Optional cap on simultaneously live ancillas for
                the native circuit-level DEM path. When provided, the decoder
                builds its DEM from the corresponding batched ancilla-reuse
                circuit instead of the default dedicated-ancilla circuit.
        """
        self.patch = patch
        self.num_rounds = num_rounds
        self.noise = noise or NoiseModel(p2=0.01, p_meas=0.01)
        self.decoder_type = DecoderType(decoder_type)
        self.use_circuit_level_dem = use_circuit_level_dem
        self.circuit_level_dem_mode = circuit_level_dem_mode
        self.circuit_level_dem_source = circuit_level_dem_source
        self.ancilla_budget = ancilla_budget

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
        dem = generate_circuit_level_dem_from_builder(
            self.patch,
            self.num_rounds,
            self.noise,
            basis=basis,
            decompose_errors=self.circuit_level_dem_mode == "native_decomposed",
            circuit_source=self.circuit_level_dem_source,
            ancilla_budget=self.ancilla_budget,
        )
        if basis.upper() == "Z":
            self._z_dem = dem
        else:
            self._x_dem = dem
        return dem

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

    def _compute_dem_detection_events_z(
        self,
        synx_list: list[NDArray[np.uint8]],
        synz_list: list[NDArray[np.uint8]],
        final: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """Compute full detection events for Z-basis DEM-based decoding.

        The circuit-level DEM defines detectors in this order:
        1. X stabilizer detectors for rounds 1..num_rounds-1
           (X stabs are non-deterministic at round 0 for Z-basis)
        2. Z stabilizer detectors for rounds 0..num_rounds-1
           (round 0 is deterministic for Z-basis)
        3. Final round detectors: last Z syndrome vs final data parity

        Args:
            synx_list: X syndrome arrays, one per round
            synz_list: Z syndrome arrays, one per round
            final: Final data qubit measurements

        Returns:
            Detection events array matching the DEM detector ordering
        """
        geom = self.patch.geometry
        synx = np.array(synx_list, dtype=np.uint8)
        synz = np.array(synz_list, dtype=np.uint8)

        events: list[int] = []

        # 1. X stabilizer detection events (rounds 1 to num_rounds-1)
        for r in range(1, self.num_rounds):
            events.extend((synx[r] ^ synx[r - 1]).tolist())

        # 2. Z stabilizer detection events (all rounds)
        events.extend(synz[0].tolist())  # round 0: compare to expected 0
        for r in range(1, self.num_rounds):
            events.extend((synz[r] ^ synz[r - 1]).tolist())

        # 3. Final round: parity of final data on each Z stabilizer XOR last syndrome
        for stab in geom.z_stabilizers:
            data_parity = sum(int(final[q]) for q in stab.data_qubits) % 2
            last_syn = int(synz[-1][stab.index])
            events.append((data_parity ^ last_syn) & 1)

        return np.array(events, dtype=np.uint8)

    def _compute_dem_detection_events_x(
        self,
        synx_list: list[NDArray[np.uint8]],
        synz_list: list[NDArray[np.uint8]],
        final: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """Compute full detection events for X-basis DEM-based decoding.

        The circuit-level DEM defines detectors in this order:
        1. X stabilizer detectors for rounds 0..num_rounds-1
           (X stabs are deterministic at round 0 for X-basis)
        2. Z stabilizer detectors for rounds 1..num_rounds-1
           (Z stabs are non-deterministic at round 0 for X-basis)
        3. Final round detectors: last X syndrome vs final data parity

        Args:
            synx_list: X syndrome arrays, one per round
            synz_list: Z syndrome arrays, one per round
            final: Final data qubit measurements

        Returns:
            Detection events array matching the DEM detector ordering
        """
        geom = self.patch.geometry
        synx = np.array(synx_list, dtype=np.uint8)
        synz = np.array(synz_list, dtype=np.uint8)

        events: list[int] = []

        # 1. X stabilizer detection events (all rounds)
        events.extend(synx[0].tolist())  # round 0: compare to expected 0
        for r in range(1, self.num_rounds):
            events.extend((synx[r] ^ synx[r - 1]).tolist())

        # 2. Z stabilizer detection events (rounds 1 to num_rounds-1)
        for r in range(1, self.num_rounds):
            events.extend((synz[r] ^ synz[r - 1]).tolist())

        # 3. Final round: parity of final data on each X stabilizer XOR last syndrome
        for stab in geom.x_stabilizers:
            data_parity = sum(int(final[q]) for q in stab.data_qubits) % 2
            last_syn = int(synx[-1][stab.index])
            events.append((data_parity ^ last_syn) & 1)

        return np.array(events, dtype=np.uint8)

    def decode_memory_z(
        self,
        synx_list: list[NDArray[np.uint8]],
        synz_list: list[NDArray[np.uint8]],
        final: NDArray[np.uint8],
    ) -> tuple[bool, DecodingResult]:
        """Decode a Z-basis memory experiment.

        For Z-basis memory:
        - Z stabilizers detect X errors (which flip Z measurements)
        - We decode Z syndromes to find X corrections
        - Apply corrections to final measurements to get corrected logical Z parity

        For DEM-based decoders (PyMatching, Tesseract with circuit-level DEM):
        - All detection events (both X and Z syndromes + final round) are computed
          to match the DEM's detector ordering
        - The decoder returns a per-observable correction (logical flip prediction)

        For check-matrix decoders (FusionBlossom, LDPC):
        - Only Z syndrome detection events are used
        - The decoder returns a per-qubit correction

        Args:
            synx_list: List of X syndrome arrays, one per round
            synz_list: List of Z syndrome arrays, one per round
            final: Final data qubit measurements

        Returns:
            (is_logical_error, decoding_result)
        """
        geom = self.patch.geometry
        logical_z_qubits = geom.logical_z.data_qubits if geom.logical_z else ()
        final_parity = sum(final[q] for q in logical_z_qubits) % 2

        # DEM-based path: compute full detection events matching DEM detector order
        if self.use_circuit_level_dem and self.decoder_type in (
            DecoderType.PYMATCHING,
            DecoderType.TESSERACT,
        ):
            events = self._compute_dem_detection_events_z(synx_list, synz_list, final)
            events_flat = events.ravel().astype(np.uint8)

            decoder = self._get_z_decoder()

            if self.decoder_type == DecoderType.TESSERACT:
                detection_indices = [i for i, v in enumerate(events_flat) if v != 0]
                result = decoder.decode(detection_indices)
                predicted_obs = result.observables_mask & 1
                weight = result.cost
            else:
                result = decoder.decode(events_flat.tolist())
                predicted_obs = result.correction[0] if len(result.correction) > 0 else 0
                weight = result.weight

            corrected_parity = (final_parity + predicted_obs) % 2
            is_logical_error = corrected_parity != 0

            return is_logical_error, DecodingResult(
                x_correction=np.zeros(self.patch.num_data, dtype=np.uint8),
                z_correction=np.zeros(self.patch.num_data, dtype=np.uint8),
                logical_x_flip=bool(predicted_obs),
                logical_z_flip=False,
                decoding_weight=weight,
            )

        # Check-matrix path (FusionBlossom, LDPC)
        num_z_stab = len(geom.z_stabilizers)
        synz = np.array(synz_list, dtype=np.uint8)
        events = syndromes_to_detection_events(synz, self.num_rounds, num_z_stab)
        raw_syn = synz[-1] if len(synz_list) > 0 else None

        x_correction, weight = self.decode_z_syndrome(events, raw_syndrome=raw_syn)

        if len(x_correction) >= self.patch.num_data:
            correction_parity = sum(x_correction[q] for q in logical_z_qubits) % 2
        else:
            correction_parity = 0

        corrected_parity = (final_parity + correction_parity) % 2
        is_logical_error = corrected_parity != 0

        result = DecodingResult(
            x_correction=(
                x_correction
                if len(x_correction) == self.patch.num_data
                else np.zeros(self.patch.num_data, dtype=np.uint8)
            ),
            z_correction=np.zeros(self.patch.num_data, dtype=np.uint8),
            logical_x_flip=correction_parity != 0,
            logical_z_flip=False,
            decoding_weight=weight,
        )

        return is_logical_error, result

    def decode_memory_x(
        self,
        synx_list: list[NDArray[np.uint8]],
        synz_list: list[NDArray[np.uint8]],
        final: NDArray[np.uint8],
    ) -> tuple[bool, DecodingResult]:
        """Decode an X-basis memory experiment.

        For X-basis memory:
        - X stabilizers detect Z errors (which flip X measurements)
        - We decode X syndromes to find Z corrections
        - Apply corrections to final measurements to get corrected logical X parity

        For DEM-based decoders (PyMatching, Tesseract with circuit-level DEM):
        - All detection events (both X and Z syndromes + final round) are computed
          to match the DEM's detector ordering
        - The decoder returns a per-observable correction (logical flip prediction)

        For check-matrix decoders (FusionBlossom, LDPC):
        - Only X syndrome detection events are used
        - The decoder returns a per-qubit correction

        Args:
            synx_list: List of X syndrome arrays, one per round
            synz_list: List of Z syndrome arrays, one per round
            final: Final data qubit measurements

        Returns:
            (is_logical_error, decoding_result)
        """
        geom = self.patch.geometry
        logical_x_qubits = geom.logical_x.data_qubits if geom.logical_x else ()
        final_parity = sum(final[q] for q in logical_x_qubits) % 2

        # DEM-based path: compute full detection events matching DEM detector order
        if self.use_circuit_level_dem and self.decoder_type in (
            DecoderType.PYMATCHING,
            DecoderType.TESSERACT,
        ):
            events = self._compute_dem_detection_events_x(synx_list, synz_list, final)
            events_flat = events.ravel().astype(np.uint8)

            decoder = self._get_x_decoder()

            if self.decoder_type == DecoderType.TESSERACT:
                detection_indices = [i for i, v in enumerate(events_flat) if v != 0]
                result = decoder.decode(detection_indices)
                predicted_obs = result.observables_mask & 1
                weight = result.cost
            else:
                result = decoder.decode(events_flat.tolist())
                predicted_obs = result.correction[0] if len(result.correction) > 0 else 0
                weight = result.weight

            corrected_parity = (final_parity + predicted_obs) % 2
            is_logical_error = corrected_parity != 0

            return is_logical_error, DecodingResult(
                x_correction=np.zeros(self.patch.num_data, dtype=np.uint8),
                z_correction=np.zeros(self.patch.num_data, dtype=np.uint8),
                logical_x_flip=False,
                logical_z_flip=bool(predicted_obs),
                decoding_weight=weight,
            )

        # Check-matrix path (FusionBlossom, LDPC)
        num_x_stab = len(geom.x_stabilizers)
        synx = np.array(synx_list, dtype=np.uint8)
        events = syndromes_to_detection_events(synx, self.num_rounds, num_x_stab)
        raw_syn = synx[-1] if len(synx_list) > 0 else None

        z_correction, weight = self.decode_x_syndrome(events, raw_syndrome=raw_syn)

        if len(z_correction) >= self.patch.num_data:
            correction_parity = sum(z_correction[q] for q in logical_x_qubits) % 2
        else:
            correction_parity = 0

        corrected_parity = (final_parity + correction_parity) % 2
        is_logical_error = corrected_parity != 0

        result = DecodingResult(
            x_correction=np.zeros(self.patch.num_data, dtype=np.uint8),
            z_correction=(
                z_correction
                if len(z_correction) == self.patch.num_data
                else np.zeros(self.patch.num_data, dtype=np.uint8)
            ),
            logical_x_flip=False,
            logical_z_flip=correction_parity != 0,
            decoding_weight=weight,
        )

        return is_logical_error, result

    def _get_css_uf_decoder(self) -> Any:
        """Get or create the UIUF CSS UF decoder."""
        if not hasattr(self, "_css_uf_decoder") or self._css_uf_decoder is None:
            from pecos_rslib.qec import CssUfDecoder

            x_dem = self.get_dem("X", circuit_level=True)
            z_dem = self.get_dem("Z", circuit_level=True)
            # Strip logical_observable lines (not needed for matching graph).
            x_dem = "\n".join(line for line in x_dem.split("\n") if not line.startswith("logical_observable"))
            z_dem = "\n".join(line for line in z_dem.split("\n") if not line.startswith("logical_observable"))
            self._css_uf_decoder = CssUfDecoder(x_dem, z_dem)
        return self._css_uf_decoder

    def decode_memory_z_uiuf(
        self,
        synx_list: list,
        synz_list: list,
        final: NDArray[np.uint8] | list[int],
    ) -> tuple[bool, DecodingResult]:
        """Decode Z-basis memory using UIUF (joint X/Z intersection).

        Like ``decode_memory_z`` but uses both X and Z syndromes jointly
        to identify Y errors and improve accuracy.

        Args:
            synx_list: List of X syndrome arrays, one per round
            synz_list: List of Z syndrome arrays, one per round
            final: Final data qubit measurements

        Returns:
            (is_logical_error, decoding_result)
        """
        import numpy as np

        geom = self.patch.geometry
        logical_z_qubits = geom.logical_z.data_qubits if geom.logical_z else ()
        final_parity = sum(final[q] for q in logical_z_qubits) % 2

        # Compute detection events for both bases.
        x_events = self._compute_dem_detection_events_x(synx_list, synz_list, final)
        z_events = self._compute_dem_detection_events_z(synx_list, synz_list, final)
        x_flat = x_events.ravel().astype(np.uint8)
        z_flat = z_events.ravel().astype(np.uint8)

        # Joint decode via UIUF.
        decoder = self._get_css_uf_decoder()
        x_obs, z_obs = decoder.decode_css(bytes(x_flat), bytes(z_flat))

        # For Z-basis memory, we care about the Z observable (L0).
        predicted_obs = z_obs & 1
        corrected_parity = (final_parity + predicted_obs) % 2
        is_logical_error = corrected_parity != 0

        return is_logical_error, DecodingResult(
            x_correction=np.zeros(self.patch.num_data, dtype=np.uint8),
            z_correction=np.zeros(self.patch.num_data, dtype=np.uint8),
            logical_x_flip=bool(x_obs & 1),
            logical_z_flip=bool(predicted_obs),
            decoding_weight=0.0,
        )


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


def _memory_noise_model(
    physical_error_rate: float | None,
    noise_model: NoiseModel | None,
) -> NoiseModel:
    """Resolve the surface-memory noise inputs into an explicit NoiseModel."""
    if noise_model is not None:
        if physical_error_rate is not None:
            msg = "pass either physical_error_rate or noise_model, not both"
            raise ValueError(msg)
        return noise_model
    p = 0.001 if physical_error_rate is None else physical_error_rate
    return NoiseModel.uniform(p)


def surface_code_memory(
    *,
    distance: int = 3,
    physical_error_rate: float | None = None,
    noise_model: NoiseModel | None = None,
    shots: int = 1000,
    rounds: int | None = None,
    basis: str = "Z",
    decoder_type: str = "pymatching",
    seed: int | None = None,
    decode: bool = True,
    circuit_source: Literal["abstract", "traced_qis"] = "abstract",
    ancilla_budget: int | None = None,
) -> SimulationResult:
    """Run the recommended native surface-code memory workflow.

    This helper keeps the quick-start path short while using PECOS's Rust-backed
    circuit-level DEM sampler and decoder machinery internally.

    Args:
        distance: Rotated surface-code distance.
        physical_error_rate: Uniform physical error rate used for one-qubit
            gates, two-qubit gates, measurements, and preparation. Defaults to
            ``0.001`` when ``noise_model`` is not provided.
        noise_model: Explicit circuit-level noise model. Mutually exclusive
            with ``physical_error_rate``.
        shots: Number of Monte Carlo shots.
        rounds: Number of syndrome-extraction rounds. Defaults to ``distance``.
        basis: Memory basis, ``"Z"`` or ``"X"``.
        decoder_type: Decoder backend passed to ``SampleBatch.decode_count``.
        seed: Optional sampler seed.
        decode: If false, report the raw observable-flip rate.
        circuit_source: ``"abstract"`` or ``"traced_qis"`` circuit source.
        ancilla_budget: Optional cap on simultaneously live ancillas.

    Returns:
        ``SimulationResult`` with logical and raw error counts/rates.

    Example:
        >>> from pecos.qec.surface import surface_code_memory
        >>> result = surface_code_memory(distance=3, physical_error_rate=0.0, shots=4, rounds=1)
        >>> result.logical_error_rate
        0.0
    """
    from pecos.qec import ParsedDem
    from pecos.qec.surface.patch import SurfacePatch

    if distance < 1:
        msg = f"distance must be >= 1, got {distance}"
        raise ValueError(msg)
    if shots < 0:
        msg = f"shots must be >= 0, got {shots}"
        raise ValueError(msg)
    num_rounds = distance if rounds is None else rounds
    if num_rounds < 1:
        msg = f"rounds must be >= 1, got {num_rounds}"
        raise ValueError(msg)

    noise_model = _memory_noise_model(physical_error_rate, noise_model)
    patch = SurfacePatch.create(distance=distance)
    dem = generate_circuit_level_dem_from_builder(
        patch,
        num_rounds=num_rounds,
        noise=noise_model,
        basis=basis,
        decompose_errors=True,
        ancilla_budget=ancilla_budget,
        circuit_source=circuit_source,
    )
    batch = ParsedDem.from_string(dem).to_dem_sampler().generate_samples(shots, seed)
    num_raw_errors = sum(1 for shot in range(shots) if batch.get_observable_mask(shot) != 0)
    num_logical_errors = batch.decode_count(dem, decoder_type) if decode else num_raw_errors

    return SimulationResult(
        distance=distance,
        num_shots=shots,
        num_rounds=num_rounds,
        basis=basis,
        num_logical_errors=num_logical_errors,
        num_raw_errors=num_raw_errors,
        logical_error_rate=num_logical_errors / shots if shots else 0.0,
        raw_error_rate=num_raw_errors / shots if shots else 0.0,
        decoded=decode,
        decoder_type=decoder_type if decode else None,
    )


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
        >>> noise = NoiseModel(p1=0.001, p2=0.01, p_meas=0.01, p_prep=0.001)
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
        # UIUF uses pymatching-type DEMs internally (decoded via CssUfDecoder).
        dt = "pymatching" if decoder_type == "pecos_uf_uiuf" else decoder_type
        decoder = SurfaceDecoder(
            patch,
            num_rounds=num_rounds,
            noise=noise,
            decoder_type=dt,
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
        p_init=noise.p_prep,
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
            if decoder_type == "pecos_uf_uiuf" and basis.upper() == "Z":
                is_error, _ = decoder.decode_memory_z_uiuf(synx_list, synz_list, final_arr)
            elif basis.upper() == "Z":
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

    This provides a pure-PECOS alternative to Stim's DEM sampler.

    The sampler uses explicit detector and observable definitions from
    TickCircuit metadata, matching Stim's output format closely.

    Two sampling backends are available:
    - `dem` (default): sample the generated decomposed DEM via `ParsedDem`
    - `influence_dem`: sample directly from the influence-map via `DemSampler`

    Attributes:
        sampler: The underlying Rust sampler object
        detectors_json: JSON string with detector definitions
        observables_json: JSON string with observable definitions
        num_detectors: Number of detectors
        num_observables: Number of observables
        sampling_model: Which native sampling backend is active
    """

    sampler: Any
    detectors_json: str
    observables_json: str
    num_detectors: int
    num_observables: int
    sampling_model: Literal["dem", "influence_dem", "mnm"] = (
        "dem"  # "mnm" accepted for compat, mapped to "influence_dem"
    )

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
        det_events, obs_flips = self.sampler.sample_batch(num_shots, seed)
        return np.array(det_events, dtype=bool), np.array(obs_flips, dtype=bool)


def build_native_sampler(
    patch: SurfacePatch,
    num_rounds: int,
    noise: NoiseModel,
    basis: str = "Z",
    ancilla_budget: int | None = None,
    circuit_source: Literal["abstract", "traced_qis"] = "abstract",
    sampling_model: Literal[
        "dem",
        "influence_dem",
        "mnm",
    ] = "dem",  # "mnm" accepted for compat, mapped to "influence_dem",
) -> NativeSampler:
    """Build a PECOS native sampler for threshold estimation.

    This creates a sampler that can generate (detection_events, observable_flips)
    pairs using PECOS native fault propagation, providing an alternative to
    Stim's DEM sampler.

    The pipeline is:
    - `sampling_model="dem"`:
      TickCircuit -> DemBuilder -> ParsedDem -> DemSampler
    - `sampling_model="influence_dem"` (or `"mnm"` for compat):
      TickCircuit -> DagCircuit -> DagFaultAnalyzer -> InfluenceMap -> DemSampler (with detector defs)

    Args:
        patch: Surface code patch with geometry
        num_rounds: Number of syndrome extraction rounds
        noise: Noise model parameters
        basis: Memory basis ('X' or 'Z')
        ancilla_budget: Optional cap on simultaneously live ancillas
        circuit_source: Which ideal circuit to analyze for the native sampler
            path. ``"abstract"`` uses the existing high-level surface
            TickCircuit. ``"traced_qis"`` traces the lowered ideal Selene/QIS
            gate stream and replays that exact gate list into a TickCircuit
            before native PECOS fault analysis.
        sampling_model: Which native sampling backend to use. ``"dem"``
            samples the generated decomposed DEM and is the default.
            ``"influence_dem"`` uses the influence-map-based DemSampler with
            detector definitions. ``"mnm"`` is accepted for compatibility
            and maps to ``"influence_dem"``.

    Returns:
        NativeSampler that can generate samples for threshold estimation

    Example:
        >>> from pecos.qec.surface import SurfacePatch, NoiseModel, build_native_sampler
        >>> patch = SurfacePatch.create(distance=5)
        >>> noise = NoiseModel(p1=0.001, p2=0.001, p_meas=0.001)
        >>> sampler = build_native_sampler(patch, num_rounds=5, noise=noise)
        >>> detection_events, observable_flips = sampler.sample(num_shots=10000)
    """
    ancilla_budget = _canonical_ancilla_budget(patch, ancilla_budget)
    basis = basis.upper()
    patch_key = _surface_patch_cache_key(patch)
    topology = _cached_surface_native_topology(
        patch_key,
        num_rounds,
        basis,
        ancilla_budget,
        circuit_source,
        _noise_uses_dedicated_idle_noise(noise),
    )
    if sampling_model == "dem":
        dem_str = _cached_surface_native_dem_string(
            patch_key,
            num_rounds,
            basis,
            ancilla_budget,
            circuit_source,
            noise.p1,
            noise.p2,
            noise.p_meas,
            noise.p_prep,
            decompose_errors=True,
            p_idle=noise.p_idle,
            t1=noise.t1,
            t2=noise.t2,
        )
        sampler = _cached_parsed_dem(dem_str).to_dem_sampler()
        return NativeSampler(
            sampler=sampler,
            detectors_json=topology.detectors_json,
            observables_json=topology.observables_json,
            num_detectors=topology.num_detectors,
            num_observables=topology.num_observables,
            sampling_model=sampling_model,
        )
    return _build_native_sampler_from_cached_surface_topology(
        topology,
        noise,
        sampling_model=sampling_model,
    )
