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

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from functools import cache
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from pecos.qec.surface._check_plan import require_current_surface_check_plan_renderer, resolve_surface_check_plan
from pecos.quantum import validate_hosted_operations

if TYPE_CHECKING:
    import stim
    from numpy.typing import NDArray

    from pecos.qec.surface._twirl_config import TwirlConfig
    from pecos.qec.surface.patch import Stabilizer, SurfacePatch


P1Weights = Mapping[str, float] | Sequence[tuple[str, float]]
P2Weights = Mapping[str, float] | Sequence[tuple[str, float]]
# Native graphlike decompositions are decoder-facing projections of raw
# hyperedge mechanisms, not alternate exact DEM serializations.
NativeDemDecomposition = Literal["source_graphlike", "terminal_graphlike"]
CircuitLevelDemMode = Literal["native_full", "native_decomposed", "native_terminal_graphlike"]


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
    PYMATCHING_CORRELATED = "pymatching_correlated"
    PYMATCHING_UNCORRELATED = "pymatching_uncorrelated"
    FUSION_BLOSSOM = "fusion_blossom"
    BP_OSD = "bp_osd"
    BP_LSD = "bp_lsd"
    UNION_FIND = "union_find"
    TESSERACT = "tesseract"


DEM_DECODER_TYPES = {
    DecoderType.PYMATCHING,
    DecoderType.PYMATCHING_CORRELATED,
    DecoderType.PYMATCHING_UNCORRELATED,
    DecoderType.TESSERACT,
}

PYMATCHING_DECODER_TYPES = {
    DecoderType.PYMATCHING,
    DecoderType.PYMATCHING_CORRELATED,
    DecoderType.PYMATCHING_UNCORRELATED,
}


@dataclass
class NoiseModel:
    """Circuit-level noise parameters for QEC simulation.

    Matches the Rust ``NoiseConfig`` type. All parameters are optional
    beyond the four base rates.

    Attributes:
        p1: Single-qubit gate error rate.
        p1_weights: Optional relative probabilities over single-qubit Pauli
            error labels ``"X"``, ``"Y"``, and ``"Z"``. Values must sum to
            1.0; ``p1`` remains the total single-qubit error rate.
        p2: Two-qubit gate error rate.
        p2_szz: Optional total error-rate override for ``SZZ`` gates. When
            unset, ``SZZ`` uses ``p2``.
        p2_szzdg: Optional total error-rate override for ``SZZdg`` gates. When
            unset, ``SZZdg`` uses ``p2``.
        p2_weights: Optional relative probabilities over two-qubit Pauli error
            labels. Plain labels such as ``"XX"`` are post-gate Pauli branches;
            labels prefixed by ``"*"`` such as ``"*XX"`` are replacement
            branches that omit the ideal two-qubit gate before applying the
            Pauli. Values must sum to 1.0; ``p2`` remains the total two-qubit
            error rate.
        p2_replacement_approximation: Approximation used for starred
            replacement labels. ``"pauli_twirl_omitted_gate"`` convolves with
            the omitted two-qubit gate's Pauli twirl; ``"branch_impact"``
            evaluates starred entries as replacement branch impacts;
            ``"exact_branch_replay"`` uses the traced circuit context to replay
            omitted-gate branches at concrete two-qubit gate locations and
            fails loudly when a branch is not DEM-representable;
            ``"ignore_gate_removal"`` treats starred entries like plain
            post-gate Pauli entries.
        p_meas: Measurement error rate.
        p_prep: Initialization error rate.
        p_idle: Idle noise rate per time unit (uniform depolarizing).
        t1: T1 relaxation time for idle noise (same units as idle duration).
        t2: T2 dephasing time (must satisfy t2 <= 2*t1).
        p_idle_linear_rate: Legacy alias for stochastic Z-memory rate linear in idle duration.
        p_idle_quadratic_rate: Legacy alias for stochastic Z-memory rate quadratic in idle duration.
        p_idle_x_linear_rate: Stochastic X-memory rate linear in idle duration.
        p_idle_y_linear_rate: Stochastic Y-memory rate linear in idle duration.
        p_idle_z_linear_rate: Stochastic Z-memory rate linear in idle duration.
        p_idle_x_quadratic_rate: Stochastic X-memory rate quadratic in idle duration.
        p_idle_y_quadratic_rate: Stochastic Y-memory rate quadratic in idle duration.
        p_idle_z_quadratic_rate: Stochastic Z-memory rate quadratic in idle duration.
        p_idle_quadratic_sine_rate: Legacy alias for stochastic Z-memory rate
            with probability ``sin(rate * duration)^2``.
        p_idle_x_quadratic_sine_rate: Stochastic X-memory sine-law rate.
        p_idle_y_quadratic_sine_rate: Stochastic Y-memory sine-law rate.
        p_idle_z_quadratic_sine_rate: Stochastic Z-memory sine-law rate.
    """

    p1: float = 0.0
    p1_weights: P1Weights | None = None
    p2: float = 0.0
    p2_szz: float | None = None
    p2_szzdg: float | None = None
    p2_weights: P2Weights | None = None
    p2_replacement_approximation: str | None = None
    p_meas: float = 0.0
    p_prep: float = 0.0
    p_idle: float | None = None
    t1: float | None = None
    t2: float | None = None
    p_idle_linear_rate: float | None = None
    p_idle_quadratic_rate: float | None = None
    p_idle_x_linear_rate: float | None = None
    p_idle_y_linear_rate: float | None = None
    p_idle_z_linear_rate: float | None = None
    p_idle_x_quadratic_rate: float | None = None
    p_idle_y_quadratic_rate: float | None = None
    p_idle_z_quadratic_rate: float | None = None
    p_idle_quadratic_sine_rate: float | None = None
    p_idle_x_quadratic_sine_rate: float | None = None
    p_idle_y_quadratic_sine_rate: float | None = None
    p_idle_z_quadratic_sine_rate: float | None = None

    def __post_init__(self) -> None:
        """Normalize cache-sensitive inputs after dataclass initialization."""
        self.p1_weights = _normalize_p1_weights(self.p1_weights)
        self.p2_weights = _normalize_p2_weights(self.p2_weights)
        if self.p2_szz is not None:
            self.p2_szz = _validate_probability("p2_szz", self.p2_szz)
        if self.p2_szzdg is not None:
            self.p2_szzdg = _validate_probability("p2_szzdg", self.p2_szzdg)

    @property
    def effective_p_idle_z_linear_rate(self) -> float | None:
        """Z-axis linear idle rate, accepting the legacy alias."""
        return self.p_idle_z_linear_rate if self.p_idle_z_linear_rate is not None else self.p_idle_linear_rate

    @property
    def effective_p_idle_z_quadratic_rate(self) -> float | None:
        """Z-axis quadratic idle rate, accepting the legacy alias."""
        return self.p_idle_z_quadratic_rate if self.p_idle_z_quadratic_rate is not None else self.p_idle_quadratic_rate

    @property
    def effective_p_idle_z_quadratic_sine_rate(self) -> float | None:
        """Z-axis sine-law quadratic idle rate, accepting the legacy alias."""
        if self.p_idle_z_quadratic_sine_rate is not None:
            return self.p_idle_z_quadratic_sine_rate
        return self.p_idle_quadratic_sine_rate

    @property
    def idle_memory_rates(self) -> tuple[float | None, ...]:
        """All dedicated Pauli idle-memory rates that require explicit idles."""
        return (
            self.p_idle_x_linear_rate,
            self.p_idle_y_linear_rate,
            self.effective_p_idle_z_linear_rate,
            self.p_idle_x_quadratic_rate,
            self.p_idle_y_quadratic_rate,
            self.effective_p_idle_z_quadratic_rate,
            self.p_idle_x_quadratic_sine_rate,
            self.p_idle_y_quadratic_sine_rate,
            self.effective_p_idle_z_quadratic_sine_rate,
        )

    @property
    def p2_gate_rates(self) -> tuple[float | None, ...]:
        """Explicit two-qubit gate-rate overrides."""
        return (self.p2_szz, self.p2_szzdg)

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
            and all(rate is None or rate == 0.0 for rate in self.p2_gate_rates)
            and self.p_meas == 0.0
            and self.p_prep == 0.0
            and (self.p_idle is None or self.p_idle == 0.0)
            and all(rate is None or rate == 0.0 for rate in self.idle_memory_rates)
        )

    @property
    def physical_error_rate(self) -> float:
        """Approximate combined physical error rate."""
        rates = [self.p1, self.p2, self.p_meas, self.p_prep]
        rates.extend(rate for rate in self.p2_gate_rates if rate is not None)
        if self.p_idle is not None:
            rates.append(self.p_idle)
        rates.extend(rate for rate in self.idle_memory_rates if rate is not None)
        return max(rates)


def _normalize_pauli_weights(weights: P1Weights | P2Weights | None) -> tuple[tuple[str, float], ...] | None:
    if weights is None:
        return None
    items = weights.items() if isinstance(weights, Mapping) else weights
    return tuple(sorted((str(label).upper(), float(weight)) for label, weight in items))


def _normalize_p1_weights(p1_weights: P1Weights | None) -> tuple[tuple[str, float], ...] | None:
    return _normalize_pauli_weights(p1_weights)


def _p1_weights_dict(p1_weights: P1Weights | None) -> dict[str, float] | None:
    normalized = _normalize_p1_weights(p1_weights)
    return None if normalized is None else dict(normalized)


def _normalize_p2_weights(p2_weights: P2Weights | None) -> tuple[tuple[str, float], ...] | None:
    return _normalize_pauli_weights(p2_weights)


def _p2_weights_dict(p2_weights: P2Weights | None) -> dict[str, float] | None:
    normalized = _normalize_p2_weights(p2_weights)
    return None if normalized is None else dict(normalized)


def _p2_gate_rates_dict(noise: NoiseModel) -> dict[str, float] | None:
    rates: dict[str, float] = {}
    if noise.p2_szz is not None:
        rates["SZZ"] = noise.p2_szz
    if noise.p2_szzdg is not None:
        rates["SZZdg"] = noise.p2_szzdg
    return rates or None


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

    dag_circuit: Any
    influence_map: Any
    szz_physical_prefixes: bool
    z_frame_gate_p1_free: bool
    pauli_frame_lookup: Any | None
    detectors_json: str
    observables_json: str
    measurement_order: tuple[int, ...]
    num_measurements: int
    num_detectors: int
    num_observables: int
    num_pauli_sites: int
    interaction_basis: str
    check_plan: str
    resolved_check_plan: dict[str, Any]
    resolved_check_plan_hash: str


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


def _abstract_twirl_config(twirl: TwirlConfig | None) -> TwirlConfig | None:
    """Drop runtime-only Guppy record-framing fields before DEM caching."""
    if twirl is None:
        return None
    twirl.validate_runtime_supported()
    return replace(twirl, frame_output="raw", twirl_probability=1.0)


def _twirl_traced_qis_rejection_message() -> str:
    return (
        "twirl=TwirlConfig() is not supported with circuit_source='traced_qis': "
        "tracing runtime RNG twirl would bake one concrete mask realization "
        "into the circuit/DEM/lookup, and canonical frame_output may break "
        "runtime measurement result-id provenance because measurement tags can "
        "be XOR-derived expressions. Use circuit_source='abstract' for twirl "
        "for now."
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


def _copy_surface_tick_circuit_metadata(
    source_tc: Any,
    target_tc: Any,
    *,
    measurement_index_remap: dict[int, int] | None = None,
) -> None:
    """Copy the surface-level metadata needed by the native DEM/sampler builders."""
    num_measurements_text = source_tc.get_meta("num_measurements")
    num_measurements = int(num_measurements_text) if num_measurements_text is not None else None

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
            if measurement_index_remap is not None and key in (
                "detectors",
                "observables",
                "detector_descriptors",
                "observable_descriptors",
            ):
                if num_measurements is None:
                    msg = "Cannot remap surface metadata without num_measurements"
                    raise ValueError(msg)
                value = _remap_surface_record_metadata_json(
                    value,
                    measurement_index_remap=measurement_index_remap,
                    num_measurements=num_measurements,
                )
            target_tc.set_meta(key, value)


def _measurement_index_remap_for_orders(
    abstract_measurement_order: list[int],
    traced_measurement_order: list[int],
) -> dict[int, int]:
    """Map abstract record indices to runtime-traced record indices.

    The detector metadata is generated from the abstract surface schedule, but
    a runtime may legally reorder measurement operations while preserving the
    same measured qubit occurrences. This helper binds each measurement by
    ``(qubit, occurrence_count_for_that_qubit)`` so metadata can follow a pure
    scheduling reorder without accepting dropped/extra/wrong measurements.
    """
    from collections import Counter, defaultdict

    if len(abstract_measurement_order) != len(traced_measurement_order) or Counter(
        abstract_measurement_order,
    ) != Counter(traced_measurement_order):
        msg = (
            "Traced and abstract surface circuits disagree on the measured-qubit "
            "multiset; refusing to remap detector/observable metadata"
        )
        raise ValueError(msg)

    traced_occurrences: dict[tuple[int, int], int] = {}
    traced_counts: defaultdict[int, int] = defaultdict(int)
    for traced_index, qubit in enumerate(traced_measurement_order):
        occurrence = traced_counts[qubit]
        traced_occurrences[(qubit, occurrence)] = traced_index
        traced_counts[qubit] += 1

    remap: dict[int, int] = {}
    abstract_counts: defaultdict[int, int] = defaultdict(int)
    for abstract_index, qubit in enumerate(abstract_measurement_order):
        occurrence = abstract_counts[qubit]
        remap[abstract_index] = traced_occurrences[(qubit, occurrence)]
        abstract_counts[qubit] += 1

    return remap


def _remap_surface_record_metadata_json(
    metadata_json: str,
    *,
    measurement_index_remap: dict[int, int],
    num_measurements: int,
) -> str:
    """Bind abstract measurement refs to runtime-stable ``meas_ids``.

    ``measurement_index_remap`` maps abstract measurement indices to the
    stable result ids emitted by the runtime trace. Those ids are not
    positional record offsets, so remapped runtime metadata must use
    ``meas_ids`` and must drop stale ``records``.
    """
    import json

    entries = json.loads(metadata_json)
    for entry in entries:
        records = entry.pop("records", None)
        if records is not None:
            abstract_indices = []
            for record in records:
                abstract_index = num_measurements + int(record)
                if abstract_index not in measurement_index_remap:
                    msg = f"Surface metadata record {record!r} is out of range for remapping"
                    raise ValueError(msg)
                abstract_indices.append(abstract_index)
        elif "meas_ids" in entry:
            abstract_indices = [int(meas_id) for meas_id in entry["meas_ids"]]
        else:
            continue

        remapped_meas_ids = []
        for abstract_index in abstract_indices:
            if abstract_index not in measurement_index_remap:
                msg = f"Surface metadata meas_id {abstract_index!r} is out of range for remapping"
                raise ValueError(msg)
            remapped_meas_ids.append(int(measurement_index_remap[abstract_index]))
        entry["meas_ids"] = remapped_meas_ids
    return json.dumps(entries)


def _surface_runtime_measurement_remap_from_result_traces(
    abstract_tc: Any,
    result_traces: list[dict[str, Any]],
) -> dict[int, int]:
    """Map abstract surface measurement indices to runtime ``result_id``s.

    The generated surface Guppy emits scalar counted-round
    ``result("sx*/sz*:meas:N", bit)`` tags, prep-boundary
    ``result("sx*/sz*:init:meas:N", bit)`` tags, and one
    ``result("final", array(...))`` call for data readout. The abstract
    TickCircuit labels each measurement with the result tag it should bind to.
    Those tags survive runtime scheduling changes and are the stable
    detector/observable anchor.
    """
    num_measurements = int(abstract_tc.get_meta("num_measurements"))
    scalar_trace_ids, array_trace_ids = _index_surface_result_trace_ids(result_traces)
    abstract_refs = _surface_abstract_measurement_result_refs(abstract_tc)
    if len(abstract_refs) != num_measurements:
        msg = f"expected {num_measurements} abstract measurement refs, got {len(abstract_refs)}"
        raise ValueError(msg)

    occurrence_by_tag: dict[str, int] = {}
    remap: dict[int, int] = {}
    for abstract_index, ref in enumerate(abstract_refs):
        if ref[0] == "scalar":
            _, name = ref
            occurrence = occurrence_by_tag.get(name, 0)
            occurrence_by_tag[name] = occurrence + 1
            try:
                remap[abstract_index] = scalar_trace_ids[name][occurrence]
            except (KeyError, IndexError) as exc:
                msg = f"result tag {name!r} occurrence {occurrence} is missing from the runtime trace"
                raise ValueError(msg) from exc
        else:
            _, name, element = ref
            try:
                remap[abstract_index] = array_trace_ids[name][0][element]
            except (KeyError, IndexError) as exc:
                msg = f"result tag {name!r}[{element}] is missing from the runtime trace"
                raise ValueError(msg) from exc

    runtime_ids = sorted(remap.values())
    if runtime_ids != list(range(num_measurements)):
        msg = (
            "Runtime result-tag provenance is not a dense measurement-id range "
            f"0..{num_measurements - 1}; got first/last "
            f"{runtime_ids[:3]}...{runtime_ids[-3:]}"
        )
        raise ValueError(msg)
    return remap


def _index_surface_result_trace_ids(
    result_traces: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, list[int]], dict[str, list[list[int]]]]:
    """Index runtime named-result provenance by tag name."""
    scalar_trace_ids: dict[str, list[int]] = {}
    array_trace_ids: dict[str, list[list[int]]] = {}
    for trace in result_traces:
        name = trace.get("name")
        values = trace.get("values")
        result_ids = trace.get("result_ids")
        if not isinstance(name, str) or not isinstance(values, list) or not isinstance(result_ids, list):
            continue
        if _is_surface_sideband_result_tag(name):
            continue
        if len(values) != len(result_ids):
            msg = (
                f"runtime result tag {name!r} has {len(values)} value(s) but "
                f"{len(result_ids)} result id(s); cannot bind surface metadata"
            )
            raise ValueError(msg)
        ids = [int(result_id) for result_id in result_ids]
        is_scalar_syndrome_tag = name.startswith(("sx", "sz")) and ":meas:" in name
        if is_scalar_syndrome_tag and len(ids) == 1:
            scalar_trace_ids.setdefault(name, []).append(ids[0])
        else:
            array_trace_ids.setdefault(name, []).append(ids)
    if not scalar_trace_ids and not array_trace_ids:
        msg = "runtime trace does not contain named_result_traces; rebuild PECOS with result-tag provenance support"
        raise ValueError(msg)
    return scalar_trace_ids, array_trace_ids


def _is_surface_sideband_result_tag(name: str) -> bool:
    """Return true for non-detector-bearing surface result tags."""
    return name.startswith(("pauli_mask:", "pauli_active:", "frame_mode:", "raw:"))


def _surface_abstract_measurement_result_refs(abstract_tc: Any) -> list[tuple[str, str] | tuple[str, str, int]]:
    """Return the result-tag reference for each abstract surface measurement."""
    refs: list[tuple[str, str] | tuple[str, str, int]] = []
    syndrome_measure_index_by_round: dict[int, int] = {}
    measurement_gate_types = {"MZ", "MeasureFree"}
    for tick_index in range(abstract_tc.num_ticks()):
        tick = abstract_tc.get_tick(tick_index)
        if tick is None:
            continue
        for gate_index, gate in enumerate(tick.gate_batches()):
            gate_type = str(getattr(gate, "gate_type", "")).rsplit(".", maxsplit=1)[-1]
            if gate_type not in measurement_gate_types:
                continue
            label = str(abstract_tc.get_gate_meta(tick_index, gate_index, "label") or "")
            if label.startswith(("sx", "sz")):
                round_value = abstract_tc.get_gate_meta(tick_index, gate_index, "syndrome_round")
                if round_value is None:
                    msg = f"surface syndrome measurement {label!r} is missing syndrome_round metadata"
                    raise ValueError(msg)
                round_index = int(round_value)
                measurement_index = syndrome_measure_index_by_round.get(round_index, 0)
                syndrome_measure_index_by_round[round_index] = measurement_index + 1
                phase = "init:meas" if round_index < 0 else "meas"
                refs.append(("scalar", f"{label}:{phase}:{measurement_index}"))
                continue
            if label.startswith("final[") and label.endswith("]"):
                refs.append(("array", "final", int(label.removeprefix("final[").removesuffix("]"))))
                continue
            msg = f"surface measurement is missing a result-tag-compatible label: {label!r}"
            raise ValueError(msg)
    return refs


def _extract_measurement_meas_ids(tc: Any) -> list[int]:
    """Return stable measurement ids in TickCircuit execution order."""
    ids: list[int] = []
    for tick_idx in range(tc.num_ticks()):
        tick = tc.get_tick(tick_idx)
        if tick is None:
            continue
        for gate in tick.gate_batches():
            gate_type = str(getattr(gate, "gate_type", "")).rsplit(".", maxsplit=1)[-1]
            if gate_type not in {"MZ", "MeasureFree"}:
                continue
            qubits = list(getattr(gate, "qubits", []))
            meas_ids = list(getattr(gate, "meas_ids", []))
            if len(meas_ids) != len(qubits):
                msg = (
                    f"traced measurement gate {gate_type} in tick {tick_idx} carries "
                    f"{len(meas_ids)} MeasId(s) for {len(qubits)} qubit(s)"
                )
                raise ValueError(msg)
            ids.extend(int(meas_id) for meas_id in meas_ids)
    return ids


def _validate_result_tag_remap_against_traced_measurements(
    traced_tc: Any,
    measurement_index_remap: Mapping[int, int],
    *,
    expected_measurements: int,
) -> None:
    """Fail loudly unless result-tag bindings exactly cover traced MeasIds."""
    expected_abstract_indices = list(range(expected_measurements))
    actual_abstract_indices = sorted(measurement_index_remap)
    if actual_abstract_indices != expected_abstract_indices:
        msg = (
            "runtime result-tag remap does not cover every abstract measurement; "
            f"expected indices {expected_abstract_indices[:3]}...{expected_abstract_indices[-3:]}, "
            f"got {actual_abstract_indices[:3]}...{actual_abstract_indices[-3:]}"
        )
        raise ValueError(msg)

    traced_meas_ids = _extract_measurement_meas_ids(traced_tc)
    if len(traced_meas_ids) != expected_measurements:
        msg = (
            "traced circuit contains "
            f"{len(traced_meas_ids)} measured MeasId(s), but result-tag metadata "
            f"expects {expected_measurements}"
        )
        raise ValueError(msg)
    if len(set(traced_meas_ids)) != len(traced_meas_ids):
        duplicates = sorted(meas_id for meas_id in set(traced_meas_ids) if traced_meas_ids.count(meas_id) > 1)
        msg = f"traced circuit contains duplicate measured MeasId(s): {duplicates[:8]}"
        raise ValueError(msg)

    expected_meas_ids = sorted(int(meas_id) for meas_id in measurement_index_remap.values())
    actual_meas_ids = sorted(traced_meas_ids)
    if actual_meas_ids != expected_meas_ids:
        expected_set = set(expected_meas_ids)
        actual_set = set(actual_meas_ids)
        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)
        msg = (
            "runtime result-tag bindings do not exactly match the traced circuit's "
            f"measured MeasIds; missing={missing[:8]}, extra={extra[:8]}"
        )
        raise ValueError(msg)


def _runtime_idle_seconds_to_time_units(duration_seconds: float) -> Any:
    """Convert runtime idle seconds into PECOS nanosecond time units."""
    import math

    from pecos_rslib import TimeUnits

    if not math.isfinite(duration_seconds) or duration_seconds < 0.0:
        msg = f"Idle duration must be finite and non-negative, got {duration_seconds!r}"
        raise ValueError(msg)

    units = round(duration_seconds * 1_000_000_000.0)
    if duration_seconds > 0.0:
        units = max(1, units)
    return TimeUnits(units)


def _validate_measurement_crosstalk_topology(
    measurement_crosstalk_topology: str | None,
) -> str | None:
    if measurement_crosstalk_topology in (None, "none", "runtime_payloads"):
        return None
    if measurement_crosstalk_topology == "global_from_measurements":
        return measurement_crosstalk_topology
    msg = "measurement_crosstalk_topology must be None, 'runtime_payloads', or 'global_from_measurements'"
    raise ValueError(msg)


def _should_add_global_measurement_crosstalk_payload(
    measurement_crosstalk_topology: str | None,
) -> bool:
    return _validate_measurement_crosstalk_topology(measurement_crosstalk_topology) == "global_from_measurements"


def _replay_qis_trace_into_tick_circuit(
    operations: list[dict[str, Any]],
    *,
    measurement_crosstalk_topology: str | None = None,
) -> Any:
    """Replay traced QIS operations into a PECOS TickCircuit."""
    import heapq

    from pecos_rslib.quantum import TickCircuit

    measurement_crosstalk_topology = _validate_measurement_crosstalk_topology(
        measurement_crosstalk_topology,
    )
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
        elif op_name == "Idle":
            duration, program_id = tuple_args(payload, op_name, 2)
            tick.idle(
                _runtime_idle_seconds_to_time_units(float(duration)),
                [mapped_slot(int(program_id), op_name)],
            )
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
            measurement_qubit = mapped_slot(int(program_id), op_name)
            if _should_add_global_measurement_crosstalk_payload(
                measurement_crosstalk_topology,
            ):
                # Global crosstalk payload qubits are guaranteed not to be
                # affected; for measurement-induced global crosstalk this is
                # exactly the measured payload.
                tick_circuit.tick().add_gate(
                    "MeasCrosstalkGlobalPayload",
                    [measurement_qubit],
                )
            # Stamp the QIS-provided result_id as the MeasId rather than
            # discarding it and letting assign_missing_meas_ids() invent
            # sequential ids (which would be wrong for non-sequential ids).
            tick.mz_with_ids(
                [measurement_qubit],
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


def _lowered_gate_metadata(gate: Mapping[str, Any]) -> dict[str, Any]:
    """Return validated runtime/source metadata for a lowered trace gate."""
    metadata = gate.get("metadata")
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        msg = f"Lowered gate metadata must be an object, got {metadata!r}"
        raise TypeError(msg)
    return {str(key): value for key, value in metadata.items()}


def _set_lowered_gate_metadata(tick: Any, metadata: Mapping[str, Any]) -> None:
    """Attach lowered trace metadata to the gate most recently added to ``tick``."""
    if not metadata:
        return
    tick.metas(metadata)


def _replay_lowered_qis_trace_into_tick_circuit(
    chunks: list[dict[str, Any]],
    *,
    measurement_crosstalk_topology: str | None = None,
) -> Any:
    """Replay lowered post-Selene ByteMessage gate batches into a TickCircuit.

    The lowered trace emits gates one at a time. We replay each into its own
    tick, then compact (ASAP schedule) so that gates on disjoint qubits share
    a tick --- matching the parallel structure of the abstract circuit.

    MeasIds flow from runtime-lowered measurement provenance:
    ``lowered_quantum_ops`` MZ entries must carry ``measurement_result_ids``.
    This avoids inferring lowered measurement IDs from raw QIS operation order,
    which is not stable under runtime scheduling or transport.
    """
    from pecos_rslib.quantum import TickCircuit

    measurement_crosstalk_topology = _validate_measurement_crosstalk_topology(
        measurement_crosstalk_topology,
    )
    tick_circuit = TickCircuit()

    for chunk in chunks:
        for gate in chunk.get("lowered_quantum_ops") or []:
            gate_type = str(gate["gate_type"])
            qubits = [int(q) for q in gate.get("qubits", [])]
            angles = [float(theta) for theta in gate.get("angles", [])]
            params = [float(param) for param in gate.get("params", [])]
            metadata = _lowered_gate_metadata(gate)
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
            elif gate_type == "Idle":
                if len(params) != 1:
                    msg = f"Lowered Idle gate expected one duration param, got {params!r}"
                    raise ValueError(msg)
                tick.idle(_runtime_idle_seconds_to_time_units(params[0]), qubits)
            elif gate_type == "MZ":
                meas_ids = gate.get("measurement_result_ids")
                if not isinstance(meas_ids, list):
                    msg = (
                        "Lowered MZ trace is missing measurement_result_ids; "
                        "rebuild PECOS so runtime-lowered measurements carry "
                        "their result-id provenance instead of relying on "
                        "operation-order inference."
                    )
                    raise ValueError(msg)
                if len(meas_ids) != len(qubits):
                    msg = f"Lowered MZ gate carries {len(meas_ids)} measurement_result_ids for {len(qubits)} qubit(s)"
                    raise ValueError(msg)
                if _should_add_global_measurement_crosstalk_payload(
                    measurement_crosstalk_topology,
                ):
                    # Global crosstalk payload qubits are guaranteed not to be
                    # affected; for measurement-induced global crosstalk this is
                    # exactly the measured payload.
                    tick_circuit.tick().add_gate(
                        "MeasCrosstalkGlobalPayload",
                        qubits,
                    )
                tick.mz_with_ids(qubits, [int(meas_id) for meas_id in meas_ids])
            elif gate_type == "MeasCrosstalkGlobalPayload":
                tick.add_gate("MeasCrosstalkGlobalPayload", qubits)
            elif gate_type == "MeasCrosstalkLocalPayload":
                tick.add_gate("MeasCrosstalkLocalPayload", qubits)
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
            _set_lowered_gate_metadata(tick, metadata)

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


def _replay_qis_trace_chunks_into_tick_circuit(
    chunks: list[dict[str, Any]],
    *,
    measurement_crosstalk_topology: str | None = None,
) -> Any:
    """Replay captured QIS operation trace chunks into a ``TickCircuit``."""
    measurement_crosstalk_topology = _validate_measurement_crosstalk_topology(
        measurement_crosstalk_topology,
    )
    if any(chunk.get("lowered_quantum_ops") for chunk in chunks):
        _reject_partially_lowered_trace(chunks)
        try:
            return _replay_lowered_qis_trace_into_tick_circuit(
                chunks,
                measurement_crosstalk_topology=measurement_crosstalk_topology,
            )
        except ValueError as exc:
            if "missing measurement_result_ids" not in str(exc):
                raise
            # Older local Selene/qis-compiler builds can emit lowered gates
            # without measurement_result_ids while still carrying the raw QIS
            # operations, whose Measure payloads include the stable result ids.
            # Replay the raw operations in that compatibility case instead of
            # losing provenance.

    operations: list[dict[str, Any]] = []
    for chunk in chunks:
        operations.extend(list(chunk.get("operations", [])))
    return _replay_qis_trace_into_tick_circuit(
        operations,
        measurement_crosstalk_topology=measurement_crosstalk_topology,
    )


def named_result_traces_from_operation_trace(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return runtime `result(...)` provenance records from operation trace chunks."""
    traces: list[dict[str, Any]] = []
    for chunk in chunks:
        traces.extend(trace for trace in (chunk.get("named_result_traces") or []) if isinstance(trace, dict))
    return traces


def capture_guppy_operation_trace(
    program: Any,
    num_qubits: int,
    *,
    seed: int = 0,
    runtime: object | None = None,
) -> list[dict[str, Any]]:
    """Capture a Guppy/QIS program's Selene operation trace chunks."""
    import pecos_rslib

    import pecos

    # Trace capture records the runtime-lowered QIS operations and result tags;
    # DEM validation/fault propagation happens after replay.  Use a permissive
    # trace backend instead of asking stabilizer evolution to validate every
    # runtime-emitted rotation while we are only collecting provenance.
    sim_builder = (
        pecos.sim(program)
        .classical(pecos.selene_engine(runtime))
        .quantum(pecos_rslib.coin_toss())
        .qubits(num_qubits)
        .seed(seed)
    )
    return list(sim_builder.capture_operation_trace())


def trace_guppy_into_tick_circuit_with_result_traces(
    program: Any,
    num_qubits: int,
    *,
    seed: int = 0,
    runtime: object | None = None,
    measurement_crosstalk_topology: str | None = None,
    require_hosted_operation_order: bool = False,
    max_hosted_tick_separation: int | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Trace a Guppy/QIS program into a ``TickCircuit`` plus result-tag provenance."""
    chunks = capture_guppy_operation_trace(program, num_qubits, seed=seed, runtime=runtime)
    tick_circuit = _replay_qis_trace_chunks_into_tick_circuit(
        chunks,
        measurement_crosstalk_topology=measurement_crosstalk_topology,
    )
    _validate_trace_hosted_operations_if_requested(
        tick_circuit,
        require_hosted_operation_order=require_hosted_operation_order,
        max_hosted_tick_separation=max_hosted_tick_separation,
        context="trace_guppy_into_tick_circuit_with_result_traces",
    )
    return tick_circuit, named_result_traces_from_operation_trace(chunks)


def trace_guppy_into_tick_circuit(
    program: Any,
    num_qubits: int,
    *,
    seed: int = 0,
    runtime: object | None = None,
    measurement_crosstalk_topology: str | None = None,
    require_hosted_operation_order: bool = False,
    max_hosted_tick_separation: int | None = None,
) -> Any:
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
        runtime: Optional Selene runtime selector/plugin. ``None`` selects the
            default Selene runtime. Runtime plugin objects are passed through to
            ``pecos.selene_engine(runtime)``.
        measurement_crosstalk_topology: Optional measurement-crosstalk replay
            mode for stamping global measurement-crosstalk payload markers.
        require_hosted_operation_order: If true, validate generic hosted-operation
            metadata after trace replay. A gate with ``local_role`` metadata
            must bind to a later same-``host_id`` host gate sharing a qubit.
            This catches runtime/compiler lowering that reorders hosted local
            pulses after the operation they semantically prepare.
        max_hosted_tick_separation: Optional maximum absolute signed tick
            separation accepted by the hosted-operation validator.

    Returns:
        A ``TickCircuit`` with no detector/observable metadata attached; the
        caller supplies that.
    """
    chunks = capture_guppy_operation_trace(program, num_qubits, seed=seed, runtime=runtime)
    tick_circuit = _replay_qis_trace_chunks_into_tick_circuit(
        chunks,
        measurement_crosstalk_topology=measurement_crosstalk_topology,
    )
    _validate_trace_hosted_operations_if_requested(
        tick_circuit,
        require_hosted_operation_order=require_hosted_operation_order,
        max_hosted_tick_separation=max_hosted_tick_separation,
        context="trace_guppy_into_tick_circuit",
    )
    return tick_circuit


def _validate_trace_hosted_operations_if_requested(
    tick_circuit: Any,
    *,
    require_hosted_operation_order: bool,
    max_hosted_tick_separation: int | None,
    context: str,
) -> None:
    if not require_hosted_operation_order and max_hosted_tick_separation is None:
        return
    validate_hosted_operations(
        tick_circuit,
        max_tick_separation=max_hosted_tick_separation,
        require_host_after_local=require_hosted_operation_order,
        require_unique_host_id=True,
        context=context,
    )


def _generate_traced_surface_tick_circuit(
    patch: SurfacePatch,
    num_rounds: int,
    basis: str,
    *,
    ancilla_budget: int | None = None,
    interaction_basis: str | None = None,
    check_plan: str | None = None,
    runtime: object | None = None,
    clifford_frame_policy: str | None = None,
    szz_runtime_barriers: bool | str = False,
    require_hosted_operation_order: bool = False,
    max_hosted_tick_separation: int | None = None,
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
    tc, _ = _generate_traced_surface_tick_circuit_with_result_traces(
        patch,
        num_rounds,
        basis,
        ancilla_budget=ancilla_budget,
        interaction_basis=interaction_basis,
        check_plan=check_plan,
        runtime=runtime,
        clifford_frame_policy=clifford_frame_policy,
        szz_runtime_barriers=szz_runtime_barriers,
        require_hosted_operation_order=require_hosted_operation_order,
        max_hosted_tick_separation=max_hosted_tick_separation,
    )
    return tc


def _generate_traced_surface_tick_circuit_with_result_traces(
    patch: SurfacePatch,
    num_rounds: int,
    basis: str,
    *,
    ancilla_budget: int | None = None,
    interaction_basis: str | None = None,
    check_plan: str | None = None,
    runtime: object | None = None,
    clifford_frame_policy: str | None = None,
    szz_runtime_barriers: bool | str = False,
    require_hosted_operation_order: bool = False,
    max_hosted_tick_separation: int | None = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Trace a surface Guppy program into a ``TickCircuit`` plus result provenance."""
    from pecos.guppy.surface import generate_memory_experiment, get_num_qubits

    program = generate_memory_experiment(
        patch,
        num_rounds,
        basis,
        ancilla_budget=ancilla_budget,
        interaction_basis=interaction_basis,
        check_plan=check_plan,
        clifford_frame_policy=clifford_frame_policy,
        szz_runtime_barriers=szz_runtime_barriers,
    )
    return trace_guppy_into_tick_circuit_with_result_traces(
        program,
        get_num_qubits(
            patch=patch,
            ancilla_budget=ancilla_budget,
            interaction_basis=interaction_basis,
            check_plan=check_plan,
            clifford_frame_policy=clifford_frame_policy,
        ),
        seed=0,
        runtime=runtime,
        require_hosted_operation_order=require_hosted_operation_order,
        max_hosted_tick_separation=max_hosted_tick_separation,
    )


def _build_surface_tick_circuit_for_native_model(
    patch: SurfacePatch,
    num_rounds: int,
    basis: str,
    *,
    ancilla_budget: int | None = None,
    circuit_source: Literal["abstract", "traced_qis"] = "abstract",
    runtime: object | None = None,
    twirl: TwirlConfig | None = None,
    interaction_basis: str | None = None,
    check_plan: str | None = None,
    szz_physical_prefixes: bool = False,
    clifford_frame_policy: str | None = None,
    szz_runtime_barriers: bool | str = False,
    require_hosted_operation_order: bool = False,
    max_hosted_tick_separation: int | None = None,
) -> Any:
    """Build the TickCircuit used by the native DEM and sampler paths."""
    from pecos.qec.surface.circuit_builder import _normalize_interaction_basis, generate_tick_circuit_from_patch

    if twirl is not None:
        twirl.validate_runtime_supported()
    resolved_plan = resolve_surface_check_plan(
        interaction_basis=interaction_basis,
        check_plan=check_plan,
    )
    require_current_surface_check_plan_renderer(
        resolved_plan,
        context="surface native TickCircuit generation",
    )
    interaction_basis = _normalize_interaction_basis(resolved_plan.interaction_basis)
    if szz_physical_prefixes and (interaction_basis != "szz" or circuit_source != "abstract"):
        msg = "SZZ physical-prefix lowering requires interaction_basis='szz' and circuit_source='abstract'"
        raise ValueError(msg)
    abstract_tc = generate_tick_circuit_from_patch(
        patch,
        num_rounds,
        basis,
        ancilla_budget=ancilla_budget,
        add_typed_annotations=False,
        twirl=twirl,
        interaction_basis=interaction_basis,
        check_plan=resolved_plan.plan_id,
        szz_physical_prefixes=szz_physical_prefixes,
        clifford_frame_policy=clifford_frame_policy,
    )

    if circuit_source == "abstract":
        return abstract_tc

    if circuit_source != "traced_qis":
        msg = f"Unknown circuit_source {circuit_source!r}"
        raise ValueError(msg)

    if twirl is not None:
        raise ValueError(_twirl_traced_qis_rejection_message())

    traced_tc, result_traces = _generate_traced_surface_tick_circuit_with_result_traces(
        patch,
        num_rounds,
        basis,
        ancilla_budget=ancilla_budget,
        interaction_basis=interaction_basis,
        check_plan=resolved_plan.plan_id,
        runtime=runtime,
        clifford_frame_policy=clifford_frame_policy,
        szz_runtime_barriers=szz_runtime_barriers,
        require_hosted_operation_order=require_hosted_operation_order,
        max_hosted_tick_separation=max_hosted_tick_separation,
    )

    measurement_index_remap = _surface_runtime_measurement_remap_from_result_traces(abstract_tc, result_traces)
    _validate_result_tag_remap_against_traced_measurements(
        traced_tc,
        measurement_index_remap,
        expected_measurements=int(abstract_tc.get_meta("num_measurements")),
    )
    _copy_surface_tick_circuit_metadata(
        abstract_tc,
        traced_tc,
        measurement_index_remap=measurement_index_remap,
    )
    traced_tc.set_meta("surface_metadata_record_binding", "runtime_result_tags")

    traced_tc.set_meta("circuit_source", circuit_source)
    return traced_tc


def _pauli_masks_as_int64(pauli_masks: Any) -> NDArray[np.int64]:
    """Return Pauli-mask input in the integer dtype accepted by Rust bindings."""
    masks_arr = np.asarray(pauli_masks)
    if not np.issubdtype(masks_arr.dtype, np.integer):
        msg = "pauli_masks must be an integer array with values 0=I, 1=X, 2=Y, 3=Z"
        raise TypeError(msg)
    return np.asarray(masks_arr, dtype=np.int64)


def build_memory_circuit(
    *,
    rounds: int,
    distance: int | None = None,
    patch: SurfacePatch | None = None,
    basis: str = "Z",
    ancilla_budget: int | None = None,
    circuit_source: Literal["abstract", "traced_qis"] = "abstract",
    runtime: object | None = None,
    twirl: TwirlConfig | None = None,
    interaction_basis: str | None = None,
    check_plan: str | None = None,
    clifford_frame_policy: str | None = None,
    szz_runtime_barriers: bool | str = False,
    require_hosted_operation_order: bool = False,
    max_hosted_tick_separation: int | None = None,
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
        runtime: Optional Selene runtime selector/plugin used when
            ``circuit_source="traced_qis"``.
        twirl: Optional Pauli-frame randomization layout. Currently supported
            only with ``circuit_source="abstract"``; traced-QIS twirl is
            rejected because a runtime trace would bake one sampled mask into
            the circuit and can lose canonical result-id provenance.
        interaction_basis: Surface-memory two-qubit interaction basis.
        check_plan: Named surface check-plan preset.
        clifford_frame_policy: Optional source-level Clifford-deformation
            policy for native abstract SZZ generation.
        szz_runtime_barriers: Optional SZZ/SZZdg runtime-barrier policy used
            for traced-QIS Guppy generation. This emits PECOS runtime barrier
            helpers between selected data-prefix pulses and their host
            SZZ/SZZdg operations.
        require_hosted_operation_order: For ``circuit_source="traced_qis"``,
            validate generic hosted-operation metadata after trace replay. A
            hosted local gate must appear before its same-``host_id`` host.
        max_hosted_tick_separation: Optional maximum absolute signed tick
            separation accepted by the hosted-operation validator.

    Returns:
        A Rust-backed ``TickCircuit`` with detector and observable metadata.

    Example:
        >>> from pecos.qec.surface import build_memory_circuit
        >>> tc = build_memory_circuit(distance=3, rounds=3, basis="Z")
        >>> int(tc.get_meta("num_measurements")) > 0
        True
    """
    from pecos.qec.surface.patch import SurfacePatch

    if rounds < 0:
        msg = f"rounds must be >= 0, got {rounds}"
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
        runtime=runtime,
        twirl=twirl,
        interaction_basis=interaction_basis,
        check_plan=check_plan,
        clifford_frame_policy=clifford_frame_policy,
        szz_runtime_barriers=szz_runtime_barriers,
        require_hosted_operation_order=require_hosted_operation_order,
        max_hosted_tick_separation=max_hosted_tick_separation,
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
    p_idle_linear_rate: float | None = None,
    p_idle_quadratic_rate: float | None = None,
    p_idle_x_linear_rate: float | None = None,
    p_idle_y_linear_rate: float | None = None,
    p_idle_z_linear_rate: float | None = None,
    p_idle_x_quadratic_rate: float | None = None,
    p_idle_y_quadratic_rate: float | None = None,
    p_idle_z_quadratic_rate: float | None = None,
    p_idle_quadratic_sine_rate: float | None = None,
    p_idle_x_quadratic_sine_rate: float | None = None,
    p_idle_y_quadratic_sine_rate: float | None = None,
    p_idle_z_quadratic_sine_rate: float | None = None,
) -> bool:
    """Return True when noise parameters require explicit idle locations."""
    return (
        (p_idle is not None and p_idle > 0.0)
        or (t1 is not None and t2 is not None)
        or (p_idle_linear_rate is not None and p_idle_linear_rate > 0.0)
        or (p_idle_quadratic_rate is not None and p_idle_quadratic_rate != 0.0)
        or any(
            rate is not None and rate > 0.0
            for rate in (
                p_idle_x_linear_rate,
                p_idle_y_linear_rate,
                p_idle_z_linear_rate,
                p_idle_x_quadratic_rate,
                p_idle_y_quadratic_rate,
                p_idle_z_quadratic_rate,
                p_idle_quadratic_sine_rate,
                p_idle_x_quadratic_sine_rate,
                p_idle_y_quadratic_sine_rate,
                p_idle_z_quadratic_sine_rate,
            )
        )
    )


def _noise_uses_dedicated_idle_noise(noise: NoiseModel) -> bool:
    """Return True when this noise model requires explicit idle locations."""
    return _uses_dedicated_idle_noise(
        p_idle=noise.p_idle,
        t1=noise.t1,
        t2=noise.t2,
        p_idle_linear_rate=noise.p_idle_linear_rate,
        p_idle_quadratic_rate=noise.p_idle_quadratic_rate,
        p_idle_x_linear_rate=noise.p_idle_x_linear_rate,
        p_idle_y_linear_rate=noise.p_idle_y_linear_rate,
        p_idle_z_linear_rate=noise.p_idle_z_linear_rate,
        p_idle_x_quadratic_rate=noise.p_idle_x_quadratic_rate,
        p_idle_y_quadratic_rate=noise.p_idle_y_quadratic_rate,
        p_idle_z_quadratic_rate=noise.p_idle_z_quadratic_rate,
        p_idle_quadratic_sine_rate=noise.p_idle_quadratic_sine_rate,
        p_idle_x_quadratic_sine_rate=noise.p_idle_x_quadratic_sine_rate,
        p_idle_y_quadratic_sine_rate=noise.p_idle_y_quadratic_sine_rate,
        p_idle_z_quadratic_sine_rate=noise.p_idle_z_quadratic_sine_rate,
    )


def _reject_szz_unlowered_physical_noise(
    noise: NoiseModel,
    interaction_basis: str,
    circuit_source: Literal["abstract", "traced_qis"],
) -> None:
    """Reject SZZ surface DEM noise without well-defined gate locations."""
    if interaction_basis != "szz":
        return
    reasons: list[str] = []
    if _noise_uses_dedicated_idle_noise(noise) and circuit_source != "abstract":
        reasons.append("dedicated idle noise with circuit_source='traced_qis'")
    if not reasons:
        return
    joined = ", ".join(reasons)
    msg = (
        "interaction_basis='szz' surface DEM generation does not yet support "
        f"{joined} because idle noise needs explicit post-flow idle locations; "
        "use circuit_source='abstract' for dedicated idle noise"
    )
    raise ValueError(msg)


def _use_szz_physical_prefixes(
    noise: NoiseModel,
    interaction_basis: str,
    circuit_source: Literal["abstract", "traced_qis"],
) -> bool:
    return (
        interaction_basis == "szz"
        and circuit_source == "abstract"
        and (noise.p1 > 0.0 or _noise_uses_dedicated_idle_noise(noise))
    )


def _szz_z_frame_p1_gate_rates(topology: _CachedNativeSurfaceTopology) -> dict[str, float] | None:
    """Return virtual-Z frame p1 overrides for the staged SZZ device model.

    The current SZZ surface basis treats Z/SZ/SZdg frame updates as noiseless
    virtual operations. That device-model assumption is keyed from
    ``interaction_basis == "szz"`` in the staged API, so CX-vs-SZZ p1 location
    comparisons include this free-Z modeling choice as well as gate basis
    differences.
    """
    if not topology.z_frame_gate_p1_free:
        return None
    return {"Z": 0.0, "SZ": 0.0, "SZdg": 0.0}


def _with_noise_compat(
    builder: Any,
    noise: NoiseModel,
    *,
    p1_gate_rates: Mapping[str, float] | None = None,
) -> Any:
    """Call Rust ``with_noise`` using the richest signature this binding supports."""
    noise_kwargs = {
        "p_idle": noise.p_idle,
        "t1": noise.t1,
        "t2": noise.t2,
        "p_idle_linear_rate": noise.p_idle_linear_rate,
        "p_idle_quadratic_rate": noise.p_idle_quadratic_rate,
        "p_idle_x_linear_rate": noise.p_idle_x_linear_rate,
        "p_idle_y_linear_rate": noise.p_idle_y_linear_rate,
        "p_idle_z_linear_rate": noise.p_idle_z_linear_rate,
        "p_idle_x_quadratic_rate": noise.p_idle_x_quadratic_rate,
        "p_idle_y_quadratic_rate": noise.p_idle_y_quadratic_rate,
        "p_idle_z_quadratic_rate": noise.p_idle_z_quadratic_rate,
        "p_idle_quadratic_sine_rate": noise.p_idle_quadratic_sine_rate,
        "p_idle_x_quadratic_sine_rate": noise.p_idle_x_quadratic_sine_rate,
        "p_idle_y_quadratic_sine_rate": noise.p_idle_y_quadratic_sine_rate,
        "p_idle_z_quadratic_sine_rate": noise.p_idle_z_quadratic_sine_rate,
        "p1_weights": _p1_weights_dict(noise.p1_weights),
        "p2_weights": _p2_weights_dict(noise.p2_weights),
    }
    if p1_gate_rates is not None:
        noise_kwargs["p1_gate_rates"] = {str(gate): float(rate) for gate, rate in p1_gate_rates.items()}
    p2_gate_rates = _p2_gate_rates_dict(noise)
    if p2_gate_rates is not None:
        noise_kwargs["p2_gate_rates"] = p2_gate_rates
    if noise.p2_replacement_approximation is not None:
        noise_kwargs["p2_replacement_approximation"] = noise.p2_replacement_approximation

    try:
        return builder.with_noise(
            noise.p1,
            noise.p2,
            noise.p_meas,
            noise.p_prep,
            **noise_kwargs,
        )
    except TypeError as exc:
        unsupported = {
            key: value for key, value in noise_kwargs.items() if key not in {"p_idle", "t1", "t2"} and value is not None
        }
        if unsupported:
            msg = (
                "This pecos_rslib build does not support the requested advanced "
                f"surface noise options: {sorted(unsupported)}"
            )
            raise TypeError(msg) from exc
        return builder.with_noise(
            noise.p1,
            noise.p2,
            noise.p_meas,
            noise.p_prep,
            p_idle=noise.p_idle,
            t1=noise.t1,
            t2=noise.t2,
        )


def _surface_native_topology(
    patch_key: tuple[int, int, str, bool],
    num_rounds: int,
    basis: str,
    ancilla_budget: int | None,
    circuit_source: Literal["abstract", "traced_qis"],
    include_idle_gates: bool,
    *,
    runtime: object | None = None,
    twirl: TwirlConfig | None = None,
    interaction_basis: str | None = None,
    check_plan: str | None = None,
    szz_physical_prefixes: bool = False,
    clifford_frame_policy: str | None = None,
    szz_runtime_barriers: bool | str = False,
    require_hosted_operation_order: bool = False,
    max_hosted_tick_separation: int | None = None,
) -> _CachedNativeSurfaceTopology:
    """Build topology-only native analysis shared across noise parameters."""
    import json

    from pecos.qec.surface.circuit_builder import (
        _build_canonical_dem_influence_map,
        _extract_measurement_order,
        _metadata_record_offsets,
        _metadata_uses_record_offsets,
        normalize_traced_qis_tick_circuit,
    )

    resolved_plan = resolve_surface_check_plan(interaction_basis=interaction_basis, check_plan=check_plan)
    require_current_surface_check_plan_renderer(
        resolved_plan,
        context="surface native topology construction",
    )
    interaction_basis = resolved_plan.interaction_basis
    patch = _cached_surface_patch(patch_key)
    tc = _build_surface_tick_circuit_for_native_model(
        patch,
        num_rounds,
        basis,
        ancilla_budget=ancilla_budget,
        circuit_source=circuit_source,
        runtime=runtime,
        twirl=twirl,
        interaction_basis=interaction_basis,
        check_plan=resolved_plan.plan_id,
        szz_physical_prefixes=szz_physical_prefixes,
        clifford_frame_policy=clifford_frame_policy,
        szz_runtime_barriers=szz_runtime_barriers,
        require_hosted_operation_order=require_hosted_operation_order,
        max_hosted_tick_separation=max_hosted_tick_separation,
    )
    if circuit_source == "traced_qis":
        # Keep this surface helper aligned with DetectorErrorModel.from_guppy:
        # traced QIS emits parameterized Clifford rotations, while DEM
        # replacement-branch approximations operate on named Clifford gates.
        normalize_traced_qis_tick_circuit(tc, context="surface traced-QIS native topology")
    if include_idle_gates:
        # Insert idle gates only when the requested noise model includes a
        # dedicated idle channel. Otherwise inserted idle gates receive ordinary
        # one-qubit gate noise and change the explicit circuit-level DEM.
        tc.fill_idle_gates()

    dag = tc.to_dag_circuit()
    influence_map = _build_canonical_dem_influence_map(dag)

    detectors_json = tc.get_meta("detectors") or "[]"
    observables_json = tc.get_meta("observables") or "[]"
    measurement_order = (
        tuple(_extract_measurement_order(tc)) if _metadata_uses_record_offsets(detectors_json, observables_json) else ()
    )
    num_measurements = int(tc.get_meta("num_measurements") or str(len(measurement_order)))
    det_records = (
        [_metadata_record_offsets(detector, num_measurements) for detector in json.loads(detectors_json)]
        if detectors_json
        else []
    )
    obs_records = (
        [_metadata_record_offsets(observable, num_measurements) for observable in json.loads(observables_json)]
        if observables_json
        else []
    )

    pauli_frame_lookup = None
    num_pauli_sites = 0
    if twirl is not None:
        from pecos_rslib.qec import PauliFrameLookup

        pauli_frame_lookup = PauliFrameLookup.from_circuit(dag, det_records, obs_records)
        num_pauli_sites = pauli_frame_lookup.num_pauli_sites

    return _CachedNativeSurfaceTopology(
        dag_circuit=dag,
        influence_map=influence_map,
        szz_physical_prefixes=szz_physical_prefixes,
        # Staged SZZ device model: Z/SZ/SZdg frame updates are virtual and
        # receive no p1 noise. Keep CX-vs-SZZ p1 location comparisons scoped to
        # that asymmetric device assumption.
        z_frame_gate_p1_free=interaction_basis == "szz",
        pauli_frame_lookup=pauli_frame_lookup,
        detectors_json=detectors_json,
        observables_json=observables_json,
        measurement_order=measurement_order,
        num_measurements=num_measurements,
        num_detectors=len(det_records),
        num_observables=len(obs_records),
        num_pauli_sites=num_pauli_sites,
        interaction_basis=resolved_plan.interaction_basis,
        check_plan=resolved_plan.plan_id,
        resolved_check_plan=resolved_plan.resolved_metadata,
        resolved_check_plan_hash=resolved_plan.resolved_hash,
    )


@cache
def _cached_surface_native_topology(
    patch_key: tuple[int, int, str, bool],
    num_rounds: int,
    basis: str,
    ancilla_budget: int | None,
    circuit_source: Literal["abstract", "traced_qis"],
    include_idle_gates: bool,
    *,
    twirl: TwirlConfig | None = None,
    interaction_basis: str | None = None,
    check_plan: str | None = None,
    szz_physical_prefixes: bool = False,
    resolved_check_plan_hash: str = "",
    clifford_frame_policy: str | None = None,
    szz_runtime_barriers: bool | str = False,
    require_hosted_operation_order: bool = False,
    max_hosted_tick_separation: int | None = None,
) -> _CachedNativeSurfaceTopology:
    """Cache topology-only native analysis shared across noise parameters."""
    _ = resolved_check_plan_hash
    return _surface_native_topology(
        patch_key,
        num_rounds,
        basis,
        ancilla_budget,
        circuit_source,
        include_idle_gates,
        twirl=twirl,
        interaction_basis=interaction_basis,
        check_plan=check_plan,
        szz_physical_prefixes=szz_physical_prefixes,
        clifford_frame_policy=clifford_frame_policy,
        szz_runtime_barriers=szz_runtime_barriers,
        require_hosted_operation_order=require_hosted_operation_order,
        max_hosted_tick_separation=max_hosted_tick_separation,
    )


def _dem_string_from_cached_surface_topology(
    topology: _CachedNativeSurfaceTopology,
    noise: NoiseModel,
    *,
    decompose_errors: bool,
    dem_decomposition: NativeDemDecomposition = "source_graphlike",
) -> str:
    """Build a DEM string from cached topology and fresh noise parameters."""
    from pecos.qec import DemBuilder

    builder = _with_noise_compat(
        DemBuilder(topology.influence_map),
        noise,
        p1_gate_rates=_szz_z_frame_p1_gate_rates(topology),
    )
    if hasattr(builder, "with_exact_branch_replay_circuit"):
        builder = builder.with_exact_branch_replay_circuit(topology.dag_circuit)

    builder = builder.with_num_measurements(topology.num_measurements)
    if topology.measurement_order:
        builder = builder.with_measurement_order(list(topology.measurement_order))
    dem = (
        builder.with_detectors_json(topology.detectors_json)
        .with_observables_json(
            topology.observables_json,
        )
        .build_with_source_tracking()
    )
    if not decompose_errors:
        return dem.to_string()
    if dem_decomposition == "source_graphlike":
        source_graphlike = getattr(dem, "to_string_source_graphlike_decomposed", None)
        if source_graphlike is not None:
            return source_graphlike()
        return dem.to_string_decomposed()
    if dem_decomposition == "terminal_graphlike":
        terminal_graphlike = getattr(dem, "to_string_terminal_graphlike_decomposed", None)
        if terminal_graphlike is None:
            msg = "This pecos_rslib build does not support terminal graphlike DEM decomposition"
            raise RuntimeError(msg)
        return terminal_graphlike()
    msg = f"Unknown native DEM decomposition mode {dem_decomposition!r}"
    raise ValueError(msg)


@cache
def _cached_surface_native_dem_string(
    patch_key: tuple[int, int, str, bool],
    num_rounds: int,
    basis: str,
    ancilla_budget: int | None,
    circuit_source: Literal["abstract", "traced_qis"],
    p1: float,
    p1_weights: tuple[tuple[str, float], ...] | None,
    p2: float,
    p2_szz: float | None,
    p2_szzdg: float | None,
    p_meas: float,
    p_prep: float,
    decompose_errors: bool,
    dem_decomposition: NativeDemDecomposition = "source_graphlike",
    p2_weights: tuple[tuple[str, float], ...] | None = None,
    p2_replacement_approximation: str | None = None,
    p_idle: float | None = None,
    t1: float | None = None,
    t2: float | None = None,
    p_idle_linear_rate: float | None = None,
    p_idle_quadratic_rate: float | None = None,
    p_idle_x_linear_rate: float | None = None,
    p_idle_y_linear_rate: float | None = None,
    p_idle_z_linear_rate: float | None = None,
    p_idle_x_quadratic_rate: float | None = None,
    p_idle_y_quadratic_rate: float | None = None,
    p_idle_z_quadratic_rate: float | None = None,
    p_idle_quadratic_sine_rate: float | None = None,
    p_idle_x_quadratic_sine_rate: float | None = None,
    p_idle_y_quadratic_sine_rate: float | None = None,
    p_idle_z_quadratic_sine_rate: float | None = None,
    twirl: TwirlConfig | None = None,
    interaction_basis: str = "cx",
    check_plan: str | None = None,
    resolved_check_plan_hash: str = "",
    clifford_frame_policy: str | None = None,
    *,
    szz_runtime_barriers: bool | str = False,
    require_hosted_operation_order: bool = False,
    max_hosted_tick_separation: int | None = None,
) -> str:
    """Cache native DEM strings across callers for one topology + noise tuple."""
    _ = resolved_check_plan_hash
    include_idle_gates = _uses_dedicated_idle_noise(
        p_idle=p_idle,
        t1=t1,
        t2=t2,
        p_idle_linear_rate=p_idle_linear_rate,
        p_idle_quadratic_rate=p_idle_quadratic_rate,
        p_idle_x_linear_rate=p_idle_x_linear_rate,
        p_idle_y_linear_rate=p_idle_y_linear_rate,
        p_idle_z_linear_rate=p_idle_z_linear_rate,
        p_idle_x_quadratic_rate=p_idle_x_quadratic_rate,
        p_idle_y_quadratic_rate=p_idle_y_quadratic_rate,
        p_idle_z_quadratic_rate=p_idle_z_quadratic_rate,
        p_idle_quadratic_sine_rate=p_idle_quadratic_sine_rate,
        p_idle_x_quadratic_sine_rate=p_idle_x_quadratic_sine_rate,
        p_idle_y_quadratic_sine_rate=p_idle_y_quadratic_sine_rate,
        p_idle_z_quadratic_sine_rate=p_idle_z_quadratic_sine_rate,
    )
    szz_physical_prefixes = (
        interaction_basis == "szz" and circuit_source == "abstract" and (p1 > 0.0 or include_idle_gates)
    )
    topology = _cached_surface_native_topology(
        patch_key,
        num_rounds,
        basis,
        ancilla_budget,
        circuit_source,
        include_idle_gates,
        twirl=twirl,
        interaction_basis=interaction_basis,
        check_plan=check_plan,
        szz_physical_prefixes=szz_physical_prefixes,
        resolved_check_plan_hash=resolved_check_plan_hash,
        clifford_frame_policy=clifford_frame_policy,
        szz_runtime_barriers=szz_runtime_barriers,
        require_hosted_operation_order=require_hosted_operation_order,
        max_hosted_tick_separation=max_hosted_tick_separation,
    )
    return _dem_string_from_cached_surface_topology(
        topology,
        NoiseModel(
            p1=p1,
            p1_weights=p1_weights,
            p2=p2,
            p2_szz=p2_szz,
            p2_szzdg=p2_szzdg,
            p2_weights=p2_weights,
            p2_replacement_approximation=p2_replacement_approximation,
            p_meas=p_meas,
            p_prep=p_prep,
            p_idle=p_idle,
            t1=t1,
            t2=t2,
            p_idle_linear_rate=p_idle_linear_rate,
            p_idle_quadratic_rate=p_idle_quadratic_rate,
            p_idle_x_linear_rate=p_idle_x_linear_rate,
            p_idle_y_linear_rate=p_idle_y_linear_rate,
            p_idle_z_linear_rate=p_idle_z_linear_rate,
            p_idle_x_quadratic_rate=p_idle_x_quadratic_rate,
            p_idle_y_quadratic_rate=p_idle_y_quadratic_rate,
            p_idle_z_quadratic_rate=p_idle_z_quadratic_rate,
            p_idle_quadratic_sine_rate=p_idle_quadratic_sine_rate,
            p_idle_x_quadratic_sine_rate=p_idle_x_quadratic_sine_rate,
            p_idle_y_quadratic_sine_rate=p_idle_y_quadratic_sine_rate,
            p_idle_z_quadratic_sine_rate=p_idle_z_quadratic_sine_rate,
        ),
        decompose_errors=decompose_errors,
        dem_decomposition=dem_decomposition,
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
    from pecos.qec import ParsedDem

    if sampling_model == "dem":
        dem_str = _dem_string_from_cached_surface_topology(
            topology,
            noise,
            decompose_errors=True,
        )
        sampler = ParsedDem.from_string(dem_str).to_dem_sampler()
    elif sampling_model in ("influence_dem", "mnm"):
        from pecos.qec import DemSamplerBuilder

        sampler_builder = (
            _with_noise_compat(
                DemSamplerBuilder(topology.influence_map),
                noise,
                p1_gate_rates=_szz_z_frame_p1_gate_rates(topology),
            )
            .with_detectors_json(topology.detectors_json)
            .with_observables_json(topology.observables_json)
        )
        if topology.measurement_order:
            sampler_builder = sampler_builder.with_measurement_order(list(topology.measurement_order))
        sampler = sampler_builder.build()
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
        pauli_frame_lookup=topology.pauli_frame_lookup,
        num_pauli_sites=topology.num_pauli_sites,
        sampling_model=sampling_model,
        interaction_basis=topology.interaction_basis,
        check_plan=topology.check_plan,
        resolved_check_plan=topology.resolved_check_plan,
        resolved_check_plan_hash=topology.resolved_check_plan_hash,
    )


def generate_circuit_level_dem_from_builder(
    patch: SurfacePatch,
    num_rounds: int,
    noise: NoiseModel,
    basis: str = "Z",
    *,
    decompose_errors: bool = False,
    dem_decomposition: NativeDemDecomposition = "source_graphlike",
    ancilla_budget: int | None = None,
    circuit_source: Literal["abstract", "traced_qis"] = "abstract",
    runtime: object | None = None,
    twirl: TwirlConfig | None = None,
    interaction_basis: str | None = None,
    check_plan: str | None = None,
    clifford_frame_policy: str | None = None,
    szz_runtime_barriers: bool | str = False,
    require_hosted_operation_order: bool = False,
    max_hosted_tick_separation: int | None = None,
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
        decompose_errors: If True, return PECOS's native graphlike-decomposed
            DEM representation for graph decoders such as PyMatching. The
            decomposition is a lossy hyperedge-to-edge projection; it preserves
            correlated mechanism metadata with ``^`` separators where available,
            but it is not an exact raw DEM serialization.
        dem_decomposition: Which native graphlike projection to use when
            ``decompose_errors=True``. ``"source_graphlike"`` preserves the
            existing source-informed decomposition. ``"terminal_graphlike"``
            groups raw mechanisms first, then pairs only detector terminals
            present in each raw effect by coordinate distance. Both modes are
            decoder-facing approximations of raw hyperedge mechanisms.
        ancilla_budget: Optional cap on simultaneously live ancillas. When
            provided below the total stabilizer count, the native DEM is built
            from the same batched ancilla-reuse circuit family used by Guppy.
        circuit_source: Which ideal circuit to analyze for the native DEM path.
            ``"abstract"`` uses the existing high-level surface TickCircuit.
            ``"traced_qis"`` traces the lowered ideal Selene/QIS gate stream
            and replays that exact gate list into a TickCircuit before running
            native PECOS fault analysis.
        runtime: Optional Selene runtime selector/plugin used when
            ``circuit_source="traced_qis"``. Custom runtime topologies are not
            kept in PECOS's in-process topology cache because plugin objects
            can carry private mutable state.
        twirl: Optional Pauli-frame randomization layout. Canonical Guppy
            frame-output mode is normalized to the same abstract raw lookup
            and DEM topology.
        interaction_basis: Backward-compatible selector for the default
            ``check_plan`` of a two-qubit interaction basis.
        check_plan: Named surface check-plan preset. This is the source of
            truth when supplied; ``interaction_basis`` must agree if also
            supplied. The staged SZZ plan currently assumes a virtual-Z device
            model: Z/SZ/SZdg frame updates are p1-free. That is a device
            assumption keyed from the resolved plan, not a general claim about
            CX hardware.
        clifford_frame_policy: Optional source-level Clifford-deformation
            policy for native SZZ generation. For ``circuit_source="traced_qis"``,
            the Guppy program is generated from the same concrete deformed
            checks before runtime result tags are bound to surface metadata.
        szz_runtime_barriers: Optional SZZ/SZZdg runtime-barrier policy for
            traced-QIS Guppy generation.
        require_hosted_operation_order: For ``circuit_source="traced_qis"``,
            validate generic hosted-operation metadata after runtime trace
            replay. This is intended for source-local pulses that semantically
            prepare a later host operation, such as SZZ/SZZdg data prefixes.
        max_hosted_tick_separation: Optional maximum absolute signed tick
            separation accepted by the hosted-operation validator.

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
    twirl = _abstract_twirl_config(twirl)

    resolved_plan = resolve_surface_check_plan(interaction_basis=interaction_basis, check_plan=check_plan)
    interaction_basis = resolved_plan.interaction_basis
    _reject_szz_unlowered_physical_noise(noise, interaction_basis, circuit_source)
    patch_key = _surface_patch_cache_key(patch)
    include_idle_gates = _noise_uses_dedicated_idle_noise(noise)
    szz_physical_prefixes = _use_szz_physical_prefixes(noise, interaction_basis, circuit_source)
    if runtime is not None:
        topology = _surface_native_topology(
            patch_key,
            num_rounds,
            basis.upper(),
            ancilla_budget,
            circuit_source,
            include_idle_gates,
            runtime=runtime,
            twirl=twirl,
            interaction_basis=interaction_basis,
            check_plan=resolved_plan.plan_id,
            szz_physical_prefixes=szz_physical_prefixes,
            clifford_frame_policy=clifford_frame_policy,
            szz_runtime_barriers=szz_runtime_barriers,
            require_hosted_operation_order=require_hosted_operation_order,
            max_hosted_tick_separation=max_hosted_tick_separation,
        )
        return _dem_string_from_cached_surface_topology(
            topology,
            noise,
            decompose_errors=decompose_errors,
            dem_decomposition=dem_decomposition,
        )

    cache_kwargs = {
        "p2_weights": noise.p2_weights,
        "p2_replacement_approximation": noise.p2_replacement_approximation,
        "p_idle": noise.p_idle,
        "t1": noise.t1,
        "t2": noise.t2,
        "p_idle_linear_rate": noise.p_idle_linear_rate,
        "p_idle_quadratic_rate": noise.p_idle_quadratic_rate,
        "p_idle_x_linear_rate": noise.p_idle_x_linear_rate,
        "p_idle_y_linear_rate": noise.p_idle_y_linear_rate,
        "p_idle_z_linear_rate": noise.p_idle_z_linear_rate,
        "p_idle_x_quadratic_rate": noise.p_idle_x_quadratic_rate,
        "p_idle_y_quadratic_rate": noise.p_idle_y_quadratic_rate,
        "p_idle_z_quadratic_rate": noise.p_idle_z_quadratic_rate,
        "p_idle_quadratic_sine_rate": noise.p_idle_quadratic_sine_rate,
        "p_idle_x_quadratic_sine_rate": noise.p_idle_x_quadratic_sine_rate,
        "p_idle_y_quadratic_sine_rate": noise.p_idle_y_quadratic_sine_rate,
        "p_idle_z_quadratic_sine_rate": noise.p_idle_z_quadratic_sine_rate,
        "twirl": twirl,
        "interaction_basis": interaction_basis,
        "check_plan": resolved_plan.plan_id,
        "resolved_check_plan_hash": resolved_plan.resolved_hash,
        "clifford_frame_policy": clifford_frame_policy,
        "szz_runtime_barriers": szz_runtime_barriers,
        "require_hosted_operation_order": require_hosted_operation_order,
        "max_hosted_tick_separation": max_hosted_tick_separation,
    }
    if dem_decomposition != "source_graphlike":
        cache_kwargs["dem_decomposition"] = dem_decomposition

    return _cached_surface_native_dem_string(
        patch_key,
        num_rounds,
        basis.upper(),
        ancilla_budget,
        circuit_source,
        noise.p1,
        noise.p1_weights,
        noise.p2,
        noise.p2_szz,
        noise.p2_szzdg,
        noise.p_meas,
        noise.p_prep,
        decompose_errors=decompose_errors,
        **cache_kwargs,
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
            "pymatching_correlated",
            "pymatching_uncorrelated",
            "fusion_blossom",
            "bp_osd",
            "bp_lsd",
            "union_find",
            "tesseract",
        ] = "pymatching",
        *,
        use_circuit_level_dem: bool = True,
        circuit_level_dem_mode: CircuitLevelDemMode = "native_full",
        circuit_level_dem_source: Literal["abstract", "traced_qis"] = "abstract",
        ancilla_budget: int | None = None,
        interaction_basis: str = "cx",
    ) -> None:
        """Initialize decoder from surface code patch.

        Args:
            patch: Surface code patch with geometry
            num_rounds: Number of syndrome extraction rounds
            noise: Noise model for edge weights (defaults to uniform)
            decoder_type: Decoder backend to use:
                - "pymatching": Fast C++ MWPM decoder (default). For
                  decomposed circuit-level DEMs, this enables PyMatching's
                  DEM-correlation metadata when available.
                - "pymatching_correlated": Explicit alias for the circuit-level
                  correlated PyMatching path.
                - "pymatching_uncorrelated": Plain graphlike PyMatching path,
                  useful for A/B diagnostics.
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
                returns the source-informed graphlike projection for graph
                decoders.
                ``"native_terminal_graphlike"`` first groups raw mechanisms,
                then projects each mechanism onto graphlike terminal components.
                Decomposed modes are lossy decoder-facing approximations of
                hyperedge correlations, not exact raw DEMs. Correlated graph
                decoding can use some preserved ``^`` metadata, but raw-DEM
                decoders should use ``"native_full"``.
            circuit_level_dem_source: Which ideal circuit to analyze when
                building native circuit-level DEMs. ``"abstract"`` uses the
                high-level surface TickCircuit, while ``"traced_qis"`` traces
                the lowered ideal Selene/QIS gate stream and analyzes that.
            ancilla_budget: Optional cap on simultaneously live ancillas for
                the native circuit-level DEM path. When provided, the decoder
                builds its DEM from the corresponding batched ancilla-reuse
                circuit instead of the default dedicated-ancilla circuit.
            interaction_basis: Surface-memory two-qubit interaction basis,
                ``"cx"`` or ``"szz"``. The staged ``"szz"`` path currently
                treats Z/SZ/SZdg frame updates as p1-free virtual operations.
        """
        from pecos.qec.surface.circuit_builder import _normalize_interaction_basis

        self.patch = patch
        self.num_rounds = num_rounds
        self.noise = noise or NoiseModel(p2=0.01, p_meas=0.01)
        self.decoder_type = DecoderType(decoder_type)
        self.use_circuit_level_dem = use_circuit_level_dem
        if circuit_level_dem_mode not in {
            "native_full",
            "native_decomposed",
            "native_terminal_graphlike",
        }:
            msg = f"Unknown circuit_level_dem_mode {circuit_level_dem_mode!r}"
            raise ValueError(msg)
        self.circuit_level_dem_mode = circuit_level_dem_mode
        self.circuit_level_dem_source = circuit_level_dem_source
        self.ancilla_budget = ancilla_budget
        self.interaction_basis = _normalize_interaction_basis(interaction_basis)

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
        dem_decomposition: NativeDemDecomposition = (
            "terminal_graphlike" if self.circuit_level_dem_mode == "native_terminal_graphlike" else "source_graphlike"
        )
        dem = generate_circuit_level_dem_from_builder(
            self.patch,
            self.num_rounds,
            self.noise,
            basis=basis,
            decompose_errors=self.circuit_level_dem_mode != "native_full",
            dem_decomposition=dem_decomposition,
            circuit_source=self.circuit_level_dem_source,
            ancilla_budget=self.ancilla_budget,
            interaction_basis=self.interaction_basis,
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
            if self.use_circuit_level_dem and self.decoder_type in DEM_DECODER_TYPES:
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

        if self.decoder_type in PYMATCHING_DECODER_TYPES:
            from pecos.decoders import CheckMatrix, PyMatchingDecoder

            if self.decoder_type == DecoderType.PYMATCHING_CORRELATED:
                msg = "pymatching_correlated requires circuit-level DEM decoding"
                raise ValueError(msg)

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
            from pecos.decoders import FusionBlossomDecoder

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

        if self.decoder_type in PYMATCHING_DECODER_TYPES:
            from pecos.decoders import PyMatchingDecoder

            if self.circuit_level_dem_mode == "native_full":
                if self.decoder_type == DecoderType.PYMATCHING_CORRELATED:
                    msg = "pymatching_correlated requires a decomposed circuit-level DEM mode"
                    raise ValueError(msg)
                return PyMatchingDecoder.from_dem(dem)

            if self.decoder_type == DecoderType.PYMATCHING_UNCORRELATED:
                return PyMatchingDecoder.from_dem(dem)

            return PyMatchingDecoder.from_dem_with_correlations(dem, enable_correlations=True)

        if self.decoder_type == DecoderType.TESSERACT:
            from pecos.decoders import TesseractDecoder

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
        from pecos.decoders import FusionBlossomDecoder

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
        from pecos.decoders import SparseMatrix

        sparse_H = SparseMatrix(H.tolist())

        if self.decoder_type == DecoderType.BP_OSD:
            from pecos.decoders import BpOsdBuilder

            return (
                BpOsdBuilder(sparse_H, error_rate=p_data)
                .max_iter(100)
                .bp_method("product_sum")
                .osd_method("osd0")
                .osd_order(0)
                .build()
            )

        if self.decoder_type == DecoderType.BP_LSD:
            from pecos.decoders import BpLsdBuilder

            return BpLsdBuilder(sparse_H, error_rate=p_data).max_iter(100).bp_method("product_sum").lsd_order(0).build()

        if self.decoder_type == DecoderType.UNION_FIND:
            from pecos.decoders import UnionFindBuilder

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
        from pecos.decoders import TesseractDecoder

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
            if self.use_circuit_level_dem and self.decoder_type in DEM_DECODER_TYPES:
                self._x_decoder = self._create_decoder_from_dem("X")
            else:
                self._x_decoder = self._create_decoder(self._get_x_check_matrix())
        return self._x_decoder

    def _is_mwpm_decoder(self) -> bool:
        """Check if using an MWPM or Tesseract decoder (vs LDPC)."""
        return self.decoder_type in (
            DecoderType.PYMATCHING,
            DecoderType.PYMATCHING_CORRELATED,
            DecoderType.PYMATCHING_UNCORRELATED,
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
        *,
        init_synx: NDArray[np.uint8] | None = None,
    ) -> NDArray[np.uint8]:
        """Compute full detection events for Z-basis DEM-based decoding.

        The circuit-level DEM defines detectors in this order:
        1. X stabilizer detectors for rounds 0..num_rounds-1
           (round 0 compares against the init X-syndrome baseline when present)
        2. Z stabilizer detectors for rounds 0..num_rounds-1
           (round 0 is deterministic for Z-basis)
        3. Final detectors: last known Z syndrome vs final data parity
           (for r=0, the known Z syndrome is the deterministic prep sign)

        Args:
            synx_list: X syndrome arrays, one per round
            synz_list: Z syndrome arrays, one per round
            final: Final data qubit measurements
            init_synx: Initial X-syndrome baseline measured during logical prep

        Returns:
            Detection events array matching the DEM detector ordering
        """
        geom = self.patch.geometry
        synx = np.array(synx_list, dtype=np.uint8)
        synz = np.array(synz_list, dtype=np.uint8)
        if self.num_rounds > 0 and init_synx is None:
            msg = (
                "Z-basis circuit-level DEM decoding requires init_synx, the prep-baseline "
                "X syndrome measured before counted syndrome-extraction rounds."
            )
            raise ValueError(msg)

        events: list[int] = []

        if self.num_rounds > 0:
            if init_synx is None:
                msg = "init_synx is required for Z-basis circuit-level DEM decoding"
                raise ValueError(msg)
            init_synx_array = np.array(init_synx, dtype=np.uint8)
            if init_synx_array.shape != synx[0].shape:
                msg = f"init_synx has shape {init_synx_array.shape}, expected {synx[0].shape}"
                raise ValueError(msg)

            # 1. X stabilizer detection events
            events.extend((synx[0] ^ init_synx_array).tolist())
            for r in range(1, self.num_rounds):
                events.extend((synx[r] ^ synx[r - 1]).tolist())

            # 2. Z stabilizer detection events (all rounds)
            events.extend(synz[0].tolist())  # round 0: compare to expected 0
            for r in range(1, self.num_rounds):
                events.extend((synz[r] ^ synz[r - 1]).tolist())

        # 3. Final readout: final Z parity XOR the last known Z syndrome.
        for stab in geom.z_stabilizers:
            data_parity = sum(int(final[q]) for q in stab.data_qubits) % 2
            last_syn = int(synz[-1][stab.index]) if self.num_rounds > 0 else 0
            events.append((data_parity ^ last_syn) & 1)

        return np.array(events, dtype=np.uint8)

    def _compute_dem_detection_events_x(
        self,
        synx_list: list[NDArray[np.uint8]],
        synz_list: list[NDArray[np.uint8]],
        final: NDArray[np.uint8],
        *,
        init_synz: NDArray[np.uint8] | None = None,
    ) -> NDArray[np.uint8]:
        """Compute full detection events for X-basis DEM-based decoding.

        The circuit-level DEM defines detectors in this order:
        1. X stabilizer detectors for rounds 0..num_rounds-1
           (X stabs are deterministic at round 0 for X-basis)
        2. Z stabilizer detectors for rounds 0..num_rounds-1
           (round 0 compares against the init Z-syndrome baseline when present)
        3. Final detectors: last known X syndrome vs final data parity
           (for r=0, the known X syndrome is the deterministic prep sign)

        Args:
            synx_list: X syndrome arrays, one per round
            synz_list: Z syndrome arrays, one per round
            final: Final data qubit measurements
            init_synz: Initial Z-syndrome baseline measured during logical prep

        Returns:
            Detection events array matching the DEM detector ordering
        """
        geom = self.patch.geometry
        synx = np.array(synx_list, dtype=np.uint8)
        synz = np.array(synz_list, dtype=np.uint8)
        if self.num_rounds > 0 and init_synz is None:
            msg = (
                "X-basis circuit-level DEM decoding requires init_synz, the prep-baseline "
                "Z syndrome measured before counted syndrome-extraction rounds."
            )
            raise ValueError(msg)

        events: list[int] = []

        if self.num_rounds > 0:
            if init_synz is None:
                msg = "init_synz is required for X-basis circuit-level DEM decoding"
                raise ValueError(msg)
            init_synz_array = np.array(init_synz, dtype=np.uint8)
            if init_synz_array.shape != synz[0].shape:
                msg = f"init_synz has shape {init_synz_array.shape}, expected {synz[0].shape}"
                raise ValueError(msg)

            # 1. X stabilizer detection events (all rounds)
            events.extend(synx[0].tolist())  # round 0: compare to expected 0
            for r in range(1, self.num_rounds):
                events.extend((synx[r] ^ synx[r - 1]).tolist())

            # 2. Z stabilizer detection events
            events.extend((synz[0] ^ init_synz_array).tolist())
            for r in range(1, self.num_rounds):
                events.extend((synz[r] ^ synz[r - 1]).tolist())

        # 3. Final readout: final X parity XOR the last known X syndrome.
        for stab in geom.x_stabilizers:
            data_parity = sum(int(final[q]) for q in stab.data_qubits) % 2
            last_syn = int(synx[-1][stab.index]) if self.num_rounds > 0 else 0
            events.append((data_parity ^ last_syn) & 1)

        return np.array(events, dtype=np.uint8)

    def decode_memory_z(
        self,
        synx_list: list[NDArray[np.uint8]],
        synz_list: list[NDArray[np.uint8]],
        final: NDArray[np.uint8],
        *,
        init_synx: NDArray[np.uint8] | None = None,
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
            init_synx: Optional prep-baseline X syndrome for the random
                stabilizer signs established before counted Z-memory rounds.

        Returns:
            (is_logical_error, decoding_result)
        """
        geom = self.patch.geometry
        logical_z_qubits = geom.logical_z.data_qubits if geom.logical_z else ()
        final_parity = sum(final[q] for q in logical_z_qubits) % 2

        # DEM-based path: compute full detection events matching DEM detector order
        if self.use_circuit_level_dem and self.decoder_type in DEM_DECODER_TYPES:
            events = self._compute_dem_detection_events_z(synx_list, synz_list, final, init_synx=init_synx)
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
        *,
        init_synz: NDArray[np.uint8] | None = None,
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
            init_synz: Optional prep-baseline Z syndrome for the random
                stabilizer signs established before counted X-memory rounds.

        Returns:
            (is_logical_error, decoding_result)
        """
        geom = self.patch.geometry
        logical_x_qubits = geom.logical_x.data_qubits if geom.logical_x else ()
        final_parity = sum(final[q] for q in logical_x_qubits) % 2

        # DEM-based path: compute full detection events matching DEM detector order
        if self.use_circuit_level_dem and self.decoder_type in DEM_DECODER_TYPES:
            events = self._compute_dem_detection_events_x(synx_list, synz_list, final, init_synz=init_synz)
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
        interaction_basis: Surface-memory two-qubit interaction basis.
        check_plan: Named surface check-plan preset.
        resolved_check_plan: Canonical resolved check-plan metadata.
        resolved_check_plan_hash: SHA-256 hash of the resolved plan semantics.
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
    interaction_basis: str = "cx"
    check_plan: str = "cx_standard_v1"
    resolved_check_plan: dict[str, Any] | None = None
    resolved_check_plan_hash: str = ""


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


def _recommended_graphlike_decomposition_for_decoder(decoder_type: str) -> NativeDemDecomposition:
    base = decoder_type.split(":", 1)[0]
    if base in {"pymatching", "pymatching_correlated", "pymatching_uncorrelated"}:
        return "terminal_graphlike"
    return "source_graphlike"


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
    interaction_basis: str | None = None,
    check_plan: str | None = None,
    clifford_frame_policy: str | None = None,
    szz_runtime_barriers: bool | str = False,
    require_hosted_operation_order: bool = False,
    max_hosted_tick_separation: int | None = None,
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
            PyMatching-family decoders use PECOS's terminal graphlike DEM
            projection for this recommended workflow.
        seed: Optional sampler seed.
        decode: If false, report the raw observable-flip rate.
        circuit_source: ``"abstract"`` or ``"traced_qis"`` circuit source.
        ancilla_budget: Optional cap on simultaneously live ancillas.
        interaction_basis: Backward-compatible selector for the default
            ``check_plan`` of a two-qubit interaction basis.
        check_plan: Named surface check-plan preset. This is the source of
            truth when supplied; ``interaction_basis`` must agree if also
            supplied.
        clifford_frame_policy: Optional source-level Clifford-deformation
            policy for native SZZ generation.
        szz_runtime_barriers: Optional SZZ/SZZdg runtime-barrier policy for
            traced-QIS Guppy generation.
        require_hosted_operation_order: For ``circuit_source="traced_qis"``,
            validate generic hosted-operation metadata after runtime trace
            replay.
        max_hosted_tick_separation: Optional maximum absolute signed tick
            separation accepted by the hosted-operation validator.

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

    resolved_plan = resolve_surface_check_plan(interaction_basis=interaction_basis, check_plan=check_plan)
    interaction_basis = resolved_plan.interaction_basis
    if distance < 1:
        msg = f"distance must be >= 1, got {distance}"
        raise ValueError(msg)
    if shots < 0:
        msg = f"shots must be >= 0, got {shots}"
        raise ValueError(msg)
    num_rounds = distance if rounds is None else rounds
    if num_rounds < 0:
        msg = f"rounds must be >= 0, got {num_rounds}"
        raise ValueError(msg)

    noise_model = _memory_noise_model(physical_error_rate, noise_model)
    patch = SurfacePatch.create(distance=distance)
    dem = generate_circuit_level_dem_from_builder(
        patch,
        num_rounds=num_rounds,
        noise=noise_model,
        basis=basis,
        decompose_errors=True,
        dem_decomposition=_recommended_graphlike_decomposition_for_decoder(decoder_type),
        ancilla_budget=ancilla_budget,
        circuit_source=circuit_source,
        interaction_basis=interaction_basis,
        check_plan=resolved_plan.plan_id,
        clifford_frame_policy=clifford_frame_policy,
        szz_runtime_barriers=szz_runtime_barriers,
        require_hosted_operation_order=require_hosted_operation_order,
        max_hosted_tick_separation=max_hosted_tick_separation,
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
        interaction_basis=interaction_basis,
        check_plan=resolved_plan.plan_id,
        resolved_check_plan=resolved_plan.resolved_metadata,
        resolved_check_plan_hash=resolved_plan.resolved_hash,
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
    interaction_basis: str | None = None,
    check_plan: str | None = None,
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
        interaction_basis: Backward-compatible selector for the default
            ``check_plan`` of a two-qubit interaction basis.
        check_plan: Named surface check-plan preset. This is the source of
            truth when supplied; ``interaction_basis`` must agree if also
            supplied.

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

    resolved_plan = resolve_surface_check_plan(interaction_basis=interaction_basis, check_plan=check_plan)
    interaction_basis = resolved_plan.interaction_basis
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
            interaction_basis=interaction_basis,
        )

    # Build and compile circuit
    num_qubits = get_num_qubits(distance, interaction_basis=interaction_basis)
    prog = make_surface_code(
        distance=distance,
        num_rounds=num_rounds,
        basis=basis,
        interaction_basis=interaction_basis,
    )
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
        interaction_basis=interaction_basis,
        check_plan=resolved_plan.plan_id,
        resolved_check_plan=resolved_plan.resolved_metadata,
        resolved_check_plan_hash=resolved_plan.resolved_hash,
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
        pauli_frame_lookup: Optional PECOS lookup for Pauli-twirl mask composition
        num_pauli_sites: Number of Pauli-twirl mask sites
        sampling_model: Which native sampling backend is active
        dem_string: Optional graphlike-decomposed DEM string used to build the
            sampler. Populated when the ``"dem"`` sampling model is selected.
        interaction_basis: Surface-memory two-qubit interaction basis resolved
            from ``check_plan``.
        check_plan: Named surface check-plan preset.
        resolved_check_plan: Canonical resolved check-plan metadata.
        resolved_check_plan_hash: SHA-256 hash of the resolved plan semantics.
    """

    sampler: Any
    detectors_json: str
    observables_json: str
    num_detectors: int
    num_observables: int
    pauli_frame_lookup: Any | None = None
    num_pauli_sites: int = 0
    sampling_model: Literal["dem", "influence_dem", "mnm"] = (
        "dem"  # "mnm" accepted for compat, mapped to "influence_dem"
    )
    dem_string: str | None = None
    interaction_basis: str = "cx"
    check_plan: str = "cx_standard_v1"
    resolved_check_plan: dict[str, Any] | None = None
    resolved_check_plan_hash: str = ""

    def sample(
        self,
        num_shots: int,
        seed: int | None = None,
        *,
        pauli_masks: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample detection events and observable flips.

        This matches Stim's DEM sampler output format.

        Args:
            num_shots: Number of shots to sample
            seed: Optional random seed for reproducibility
            pauli_masks: Optional integer array of shape
                ``(num_shots, num_pauli_sites)`` with values 0=I, 1=X, 2=Y,
                3=Z. Requires ``build_native_sampler(...,
                twirl=TwirlConfig())``.

        Returns:
            Tuple of (detection_events, observable_flips) as numpy arrays.
            - detection_events: shape (num_shots, num_detectors)
            - observable_flips: shape (num_shots, num_observables)
        """
        if pauli_masks is None:
            det_events, obs_flips = self.sampler.sample_batch(num_shots, seed)
        else:
            if self.pauli_frame_lookup is None:
                msg = "pauli_masks require build_native_sampler(..., twirl=TwirlConfig())"
                raise ValueError(msg)
            masks_arr = _pauli_masks_as_int64(pauli_masks)
            det_events, obs_flips = self.sampler.sample_batch_with_pauli_masks(
                num_shots,
                self.pauli_frame_lookup,
                masks_arr,
                seed,
            )
        return np.array(det_events, dtype=bool), np.array(obs_flips, dtype=bool)


def build_native_sampler(
    patch: SurfacePatch,
    num_rounds: int,
    noise: NoiseModel,
    basis: str = "Z",
    ancilla_budget: int | None = None,
    circuit_source: Literal["abstract", "traced_qis"] = "abstract",
    twirl: TwirlConfig | None = None,
    interaction_basis: str | None = None,
    sampling_model: Literal[
        "dem",
        "influence_dem",
        "mnm",
    ] = "dem",  # "mnm" accepted for compat, mapped to "influence_dem",
    check_plan: str | None = None,
    clifford_frame_policy: str | None = None,
    *,
    szz_runtime_barriers: bool | str = False,
    require_hosted_operation_order: bool = False,
    max_hosted_tick_separation: int | None = None,
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
        twirl: Optional Pauli-frame randomization layout. Canonical runtime
            frame-output mode is normalized to the same abstract raw lookup.
        interaction_basis: Backward-compatible selector for the default
            ``check_plan`` of a two-qubit interaction basis.
        sampling_model: Which native sampling backend to use. ``"dem"``
            samples the generated source-graphlike DEM projection and is the
            default; this is a decoder-facing approximation of raw hyperedges,
            not the exact raw DEM.
            ``"influence_dem"`` uses the influence-map-based DemSampler with
            detector definitions. ``"mnm"`` is accepted for compatibility
            and maps to ``"influence_dem"``.
        check_plan: Named surface check-plan preset. This is the source of
            truth when supplied; ``interaction_basis`` must agree if also
            supplied.
        clifford_frame_policy: Optional source-level Clifford-deformation
            policy for native abstract SZZ generation.
        szz_runtime_barriers: Optional SZZ/SZZdg runtime-barrier policy for
            traced-QIS Guppy generation.
        require_hosted_operation_order: For ``circuit_source="traced_qis"``,
            validate generic hosted-operation metadata after runtime trace
            replay.
        max_hosted_tick_separation: Optional maximum absolute signed tick
            separation accepted by the hosted-operation validator.

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
    twirl = _abstract_twirl_config(twirl)

    resolved_plan = resolve_surface_check_plan(interaction_basis=interaction_basis, check_plan=check_plan)
    interaction_basis = resolved_plan.interaction_basis
    _reject_szz_unlowered_physical_noise(noise, interaction_basis, circuit_source)
    basis = basis.upper()
    patch_key = _surface_patch_cache_key(patch)
    szz_physical_prefixes = _use_szz_physical_prefixes(noise, interaction_basis, circuit_source)
    topology = _cached_surface_native_topology(
        patch_key,
        num_rounds,
        basis,
        ancilla_budget,
        circuit_source,
        _noise_uses_dedicated_idle_noise(noise),
        twirl=twirl,
        interaction_basis=interaction_basis,
        check_plan=resolved_plan.plan_id,
        szz_physical_prefixes=szz_physical_prefixes,
        resolved_check_plan_hash=resolved_plan.resolved_hash,
        clifford_frame_policy=clifford_frame_policy,
        szz_runtime_barriers=szz_runtime_barriers,
        require_hosted_operation_order=require_hosted_operation_order,
        max_hosted_tick_separation=max_hosted_tick_separation,
    )
    if sampling_model == "dem":
        dem_str = _cached_surface_native_dem_string(
            patch_key,
            num_rounds,
            basis,
            ancilla_budget,
            circuit_source,
            noise.p1,
            noise.p1_weights,
            noise.p2,
            noise.p2_szz,
            noise.p2_szzdg,
            noise.p_meas,
            noise.p_prep,
            decompose_errors=True,
            p2_weights=noise.p2_weights,
            p2_replacement_approximation=noise.p2_replacement_approximation,
            p_idle=noise.p_idle,
            t1=noise.t1,
            t2=noise.t2,
            p_idle_linear_rate=noise.p_idle_linear_rate,
            p_idle_quadratic_rate=noise.p_idle_quadratic_rate,
            p_idle_x_linear_rate=noise.p_idle_x_linear_rate,
            p_idle_y_linear_rate=noise.p_idle_y_linear_rate,
            p_idle_z_linear_rate=noise.p_idle_z_linear_rate,
            p_idle_x_quadratic_rate=noise.p_idle_x_quadratic_rate,
            p_idle_y_quadratic_rate=noise.p_idle_y_quadratic_rate,
            p_idle_z_quadratic_rate=noise.p_idle_z_quadratic_rate,
            p_idle_quadratic_sine_rate=noise.p_idle_quadratic_sine_rate,
            p_idle_x_quadratic_sine_rate=noise.p_idle_x_quadratic_sine_rate,
            p_idle_y_quadratic_sine_rate=noise.p_idle_y_quadratic_sine_rate,
            p_idle_z_quadratic_sine_rate=noise.p_idle_z_quadratic_sine_rate,
            twirl=twirl,
            interaction_basis=interaction_basis,
            check_plan=resolved_plan.plan_id,
            resolved_check_plan_hash=resolved_plan.resolved_hash,
            clifford_frame_policy=clifford_frame_policy,
            szz_runtime_barriers=szz_runtime_barriers,
            require_hosted_operation_order=require_hosted_operation_order,
            max_hosted_tick_separation=max_hosted_tick_separation,
        )
        sampler = _cached_parsed_dem(dem_str).to_dem_sampler()
        return NativeSampler(
            sampler=sampler,
            detectors_json=topology.detectors_json,
            observables_json=topology.observables_json,
            num_detectors=topology.num_detectors,
            num_observables=topology.num_observables,
            pauli_frame_lookup=topology.pauli_frame_lookup,
            num_pauli_sites=topology.num_pauli_sites,
            sampling_model=sampling_model,
            dem_string=dem_str,
            interaction_basis=resolved_plan.interaction_basis,
            check_plan=resolved_plan.plan_id,
            resolved_check_plan=resolved_plan.resolved_metadata,
            resolved_check_plan_hash=resolved_plan.resolved_hash,
        )
    return _build_native_sampler_from_cached_surface_topology(
        topology,
        noise,
        sampling_model=sampling_model,
    )


def build_native_sampler_from_dem(
    decomposed_dem: str,
    patch: SurfacePatch,
    num_rounds: int,
    basis: str = "Z",
    *,
    ancilla_budget: int | None = None,
    circuit_source: Literal["abstract", "traced_qis"] = "abstract",
    twirl: TwirlConfig | None = None,
    interaction_basis: str | None = None,
    check_plan: str | None = None,
    clifford_frame_policy: str | None = None,
    szz_runtime_barriers: bool | str = False,
) -> NativeSampler:
    """Build a native sampler from a caller-supplied decomposed DEM string.

    The supplied DEM is parsed directly and stored verbatim on the returned
    sampler. Surface topology metadata and optional Pauli-frame lookup are
    taken from the same abstract/traced surface circuit family as
    :func:`build_native_sampler`.
    """
    ancilla_budget = _canonical_ancilla_budget(patch, ancilla_budget)
    twirl = _abstract_twirl_config(twirl)

    resolved_plan = resolve_surface_check_plan(interaction_basis=interaction_basis, check_plan=check_plan)
    interaction_basis = resolved_plan.interaction_basis
    basis = basis.upper()
    patch_key = _surface_patch_cache_key(patch)
    topology = _cached_surface_native_topology(
        patch_key,
        num_rounds,
        basis,
        ancilla_budget,
        circuit_source,
        include_idle_gates=False,
        twirl=twirl,
        interaction_basis=interaction_basis,
        check_plan=resolved_plan.plan_id,
        resolved_check_plan_hash=resolved_plan.resolved_hash,
        clifford_frame_policy=clifford_frame_policy,
        szz_runtime_barriers=szz_runtime_barriers,
    )
    sampler = _cached_parsed_dem(decomposed_dem).to_dem_sampler()
    return NativeSampler(
        sampler=sampler,
        detectors_json=topology.detectors_json,
        observables_json=topology.observables_json,
        num_detectors=topology.num_detectors,
        num_observables=topology.num_observables,
        pauli_frame_lookup=topology.pauli_frame_lookup,
        num_pauli_sites=topology.num_pauli_sites,
        sampling_model="dem",
        dem_string=decomposed_dem,
        interaction_basis=resolved_plan.interaction_basis,
        check_plan=resolved_plan.plan_id,
        resolved_check_plan=resolved_plan.resolved_metadata,
        resolved_check_plan_hash=resolved_plan.resolved_hash,
    )


def decode_native_samples(
    sampler: NativeSampler,
    num_shots: int,
    *,
    dem: str | None = None,
    decoder_type: str = "pymatching",
    seed: int | None = None,
    pauli_masks: Any | None = None,
) -> int:
    """Sample, optionally apply a known Pauli-frame mask, de-mask, and decode."""
    from pecos_rslib.qec import SampleBatch

    dem_str = dem if dem is not None else sampler.dem_string
    if dem_str is None:
        msg = "decode_native_samples requires a DEM string; pass dem= or build the sampler with sampling_model='dem'"
        raise ValueError(msg)

    masks_arr = _pauli_masks_as_int64(pauli_masks) if pauli_masks is not None else None
    det_events, obs_flips = sampler.sample(num_shots, seed=seed, pauli_masks=masks_arr)

    if masks_arr is not None:
        if sampler.pauli_frame_lookup is None:
            msg = "pauli_masks require build_native_sampler(..., twirl=TwirlConfig())"
            raise ValueError(msg)
        det_xor, obs_xor = sampler.pauli_frame_lookup.compute_mask_xor(masks_arr)
        det_events = np.asarray(det_events, dtype=bool) ^ np.asarray(det_xor, dtype=bool)
        obs_flips = np.asarray(obs_flips, dtype=bool) ^ np.asarray(obs_xor, dtype=bool)

    det_list = np.asarray(det_events, dtype=np.uint8).tolist()
    obs_arr = np.asarray(obs_flips, dtype=np.uint64)
    if obs_arr.ndim != 2:
        msg = f"expected obs_flips to be 2-D, got shape {obs_arr.shape}"
        raise ValueError(msg)
    weights = (1 << np.arange(obs_arr.shape[1], dtype=np.uint64)).astype(np.uint64)
    obs_masks = (obs_arr * weights).sum(axis=1).astype(np.uint64).tolist()
    batch = SampleBatch(det_list, obs_masks)
    return batch.decode_count(dem_str, decoder_type)


def demask_pauli_frame_records(
    pauli_frame_lookup: Any,
    raw_events: Any,
    raw_obs: Any,
    pauli_masks: Any,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Cancel known Pauli-frame mask flips from detector/observable records."""
    events_arr = np.asarray(raw_events, dtype=bool)
    obs_arr = np.asarray(raw_obs, dtype=bool)
    masks_arr = _pauli_masks_as_int64(pauli_masks)

    if events_arr.ndim != 2:
        msg = (
            f"raw_events must be 2-D of shape (num_shots, num_detectors); "
            f"got ndim={events_arr.ndim}, shape={events_arr.shape}"
        )
        raise ValueError(msg)
    if obs_arr.ndim != 2:
        msg = (
            f"raw_obs must be 2-D of shape (num_shots, num_observables); "
            f"got ndim={obs_arr.ndim}, shape={obs_arr.shape}"
        )
        raise ValueError(msg)
    if masks_arr.ndim != 2:
        msg = (
            f"pauli_masks must be 2-D of shape (num_shots, num_pauli_sites); "
            f"got ndim={masks_arr.ndim}, shape={masks_arr.shape}"
        )
        raise ValueError(msg)
    if events_arr.shape[0] != obs_arr.shape[0] or events_arr.shape[0] != masks_arr.shape[0]:
        msg = (
            "raw_events, raw_obs, and pauli_masks must have the same "
            f"num_shots; got {events_arr.shape[0]}, {obs_arr.shape[0]}, "
            f"{masks_arr.shape[0]}"
        )
        raise ValueError(msg)

    expected_det = pauli_frame_lookup.num_detectors
    expected_obs = pauli_frame_lookup.num_observables
    expected_sites = pauli_frame_lookup.num_pauli_sites
    if events_arr.shape[1] != expected_det:
        msg = f"raw_events width {events_arr.shape[1]} != pauli_frame_lookup.num_detectors {expected_det}"
        raise ValueError(msg)
    if obs_arr.shape[1] != expected_obs:
        msg = f"raw_obs width {obs_arr.shape[1]} != pauli_frame_lookup.num_observables {expected_obs}"
        raise ValueError(msg)
    if masks_arr.shape[1] != expected_sites:
        msg = f"pauli_masks width {masks_arr.shape[1]} != pauli_frame_lookup.num_pauli_sites {expected_sites}"
        raise ValueError(msg)

    det_xor, obs_xor = pauli_frame_lookup.compute_mask_xor(masks_arr)
    return (
        events_arr ^ np.asarray(det_xor, dtype=bool),
        obs_arr ^ np.asarray(obs_xor, dtype=bool),
    )


def _extract_pauli_masks_from_results(
    results: dict[str, Any],
    *,
    num_rounds: int,
    num_data: int,
    num_shots: int,
    patch: SurfacePatch | None = None,
    basis: str = "Z",
    twirl: TwirlConfig | None = None,
) -> NDArray[np.uint8]:
    """Reconstruct per-shot Pauli-mask codes from Guppy result tags."""
    from pecos.qec.surface._twirl_sites import (
        mask_col_for,
        mask_col_for_gate_operand,
        num_pauli_sites,
        num_pauli_sites_for_schedule,
        num_two_qubit_gate_twirl_sites,
        pauli_mask_gate_tag,
        pauli_mask_round_tag,
        site_idx_for_round,
    )

    site_schedule = "between_rounds" if twirl is None else twirl.site_schedule
    if site_schedule == "before_two_qubit_gate":
        if patch is None:
            msg = "patch is required to extract before_two_qubit_gate Pauli masks"
            raise ValueError(msg)
        n_twirl = num_two_qubit_gate_twirl_sites(
            patch,
            num_rounds=num_rounds,
            basis=basis,
        )
        out = np.zeros(
            (
                num_shots,
                num_pauli_sites_for_schedule(
                    patch,
                    num_rounds=num_rounds,
                    basis=basis,
                    site_schedule="before_two_qubit_gate",
                ),
            ),
            dtype=np.uint8,
        )
        for site in range(n_twirl):
            tag = pauli_mask_gate_tag(site)
            if tag not in results:
                msg = (
                    f"missing Pauli-mask result tag {tag!r} (expected {n_twirl} gate "
                    f"tags for num_rounds={num_rounds}, basis={basis!r}); "
                    "did the program run with gate-local twirl enabled?"
                )
                raise ValueError(msg)
            per_gate = results[tag]
            if len(per_gate) != num_shots:
                msg = f"Pauli-mask tag {tag!r}: got {len(per_gate)} shots, expected {num_shots} shots"
                raise ValueError(msg)

            bits = np.asarray(per_gate, dtype=np.uint8)
            if bits.ndim != 2 or bits.shape[1] != 4:
                msg = (
                    f"Pauli-mask tag {tag!r} array has shape {bits.shape}, expected "
                    f"({num_shots}, 4) = (num_shots, 2*gate_operands)"
                )
                raise ValueError(msg)
            lo = bits[:, 0::2]
            hi = bits[:, 1::2]
            packed = (lo + (hi << 1)).astype(np.uint8)
            for operand in range(2):
                out[:, mask_col_for_gate_operand(site, operand)] = packed[:, operand]
        active = _extract_pauli_activations_from_results(
            results,
            num_rounds=num_rounds,
            num_data=num_data,
            num_shots=num_shots,
            patch=patch,
            basis=basis,
            twirl=twirl,
        )
        if np.any((~active) & (out != 0)):
            msg = "malformed Pauli twirl bundle: inactive gate-local site recorded a non-identity Pauli"
            raise ValueError(msg)
        return out

    n_twirl = max(0, num_rounds - 1)
    bits_per_round = 2 * num_data
    out = np.zeros((num_shots, num_pauli_sites(num_rounds, num_data)), dtype=np.uint8)

    for r in range(n_twirl):
        tag = pauli_mask_round_tag(r)
        if tag not in results:
            msg = (
                f"missing Pauli-mask result tag {tag!r} (expected {n_twirl} round "
                f"tags for num_rounds={num_rounds}); did the program run with twirl enabled?"
            )
            raise ValueError(msg)
        per_round = results[tag]
        if len(per_round) != num_shots:
            msg = f"Pauli-mask tag {tag!r}: got {len(per_round)} shots, expected {num_shots} shots"
            raise ValueError(msg)

        bits = np.asarray(per_round, dtype=np.uint8)
        if bits.ndim != 2 or bits.shape[1] != bits_per_round:
            msg = (
                f"Pauli-mask tag {tag!r} array has shape {bits.shape}, expected "
                f"({num_shots}, {bits_per_round}) = (num_shots, 2*num_data)"
            )
            raise ValueError(msg)

        lo = bits[:, 0::2]
        hi = bits[:, 1::2]
        packed = (lo + (hi << 1)).astype(np.uint8)

        site = site_idx_for_round(r)
        for q in range(num_data):
            out[:, mask_col_for(site, q, num_data)] = packed[:, q]

    active = _extract_pauli_activations_from_results(
        results,
        num_rounds=num_rounds,
        num_data=num_data,
        num_shots=num_shots,
        patch=patch,
        basis=basis,
        twirl=twirl,
    )
    if np.any((~active) & (out != 0)):
        msg = "malformed Pauli twirl bundle: inactive round site recorded a non-identity Pauli"
        raise ValueError(msg)
    return out


def _extract_pauli_activations_from_results(
    results: dict[str, Any],
    *,
    num_rounds: int,
    num_data: int,
    num_shots: int,
    patch: SurfacePatch | None = None,
    basis: str = "Z",
    twirl: TwirlConfig | None = None,
) -> NDArray[np.bool_]:
    """Reconstruct per-shot twirl activation bits from Guppy result tags.

    Legacy `twirl_probability=1.0` bundles have no activation tags; they are
    interpreted as active at every site. Scaled-twirl bundles must carry explicit
    activation tags so skipped sites and active identity draws remain auditable.
    """
    from pecos.qec.surface._twirl_sites import (
        mask_col_for,
        mask_col_for_gate_operand,
        num_pauli_sites,
        num_pauli_sites_for_schedule,
        num_two_qubit_gate_twirl_sites,
        pauli_active_gate_tag,
        pauli_active_round_tag,
        site_idx_for_round,
    )

    site_schedule = "between_rounds" if twirl is None else twirl.site_schedule
    probability = 1.0 if twirl is None else float(twirl.twirl_probability)
    has_active_tags = any(str(name).startswith("pauli_active:") for name in results)
    require_active_tags = has_active_tags or probability != 1.0

    if site_schedule == "before_two_qubit_gate":
        if patch is None:
            msg = "patch is required to extract before_two_qubit_gate Pauli activations"
            raise ValueError(msg)
        n_twirl = num_two_qubit_gate_twirl_sites(
            patch,
            num_rounds=num_rounds,
            basis=basis,
        )
        out = np.ones(
            (
                num_shots,
                num_pauli_sites_for_schedule(
                    patch,
                    num_rounds=num_rounds,
                    basis=basis,
                    site_schedule="before_two_qubit_gate",
                ),
            ),
            dtype=bool,
        )
        if not require_active_tags:
            return out
        out[...] = False
        for site in range(n_twirl):
            tag = pauli_active_gate_tag(site)
            if tag not in results:
                msg = f"missing Pauli-activation result tag {tag!r}"
                raise ValueError(msg)
            per_gate = results[tag]
            if len(per_gate) != num_shots:
                msg = f"Pauli-activation tag {tag!r}: got {len(per_gate)} shots, expected {num_shots} shots"
                raise ValueError(msg)
            bits = np.asarray(per_gate, dtype=bool)
            if bits.ndim != 2 or bits.shape[1] != 2:
                msg = (
                    f"Pauli-activation tag {tag!r} array has shape {bits.shape}, "
                    f"expected ({num_shots}, 2) = (num_shots, gate_operands)"
                )
                raise ValueError(msg)
            for operand in range(2):
                out[:, mask_col_for_gate_operand(site, operand)] = bits[:, operand]
        return out

    n_twirl = max(0, num_rounds - 1)
    out = np.ones((num_shots, num_pauli_sites(num_rounds, num_data)), dtype=bool)
    if not require_active_tags:
        return out
    out[...] = False
    for r in range(n_twirl):
        tag = pauli_active_round_tag(r)
        if tag not in results:
            msg = f"missing Pauli-activation result tag {tag!r}"
            raise ValueError(msg)
        per_round = results[tag]
        if len(per_round) != num_shots:
            msg = f"Pauli-activation tag {tag!r}: got {len(per_round)} shots, expected {num_shots} shots"
            raise ValueError(msg)
        bits = np.asarray(per_round, dtype=bool)
        if bits.ndim != 2 or bits.shape[1] != num_data:
            msg = (
                f"Pauli-activation tag {tag!r} array has shape {bits.shape}, "
                f"expected ({num_shots}, {num_data}) = (num_shots, num_data)"
            )
            raise ValueError(msg)
        site = site_idx_for_round(r)
        for q in range(num_data):
            out[:, mask_col_for(site, q, num_data)] = bits[:, q]
    return out


def _sample_pauli_sideband_results_from_guppy(
    patch: SurfacePatch,
    *,
    num_rounds: int,
    num_shots: int,
    basis: str,
    twirl: TwirlConfig,
    rng: Any,
    ancilla_budget: int | None = None,
) -> dict[str, list[list[Any]]]:
    """Run the Guppy memory program with twirling and harvest side-band tags."""
    from selene_sim import SimpleRuntime, Stim, build

    from pecos.compilation_pipeline import compile_guppy_to_hugr
    from pecos.guppy.surface import generate_memory_experiment, get_num_qubits

    if twirl is None or rng is None:
        msg = "sample_pauli_masks_from_guppy requires both twirl and rng to be set"
        raise ValueError(msg)
    twirl.validate_runtime_supported()
    if num_rounds < 1:
        msg = f"num_rounds must be >= 1, got {num_rounds}"
        raise ValueError(msg)

    fn = generate_memory_experiment(
        patch,
        num_rounds=num_rounds,
        basis=basis,
        twirl=twirl,
        rng=rng,
        ancilla_budget=ancilla_budget,
    )

    hugr_bytes = compile_guppy_to_hugr(fn)
    instance = build(
        hugr_bytes,
        name=f"pauli_mask_d{patch.geometry.dx}_r{num_rounds}_{basis.lower()}",
    )
    num_qubits = get_num_qubits(
        patch=patch,
        ancilla_budget=ancilla_budget,
        twirl=twirl,
    )

    results: dict[str, list[list[Any]]] = {}
    for shot_results in instance.run_shots(
        simulator=Stim(random_seed=int(rng.seed)),
        n_qubits=num_qubits,
        n_shots=num_shots,
        runtime=SimpleRuntime(),
        n_processes=1,
    ):
        for name, values in shot_results:
            try:
                shot_value = list(values)
            except TypeError:
                shot_value = [values]
            if name.startswith(("pauli_mask:", "pauli_active:")):
                results.setdefault(name, []).append(shot_value)
    return results


def sample_pauli_masks_from_guppy(
    patch: SurfacePatch,
    *,
    num_rounds: int,
    num_shots: int,
    basis: str,
    twirl: TwirlConfig,
    rng: Any,
    ancilla_budget: int | None = None,
) -> NDArray[np.uint8]:
    """Run the Guppy memory program with twirling and harvest mask columns."""
    twirl.validate_runtime_supported()
    num_data = patch.geometry.num_data
    results = _sample_pauli_sideband_results_from_guppy(
        patch,
        num_rounds=num_rounds,
        num_shots=num_shots,
        basis=basis,
        twirl=twirl,
        rng=rng,
        ancilla_budget=ancilla_budget,
    )

    return _extract_pauli_masks_from_results(
        results,
        num_rounds=num_rounds,
        num_data=num_data,
        num_shots=num_shots,
        patch=patch,
        basis=basis,
        twirl=twirl,
    )


def sample_pauli_activations_from_guppy(
    patch: SurfacePatch,
    *,
    num_rounds: int,
    num_shots: int,
    basis: str,
    twirl: TwirlConfig,
    rng: Any,
    ancilla_budget: int | None = None,
) -> NDArray[np.bool_]:
    """Run the Guppy memory program with twirling and harvest activation bits."""
    twirl.validate_runtime_supported()
    num_data = patch.geometry.num_data
    results = _sample_pauli_sideband_results_from_guppy(
        patch,
        num_rounds=num_rounds,
        num_shots=num_shots,
        basis=basis,
        twirl=twirl,
        rng=rng,
        ancilla_budget=ancilla_budget,
    )

    return _extract_pauli_activations_from_results(
        results,
        num_rounds=num_rounds,
        num_data=num_data,
        num_shots=num_shots,
        patch=patch,
        basis=basis,
        twirl=twirl,
    )
