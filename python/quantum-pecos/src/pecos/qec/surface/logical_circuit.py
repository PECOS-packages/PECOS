# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Logical circuit builder for surface codes with transversal gates.

Generates PECOS TickCircuit circuits natively, with Stim circuit strings
derived via ``tick_circuit_to_stim()``. Supports:

- Memory experiments (syndrome extraction rounds)
- Transversal Hadamard (H on all data qubits, swaps X<->Z stabilizers)
- Transversal CNOT (CX between corresponding data qubits of two patches)
- Transversal SZ via gate teleportation (CX + |+Y> ancilla consumption)

Output formats:

- ``to_tick_circuit()`` -- PECOS TickCircuit (source of truth)
- ``to_dag_circuit()`` -- PECOS DagCircuit (for fault analysis)
- ``to_stim()`` -- Stim circuit string (derived from TickCircuit)
- ``build_dem()`` -- DEM via PECOS DagFaultAnalyzer (no Stim)
- ``build_decoder()`` -- integrated decoder pipeline

References:
- Geher et al., "Error-corrected Hadamard gate" (arXiv:2312.11605)
- Sahay et al., "Error correction of transversal CNOT" (arXiv:2408.01393)
- Serra-Peralta et al., "Decoding across transversal Clifford gates" (arXiv:2505.13599)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import zip_longest
from typing import TYPE_CHECKING

from pecos.qec.surface import gadgets
from pecos.qec.surface.circuit_builder import OpType, QubitAllocation

if TYPE_CHECKING:
    from pecos.qec.surface.circuit_builder import SurfaceCircuitStep
    from pecos.qec.surface.patch import Stabilizer, SurfacePatch

PatchSnapshot = dict[str, bool]


def _validate_boundary_cardinality(segments: list[object], boundary_gates: list[object]) -> None:
    """Require exactly one boundary list between consecutive segments."""
    expected_boundaries = len(segments) - 1
    if len(boundary_gates) != expected_boundaries:
        msg = (
            f"algorithm descriptor has {len(boundary_gates)} boundary gate lists and "
            f"{len(segments)} segments; expected exactly one boundary list between "
            "consecutive segments"
        )
        raise ValueError(msg)


class LogicalGateType(Enum):
    """Types of logical operations in a surface code circuit."""

    MEMORY = auto()
    TRANSVERSAL_H = auto()
    TRANSVERSAL_SZ = auto()
    TRANSVERSAL_SZdg = auto()
    TRANSVERSAL_CX = auto()


@dataclass
class PatchState:
    """Tracks the stabilizer assignment state of a patch.

    After transversal H, X-stabilizers become Z-stabilizers and vice versa.
    This state tracks which physical stabilizers are currently X-type vs Z-type,
    so that detectors can be formed correctly across gate boundaries.
    """

    patch: SurfacePatch
    label: str
    qubit_offset: int = 0
    coord_offset: tuple[float, float] = (0.0, 0.0)
    x_z_swapped: bool = False
    stabilizers_by_index: dict[str, dict[int, Stabilizer]] = field(default_factory=dict, init=False)

    @property
    def current_x_stabilizers(self) -> list[Stabilizer]:
        """Stabilizers currently measuring X-type checks."""
        if self.x_z_swapped:
            return self.patch.geometry.z_stabilizers
        return self.patch.geometry.x_stabilizers

    @property
    def current_z_stabilizers(self) -> list[Stabilizer]:
        """Stabilizers currently measuring Z-type checks."""
        if self.x_z_swapped:
            return self.patch.geometry.x_stabilizers
        return self.patch.geometry.z_stabilizers


@dataclass
class LogicalOp:
    """A logical operation in the circuit."""

    gate_type: LogicalGateType
    patches: list[str]
    rounds: int = 0
    basis: str = "Z"
    per_patch_basis: dict[str, str] = field(default_factory=dict)
    # Teleportation consumes the target as a correction readout, separate
    # from deterministic observables on the data.
    teleportation: bool = False
    # Type of magic state injection: "T" for T-gate, "SZ" for SZ, or None.
    # Used by build_algorithm_descriptor() to emit the correct boundary gate.
    injection_type: str | None = None


def _logical_readout_is_deterministic(
    operations: list[LogicalOp],
    segment_idx: int,
    patch: str,
    logical_type: str,
) -> bool:
    """Propagate a final logical readout backwards to product preparations.

    ``logical_type`` is X or Z in the readout's current orientation (its
    measurement basis). Each crossed H swaps the type during the walk; callers
    must not also swap the initial type. Physical S/S-dagger preserves Z but
    has no supported logical X image, including the unmodelled distance-1 case.
    A dict provides insertion-ordered XOR terms, so repeated CX images cancel.
    Terms involving a patch consumed before a crossed gate are unreliable.
    This deliberately conservative rule suppresses B and C for
    M([A,B,C],2,Z); CX(A,B); CX(B,C); M([B,C],2,Z), even though the simulator's
    non-destructive MZ makes them deterministic, because A is gated after its
    final readout, an invalid program rejected on the follow-up branch.
    """
    if logical_type not in {"X", "Z"}:
        msg = f"Unsupported logical readout type {logical_type!r}; expected X or Z"
        raise ValueError(msg)
    memory_indices = []
    first_memory = {}
    last_memory = {}
    for index, op in enumerate(operations):
        if op.gate_type == LogicalGateType.MEMORY:
            memory_indices.append(index)
            for label in op.patches:
                first_memory.setdefault(label, index)
                last_memory[label] = index
    if not 0 <= segment_idx < len(memory_indices):
        msg = f"No memory segment {segment_idx} for patch '{patch}'"
        raise ValueError(msg)
    start = memory_indices[segment_idx]
    if patch not in operations[start].patches:
        msg = f"Patch '{patch}' is not in memory segment {segment_idx}"
        raise ValueError(msg)

    terms = {(patch, logical_type): None}
    for index in range(start, -1, -1):
        op = operations[index]
        if op.gate_type == LogicalGateType.MEMORY:
            for label, kind in tuple(terms):
                if first_memory.get(label) == index:
                    if op.per_patch_basis.get(label, op.basis) != kind:
                        return False
                    del terms[label, kind]
            if not terms:
                return True
            continue

        images = {}
        for label, kind in terms:
            image = [(label, kind)]
            if op.gate_type == LogicalGateType.TRANSVERSAL_H and label in op.patches:
                image = [(label, "Z" if kind == "X" else "X")]
            elif op.gate_type == LogicalGateType.TRANSVERSAL_CX:
                ctrl, tgt = op.patches
                if label == ctrl and kind == "X":
                    image.append((tgt, "X"))
                elif label == tgt and kind == "Z":
                    image.append((ctrl, "Z"))
            elif (
                op.gate_type in {LogicalGateType.TRANSVERSAL_SZ, LogicalGateType.TRANSVERSAL_SZdg}
                and label in op.patches
                and kind == "X"
            ):
                return False
            for term in image:
                term_patch = term[0]
                if term_patch in op.patches and term_patch in last_memory and last_memory[term_patch] < index:
                    return False
                if term in images:
                    del images[term]
                else:
                    images[term] = None
        terms = images
        if not terms:
            return True

    msg = f"Logical readout for patch '{patch}' has terms without preparation: {list(terms)}"
    raise ValueError(msg)


class LogicalCircuitBuilder:
    """Builds surface code circuits with transversal gates.

    Composes logical operations on one or more patches, generating Stim
    circuits with correct detector annotations across gate boundaries.

    Example::

        patch = SurfacePatch.create(distance=3)
        builder = LogicalCircuitBuilder()
        builder.add_patch(patch, "A")
        builder.add_memory("A", rounds=3, basis="Z")
        builder.add_transversal_h("A")
        builder.add_memory("A", rounds=3, basis="X")
        stim_str = builder.to_stim(p1=0.001, p2=0.001)
    """

    def __init__(self) -> None:
        """Initialize an empty logical circuit builder."""
        self._patches: dict[str, PatchState] = {}
        self._operations: list[LogicalOp] = []
        self._consumed_injection_ancillas: set[str] = set()

    def add_patch(
        self,
        patch: SurfacePatch,
        label: str,
        qubit_offset: int = 0,
        coord_offset: tuple[float, float] | None = None,
    ) -> None:
        """Register a surface code patch.

        Args:
            patch: The surface code patch.
            label: Unique label for this patch.
            qubit_offset: Offset added to all qubit indices for this patch.
                Use this when multiple patches share a qubit index space.
            coord_offset: (dx, dy) spatial offset for this patch's
                QUBIT_COORDS and DETECTOR coordinates. If None, computed
                automatically based on patch index (patches are spaced
                apart so coordinates don't overlap).
        """
        if label in self._patches:
            msg = f"Patch '{label}' already registered"
            raise ValueError(msg)
        if coord_offset is None:
            # Auto-space: shift each patch by (d*2 + 2) * patch_index in x
            patch_idx = len(self._patches)
            spacing = patch.geometry.dz * 2 + 2
            coord_offset = (patch_idx * spacing, 0.0)
        self._patches[label] = PatchState(
            patch=patch,
            label=label,
            qubit_offset=qubit_offset,
            coord_offset=coord_offset,
        )

    def add_memory(
        self,
        patch_labels: str | list[str],
        rounds: int,
        basis: str | dict[str, str] = "Z",
    ) -> None:
        """Add syndrome extraction rounds for one or more patches.

        When multiple patches are given, their syndrome extraction runs
        in parallel (same time window).

        Args:
            patch_labels: Label(s) of the patch(es). String for single
                patch, list for parallel multi-patch.
            rounds: Number of syndrome extraction rounds.
            basis: Measurement basis. Either a single string ('X', 'Y', 'Z')
                applied to all patches, or a dict mapping patch labels to
                their individual basis (e.g., ``{"D": "Z", "Y": "Y"}``).
                Only used for initialization and final measurement.
        """
        if isinstance(patch_labels, str):
            patch_labels = [patch_labels]
        for label in patch_labels:
            self._require_available_patch(label)

        if isinstance(basis, str):
            default_basis = basis.upper()
            per_patch = {}
        else:
            default_basis = "Z"
            per_patch = {k: v.upper() for k, v in basis.items()}

        for value in (default_basis, *per_patch.values()):
            if value not in {"X", "Y", "Z"}:
                msg = f"Unsupported memory basis {value!r}; expected X, Y, or Z"
                raise ValueError(msg)

        self._operations.append(
            LogicalOp(
                gate_type=LogicalGateType.MEMORY,
                patches=list(patch_labels),
                rounds=rounds,
                basis=default_basis,
                per_patch_basis=per_patch,
            ),
        )

    def _require_available_patch(self, label: str) -> None:
        if label not in self._patches:
            msg = f"Unknown patch '{label}'"
            raise ValueError(msg)
        if label in self._consumed_injection_ancillas:
            msg = f"Injection ancilla '{label}' has been consumed"
            raise ValueError(msg)

    def _require_square(self, patch_label: str, gate_name: str) -> None:
        """Check that a patch is square (dx=dz), required for transversal gates."""
        patch = self._patches[patch_label].patch
        if patch.geometry.dx != patch.geometry.dz:
            msg = f"{gate_name} requires a square patch (dx=dz), got dx={patch.geometry.dx}, dz={patch.geometry.dz}"
            raise ValueError(msg)

    def add_transversal_h(self, patch_label: str) -> None:
        """Add a transversal Hadamard gate on a patch.

        Applies H to every data qubit. After this:
        - X-stabilizers become Z-stabilizers and vice versa
        - Logical X and logical Z are exchanged
        - Detectors at the boundary compare cross-type measurements

        The patch must be square (dx=dz) for the code to remain valid.

        Args:
            patch_label: Label of the patch.
        """
        self._require_available_patch(patch_label)
        self._require_square(patch_label, "Transversal H")
        self._operations.append(
            LogicalOp(
                gate_type=LogicalGateType.TRANSVERSAL_H,
                patches=[patch_label],
            ),
        )

    def add_transversal_sz(self, patch_label: str) -> None:
        """Apply physical SZ = diag(1, i) to every data qubit of a square patch.

        This layer is not a logical S gate on this code. The future
        fold-transversal construction follows Chen, Chen, Lu, Pan
        (arXiv:2412.01391) and requires additional operations.
        """
        self._require_available_patch(patch_label)
        self._require_square(patch_label, "Transversal SZ")
        self._operations.append(
            LogicalOp(
                gate_type=LogicalGateType.TRANSVERSAL_SZ,
                patches=[patch_label],
            ),
        )

    def add_transversal_szdg(self, patch_label: str) -> None:
        """Apply physical SZdg to every data qubit of a square patch.

        This inverse physical layer is not a logical S-dagger on this code.
        The future fold-transversal construction follows Chen, Chen, Lu, Pan
        (arXiv:2412.01391) and requires additional operations.
        """
        self._require_available_patch(patch_label)
        self._require_square(patch_label, "Transversal SZdg")
        self._operations.append(
            LogicalOp(
                gate_type=LogicalGateType.TRANSVERSAL_SZdg,
                patches=[patch_label],
            ),
        )

    def add_sz_via_teleportation(
        self,
        data_label: str,
        ancilla_label: str,
        rounds_before: int = 3,
        rounds_after: int = 3,
    ) -> None:
        """Apply logical SZ via gate teleportation with |+Y> ancilla.

        Complete protocol:
        1. Prepare ancilla in |+Y> = S|+> (non-fault-tolerant injection)
        2. Syndrome rounds to project ancilla into code space
        3. Transversal CX(data=control, ancilla=target)
        4. Syndrome rounds
        5. Ancilla measured in Z-basis (final round)

        After CX, data has S|psi> (up to Z correction from ancilla outcome).
        The ancilla must have odd dx and dz for encoded logical-Y content.
        ``injection_readouts`` is emitted for a future consumer; no decoder
        applies the correction today.

        Note: The |+Y> injection is non-fault-tolerant (distance-1).
        For fault-tolerant SZ, use magic state distillation on the
        injected state before consumption.

        Args:
            data_label: Label of the data patch (receives the SZ gate).
            ancilla_label: Label of the ancilla patch (consumed).
            rounds_before: Syndrome rounds before CX.
            rounds_after: Syndrome rounds after CX.
        """
        self._require_fresh_injection_ancilla(data_label, ancilla_label)
        ancilla = self._patches[ancilla_label].patch
        if ancilla.dx % 2 == 0 or ancilla.dz % 2 == 0:
            msg = f"Injection ancilla '{ancilla_label}' requires odd dx and dz for encoded logical-Y content"
            raise ValueError(msg)
        # Step 1: Init both patches — data continues in Z, ancilla in |+Y>.
        # Per-patch basis lets us do this in a single parallel segment.
        self.add_memory(
            [data_label, ancilla_label],
            rounds=rounds_before,
            basis={data_label: "Z", ancilla_label: "Y"},
        )
        # Step 2: CX(data=control, ancilla=target) — teleports S onto data.
        # Marked as teleportation so the ancilla readout is kept separately
        # for its conditional Z correction, which preserves data Z_L.
        self._operations.append(
            LogicalOp(
                gate_type=LogicalGateType.TRANSVERSAL_CX,
                patches=[data_label, ancilla_label],
                teleportation=True,
                injection_type="SZ",
            ),
        )
        # Step 3: Post-CX extraction. Ancilla measured in Z-basis at final round.
        # If ancilla measures logical -1, apply Z correction (Pauli frame update).
        self.add_memory([data_label, ancilla_label], rounds=rounds_after, basis="Z")
        self._consumed_injection_ancillas.add(ancilla_label)

    def add_t_via_injection(
        self,
        data_label: str,
        ancilla_label: str,
        rounds_before: int = 3,
        rounds_after: int = 3,
    ) -> None:
        """Emit a Clifford stand-in for T injection using a fresh |+> ancilla.

        H on every ancilla data qubit precedes syndrome projection, transversal
        CX, and Z readout. No T gate or conditional S correction is emitted.
        The feed-forward decision point is descriptor-only; real T injection
        needs a later gadget with its own layout. ``injection_readouts`` is
        emitted for a future consumer; no decoder applies the correction today.

        Args:
            data_label: Label of the data patch.
            ancilla_label: Label of a patch not yet used in memory.
            rounds_before: Syndrome rounds before CX.
            rounds_after: Syndrome rounds after CX.
        """
        self._require_fresh_injection_ancilla(data_label, ancilla_label)
        self.add_memory(
            [data_label, ancilla_label],
            rounds=rounds_before,
            basis={data_label: "Z", ancilla_label: "X"},
        )
        # Step 2: CX(data=control, ancilla=target) in the Clifford stand-in.
        self._operations.append(
            LogicalOp(
                gate_type=LogicalGateType.TRANSVERSAL_CX,
                patches=[data_label, ancilla_label],
                teleportation=True,
                injection_type="T",
            ),
        )
        # Step 3: Post-CX extraction. Ancilla measured in Z-basis.
        # If ancilla measures logical -1 (corrected by frame), apply S.
        # This is the feed-forward decision point.
        self.add_memory(
            [data_label, ancilla_label],
            rounds=rounds_after,
            basis="Z",
        )
        self._consumed_injection_ancillas.add(ancilla_label)

    def add_transversal_cx(self, control_label: str, target_label: str) -> None:
        """Add a transversal CNOT between two patches.

        Applies CX between corresponding data qubits. After this:
        - X-errors on control propagate to target
        - Z-errors on target propagate back to control
        - Weight-3 hyperedges appear in the DEM at the gate boundary

        Both patches must have the same geometry.

        Args:
            control_label: Label of the control patch.
            target_label: Label of the target patch.
        """
        self._require_cx_geometry(control_label, target_label)
        self._operations.append(
            LogicalOp(
                gate_type=LogicalGateType.TRANSVERSAL_CX,
                patches=[control_label, target_label],
            ),
        )

    def _require_cx_geometry(self, control_label: str, target_label: str) -> None:
        for label in (control_label, target_label):
            self._require_available_patch(label)
        if not gadgets.same_static_geometry(self._patches[control_label].patch, self._patches[target_label].patch):
            msg = "Transversal CX requires the same static geometry"
            raise ValueError(msg)

    def _require_fresh_injection_ancilla(self, data_label: str, ancilla_label: str) -> None:
        self._require_cx_geometry(data_label, ancilla_label)
        if any(op.gate_type == LogicalGateType.MEMORY and ancilla_label in op.patches for op in self._operations):
            msg = f"Injection ancilla '{ancilla_label}' must be fresh (already appears in memory)"
            raise ValueError(msg)

    def _snapshot_and_reset(self) -> PatchSnapshot:
        """Snapshot patch states and reset for generation."""
        saved = {label: ps.x_z_swapped for label, ps in self._patches.items()}
        for ps in self._patches.values():
            ps.x_z_swapped = False
        return saved

    def _restore(self, saved: PatchSnapshot) -> None:
        """Restore patch states from snapshot."""
        for label, swapped in saved.items():
            self._patches[label].x_z_swapped = swapped

    def to_tick_circuit(self) -> object:
        """Generate a PECOS TickCircuit with detector and observable annotations.

        This is the primary output — the TickCircuit is the source of truth.
        Use ``to_stim()`` for Stim format (derived from TickCircuit via
        ``tick_circuit_to_stim``), or ``.to_dag_circuit()`` for fault analysis.

        Returns:
            TickCircuit with gates, detectors, and observables as metadata.
        """
        saved = self._snapshot_and_reset()
        gen = _CircuitGenerator(
            patches=self._patches,
            operations=self._operations,
        )
        try:
            return gen.generate()
        finally:
            self._restore(saved)

    def to_dag_circuit(self) -> object:
        """Generate a PECOS DagCircuit for fault analysis.

        Converts the TickCircuit to a DagCircuit, which can be used
        with ``DagFaultAnalyzer`` for fault propagation analysis.

        Returns:
            DagCircuit instance.
        """
        return self.to_tick_circuit().to_dag_circuit()

    def to_stim(
        self,
        *,
        p1: float = 0.0,
        p2: float = 0.0,
        p_meas: float = 0.0,
        p_prep: float = 0.0,
    ) -> str:
        """Generate a Stim circuit string with correct detectors.

        Builds a TickCircuit (source of truth), then converts to Stim
        format with noise injection via ``tick_circuit_to_stim()``.

        Args:
            p1: Single-qubit depolarizing error rate.
            p2: Two-qubit depolarizing error rate.
            p_meas: Measurement error rate.
            p_prep: Preparation error rate.

        Returns:
            Stim circuit string.
        """
        from pecos.qec.surface.circuit_builder import tick_circuit_to_stim

        tc = self.to_tick_circuit()
        return tick_circuit_to_stim(tc, p1=p1, p2=p2, p_meas=p_meas, p_prep=p_prep)

    def stab_coords(self) -> list[dict[str, list[tuple[float, float]]]]:
        """Compute stabilizer coordinates for all patches.

        Returns a list (one per patch, in registration order) of dicts
        with keys "X" and "Z" mapping to ancilla (x, y) positions.
        These coordinates match the detector annotations in the Stim circuit.

        Used as input to ``LogicalSubgraphDecoder``.
        """
        result = []
        for ps in self._patches.values():
            geom = ps.patch.geometry
            cx, cy = ps.coord_offset
            x_coords = []
            for s in geom.x_stabilizers:
                positions = [geom.id_to_pos[q] for q in s.data_qubits]
                avg_row = sum(r for r, c in positions) / len(positions)
                avg_col = sum(c for r, c in positions) / len(positions)
                x_coords.append((avg_col * 2 + cx, avg_row * 2 + cy))
            z_coords = []
            for s in geom.z_stabilizers:
                positions = [geom.id_to_pos[q] for q in s.data_qubits]
                avg_row = sum(r for r, c in positions) / len(positions)
                avg_col = sum(c for r, c in positions) / len(positions)
                z_coords.append((avg_col * 2 + cx, avg_row * 2 + cy))
            result.append({"X": x_coords, "Z": z_coords})
        return result

    def build_dem(
        self,
        *,
        p1: float = 0.001,
        p2: float = 0.001,
        p_meas: float = 0.001,
        p_prep: float = 0.0,
    ) -> str:
        """Generate a DEM using the PECOS-native fault analysis pipeline.

        TickCircuit -> DagCircuit -> DagFaultAnalyzer -> DemBuilder.
        No Stim dependency.

        Args:
            p1: Single-qubit depolarizing error rate.
            p2: Two-qubit depolarizing error rate.
            p_meas: Measurement error rate.
            p_prep: Preparation error rate.

        Returns:
            DEM string in Stim-compatible format.
        """
        from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder

        tc = self.to_tick_circuit()
        dc = tc.to_dag_circuit()
        analyzer = DagFaultAnalyzer(dc)
        influence_map = analyzer.build_influence_map()

        det_json = tc.get_meta("detectors")
        obs_json = tc.get_meta("observables")
        num_meas = int(tc.get_meta("num_measurements"))

        meas_order = []
        for tick_idx in range(tc.num_ticks()):
            tick = tc.get_tick(tick_idx)
            for gate in tick.gate_batches():
                if gate.gate_type.name == "MZ":
                    meas_order.extend(int(q) for q in gate.qubits)

        dem_builder = DemBuilder(influence_map)
        dem_builder = dem_builder.with_noise(p1, p2, p_meas, p_prep)
        dem_builder = dem_builder.with_detectors_json(det_json)
        dem_builder = dem_builder.with_observables_json(obs_json)
        dem_builder = dem_builder.with_num_measurements(num_meas)
        dem_builder = dem_builder.with_measurement_order(meas_order)

        return str(dem_builder.build())

    def build_sampler_and_decoder(
        self,
        *,
        p1: float = 0.001,
        p2: float = 0.001,
        p_meas: float = 0.001,
        p_prep: float = 0.0,
        inner_decoder: str = "pymatching",
    ) -> tuple[object, object, str]:
        """Build a DemSampler and OSD decoder without any string round-trip.

        Returns:
            Tuple of (DemSampler, LogicalSubgraphDecoder, dem_str).
            dem_str is also returned for compatibility with existing code.
        """
        from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder, LogicalSubgraphDecoder

        tc = self.to_tick_circuit()
        dc = tc.to_dag_circuit()
        analyzer = DagFaultAnalyzer(dc)
        influence_map = analyzer.build_influence_map()

        det_json = tc.get_meta("detectors")
        obs_json = tc.get_meta("observables")
        num_meas = int(tc.get_meta("num_measurements"))

        meas_order = []
        for tick_idx in range(tc.num_ticks()):
            tick = tc.get_tick(tick_idx)
            for gate in tick.gate_batches():
                if gate.gate_type.name == "MZ":
                    meas_order.extend(int(q) for q in gate.qubits)

        dem_builder = DemBuilder(influence_map)
        dem_builder = dem_builder.with_noise(p1, p2, p_meas, p_prep)
        dem_builder = dem_builder.with_detectors_json(det_json)
        dem_builder = dem_builder.with_observables_json(obs_json)
        dem_builder = dem_builder.with_num_measurements(num_meas)
        dem_builder = dem_builder.with_measurement_order(meas_order)

        dem = dem_builder.build()
        sampler = dem.to_sampler()
        dem_str = str(dem)

        sc = self.stab_coords()
        decoder = LogicalSubgraphDecoder(dem_str, sc, inner_decoder)

        return sampler, decoder, dem_str

    def _z_frame_slot(self, label: str) -> int:
        """Z slot in the descriptor's per-patch (X, Z) frame layout."""
        return list(self._patches).index(label) * 2 + 1

    def build_algorithm_descriptor(
        self,
        *,
        p1: float = 0.001,
        p2: float = 0.001,
        p_meas: float = 0.001,
        p_prep: float = 0.0,
        buffer: int = 0,
    ) -> dict:
        """Extract per-segment DEMs and boundary gates for LogicalAlgorithmDecoder.

        Splits the full circuit DEM at gate boundaries. Each memory operation
        becomes a segment; each transversal gate becomes a boundary gate with
        Pauli frame propagation rules.

        Returns:
            Dict with keys: segments, boundary_gates, num_observables,
            num_frame_slots, full_dem. ``num_observables`` is the full DEM's
            declared observable count; ``num_frame_slots`` is two per patch
            (X then Z). ``injection_readouts`` carries raw ancilla parity
            records separately from deterministic observables and frame slots.
            It is emitted for a future consumer; no decoder applies the
            correction today. T decision-point execution remains unsupported.
        """
        # Build the full DEM
        full_dem = self.build_dem(p1=p1, p2=p2, p_meas=p_meas, p_prep=p_prep)
        sc = self.stab_coords()

        # Parse detector time coordinates from full DEM
        det_times = {}
        for raw_line in full_dem.split("\n"):
            line = raw_line.strip()
            if line.startswith("detector("):
                paren = line.index(")")
                coords = [float(x) for x in line[len("detector(") : paren].split(",")]
                tokens = line[paren + 1 :].split()
                for tok in tokens:
                    if tok.startswith("D"):
                        det_id = int(tok[1:])
                        det_times[det_id] = coords[-1] if coords else 0.0

        # Compute segment time boundaries from operations.
        # Each MEMORY op has a number of rounds. Time coordinates are
        # sequential round indices across all segments.
        segments = []
        boundary_gates = []
        # Gates accumulate between consecutive MEMORY ops.
        pending_gates = []
        time_cursor = 0.0
        patch_labels = list(self._patches.keys())
        num_patches = len(patch_labels)

        # Track X/Z swap state per patch for stab_coords.
        # After transversal H, the X and Z stabilizer types swap.
        x_z_swapped = dict.fromkeys(patch_labels, False)

        for i, op in enumerate(self._operations):
            if op.gate_type == LogicalGateType.MEMORY:
                if not segments and pending_gates:
                    gate_kinds = ", ".join(dict.fromkeys(gate["type"] for gate in pending_gates))
                    msg = (
                        f"leading logical gates before any syndrome round have no representable "
                        f"boundary (no preceding segment): {gate_kinds}"
                    )
                    raise ValueError(msg)

                # If there are pending gates, they form the boundary
                # between the previous segment and this one.
                if segments and pending_gates:
                    boundary_gates.append(pending_gates)
                    pending_gates = []
                elif segments:
                    # No gate between segments — empty boundary
                    boundary_gates.append([])
                    pending_gates = []
                seg_start = time_cursor
                seg_end = time_cursor + op.rounds
                time_cursor = seg_end

                # Find detectors in this time range, extended by buffer.
                # Buffer extends the window into adjacent segments for
                # cross-boundary error correlation context.
                buf_start = max(0, seg_start - buffer)
                buf_end = seg_end + buffer

                is_last = all(
                    self._operations[j].gate_type != LogicalGateType.MEMORY for j in range(i + 1, len(self._operations))
                )
                if is_last:
                    seg_det_ids = sorted(d for d, t in det_times.items() if t >= buf_start)
                else:
                    seg_det_ids = sorted(d for d, t in det_times.items() if buf_start <= t < buf_end)

                # Build per-segment stab_coords respecting X/Z swap state
                seg_sc = []
                for label in patch_labels:
                    base = sc[patch_labels.index(label)]
                    if x_z_swapped[label]:
                        # Swap X and Z positions
                        seg_sc.append({"X": base["Z"], "Z": base["X"]})
                    else:
                        seg_sc.append({"X": base["X"], "Z": base["Z"]})

                segments.append(
                    {
                        "det_ids": seg_det_ids,
                        "num_detectors": len(seg_det_ids),
                        "time_start": seg_start,
                        "time_end": seg_end,
                        "stab_coords": seg_sc,
                    },
                )

            elif op.gate_type == LogicalGateType.TRANSVERSAL_H:
                label = op.patches[0]
                idx = patch_labels.index(label)
                pending_gates.append(
                    {
                        "type": "Hadamard",
                        "x_obs_bit": idx * 2,
                        "z_obs_bit": self._z_frame_slot(label),
                    },
                )
                x_z_swapped[label] = not x_z_swapped[label]

            elif op.gate_type == LogicalGateType.TRANSVERSAL_CX:
                ctrl_label, tgt_label = op.patches[0], op.patches[1]
                ctrl_idx = patch_labels.index(ctrl_label)
                tgt_idx = patch_labels.index(tgt_label)
                if op.injection_type == "T":
                    pending_gates.append(
                        {
                            "type": "TGateInjection",
                            "z_obs_bit": self._z_frame_slot(ctrl_label),
                            "ancilla_z_bit": self._z_frame_slot(tgt_label),
                        },
                    )
                else:
                    pending_gates.append(
                        {
                            "type": "Cnot",
                            "ctrl_x_bit": ctrl_idx * 2,
                            "ctrl_z_bit": self._z_frame_slot(ctrl_label),
                            "tgt_x_bit": tgt_idx * 2,
                            "tgt_z_bit": self._z_frame_slot(tgt_label),
                        },
                    )

            elif op.gate_type in (LogicalGateType.TRANSVERSAL_SZ, LogicalGateType.TRANSVERSAL_SZdg):
                label = op.patches[0]
                idx = patch_labels.index(label)
                pending_gates.append(
                    {
                        "type": "SGate",
                        "x_obs_bit": idx * 2,
                        "z_obs_bit": self._z_frame_slot(label),
                    },
                )

        if not segments:
            msg = "algorithm descriptor must contain at least one segment"
            raise ValueError(msg)

        if pending_gates:
            gate_kinds = ", ".join(dict.fromkeys(gate["type"] for gate in pending_gates))
            msg = (
                f"trailing logical gates would be dropped ({gate_kinds}): logical gates after "
                "a patch's final MEMORY would execute after final data measurement; representing "
                "them requires terminal-segment support, tracked in issue #595"
            )
            raise ValueError(msg)

        _validate_boundary_cardinality(segments, boundary_gates)

        # Build per-segment sub-DEMs by filtering the full DEM.
        # Each segment gets only the mechanisms involving its detectors.
        seg_dems = []
        for seg in segments:
            set(seg["det_ids"])
            # Build local detector index mapping
            global_to_local = {g: local_id for local_id, g in enumerate(seg["det_ids"])}

            lines = []
            # Add detector coordinate declarations
            for raw_line in full_dem.split("\n"):
                line = raw_line.strip()
                if line.startswith("detector("):
                    paren = line.index(")")
                    tokens = line[paren + 1 :].split()
                    for tok in tokens:
                        if tok.startswith("D"):
                            d_id = int(tok[1:])
                            if d_id in global_to_local:
                                local = global_to_local[d_id]
                                coords = line[len("detector(") : paren]
                                lines.append(f"detector({coords}) D{local}")

            # Add error mechanisms (remap detector IDs)
            for raw_line in full_dem.split("\n"):
                line = raw_line.strip()
                if not line.startswith("error("):
                    continue
                tokens = line.split()
                prob_tok = tokens[0]
                new_tokens = [prob_tok]
                has_local_det = False
                for tok in tokens[1:]:
                    if tok.startswith("D"):
                        d_id = int(tok[1:])
                        if d_id in global_to_local:
                            new_tokens.append(f"D{global_to_local[d_id]}")
                            has_local_det = True
                    elif tok.startswith("L"):
                        new_tokens.append(tok)
                if has_local_det:
                    lines.append(" ".join(new_tokens))

            seg_dems.append("\n".join(lines))

        # Physical code distance for latency/windowing decisions. With multiple
        # patches use the minimum (the weakest bound governs latency). This is the
        # real surface-code distance, NOT a count of logical patches.
        distance = min(
            (min(ps.patch.geometry.dx, ps.patch.geometry.dz) for ps in self._patches.values()),
            default=0,
        )

        from pecos_rslib.qec import ParsedDem

        num_observables = ParsedDem.from_string(full_dem).num_observables
        injection_readouts = json.loads(self.to_tick_circuit().get_meta("injection_readouts"))
        for readout in injection_readouts:
            readout["data_z_frame_slot"] = self._z_frame_slot(readout["data_patch"])
            readout["ancilla_z_frame_slot"] = self._z_frame_slot(readout["ancilla_patch"])

        return {
            "segments": [
                {
                    "dem": seg_dems[i],
                    "num_detectors": segments[i]["num_detectors"],
                    "stab_coords": segments[i]["stab_coords"],
                }
                for i in range(len(segments))
            ],
            "boundary_gates": boundary_gates,
            "num_observables": num_observables,
            "num_frame_slots": num_patches * 2,
            "injection_readouts": injection_readouts,
            "full_dem": full_dem,
            "distance": distance,
        }

    def build_decoder(
        self,
        *,
        p1: float = 0.001,
        p2: float = 0.001,
        p_meas: float = 0.001,
        p_prep: float = 0.0,
        inner_decoder: str = "fusion_blossom_serial",
        use_stim_dem: bool = True,
    ) -> tuple[object, object]:
        """Build an LogicalSubgraphDecoder for this circuit.

        Args:
            p1: Single-qubit depolarizing error rate.
            p2: Two-qubit depolarizing error rate.
            p_meas: Measurement error rate.
            p_prep: Preparation error rate.
            inner_decoder: Decoder type for each subgraph.
            use_stim_dem: If True, use Stim for DEM generation (more error
                mechanisms). If False, use PECOS-native DEM pipeline.

        Returns:
            Tuple of (stim.Circuit, LogicalSubgraphDecoder).
        """
        import stim
        from pecos_rslib.qec import LogicalSubgraphDecoder

        stim_str = self.to_stim(p1=p1, p2=p2, p_meas=p_meas, p_prep=p_prep)
        circuit = stim.Circuit(stim_str)

        if use_stim_dem:
            dem = circuit.detector_error_model(ignore_decomposition_failures=True)
            dem_str = str(dem)
        else:
            dem_str = self.build_dem(p1=p1, p2=p2, p_meas=p_meas, p_prep=p_prep)

        sc = self.stab_coords()
        decoder = LogicalSubgraphDecoder(dem_str, sc, inner_decoder)
        return circuit, decoder


class _CircuitGenerator:
    """Internal: generates a PECOS TickCircuit for logical circuits.

    Builds a TickCircuit with detector and observable annotations as
    JSON metadata. The TickCircuit is the source of truth; Stim circuit
    strings are derived from it via tick_circuit_to_stim().
    """

    def __init__(
        self,
        patches: dict[str, PatchState],
        operations: list[LogicalOp],
    ) -> None:
        from pecos_rslib.quantum import TickCircuit

        self.patches = patches
        self.operations = operations
        # Snapshot live geometry once per generation, including edits made
        # after registration or between repeated builder renderings.
        for ps in patches.values():
            ps.stabilizers_by_index = {
                "X": {s.index: s for s in ps.patch.geometry.x_stabilizers},
                "Z": {s.index: s for s in ps.patch.geometry.z_stabilizers},
            }

        self.tc = TickCircuit()
        self._current_tick = None
        self._allocated: set[int] = set()
        self.meas_count = 0

        self.stab_meas: dict[tuple[str, str, int, int, int], int] = {}
        self.data_meas: dict[tuple[str, int], int] = {}
        self._injection_ops = [op for op in operations if op.teleportation]
        self._injection_ancillas = {op.patches[1] for op in self._injection_ops}
        self._injection_readouts: dict[str, dict] = {}

        self._prepared: set[str] = set()
        self.segment_idx = 0
        self.next_observable_idx = 0
        self.round_time = 0.0

        self._det_json: list[dict] = []
        self._obs_json: list[dict] = []

    def _new_tick(self) -> object:
        self._current_tick = self.tc.tick()
        return self._current_tick

    def _tick(self) -> object:
        if self._current_tick is None:
            return self._new_tick()
        return self._current_tick

    def _end_tick(self) -> None:
        self._current_tick = None

    def _emit_qalloc_or_reset(self, qubits: list[int]) -> None:
        t = self._tick()
        new_qs = [q for q in qubits if q not in self._allocated]
        old_qs = [q for q in qubits if q in self._allocated]
        if new_qs:
            t.qalloc(new_qs)
            self._allocated.update(new_qs)
        if old_qs:
            t.pz(old_qs)

    def generate(self) -> object:
        """Generate the TickCircuit with detector/observable metadata."""
        # Per-patch last memory index: for each patch, the last MEMORY
        # operation that includes it. This ensures each patch gets its
        # final measurement emitted in the correct segment.
        last_mem_for_patch: dict[str, int] = {}
        for i, op in enumerate(self.operations):
            if op.gate_type == LogicalGateType.MEMORY:
                for label in op.patches:
                    last_mem_for_patch[label] = i

        for op_idx, op in enumerate(self.operations):
            if op.gate_type == LogicalGateType.MEMORY:
                # A patch is "last" in this segment if this is its last memory op.
                last_patches = {label for label in op.patches if last_mem_for_patch.get(label) == op_idx}
                self._emit_memory_segment(
                    op,
                    is_last=bool(last_patches),
                    last_patches=last_patches,
                )
                self.segment_idx += 1

            elif op.gate_type == LogicalGateType.TRANSVERSAL_H:
                self._emit_transversal_h(op)

            elif op.gate_type == LogicalGateType.TRANSVERSAL_SZ:
                self._emit_transversal_sz(op)

            elif op.gate_type == LogicalGateType.TRANSVERSAL_SZdg:
                self._emit_transversal_szdg(op)

            elif op.gate_type == LogicalGateType.TRANSVERSAL_CX:
                self._emit_transversal_cx(op)

        # Build detector/observable definitions with both formats:
        # - "records": negative offsets (Stim compatibility, legacy)
        # - "meas_ids": absolute MeasResult IDs (stable, preferred)
        total = self.meas_count
        det_out = [
            {
                "id": d["id"],
                "coords": d["coords"],
                "records": [idx - total for idx in d["abs_records"]],
                "meas_ids": d["abs_records"],
            }
            for d in self._det_json
        ]
        obs_out = [
            {
                "id": o["id"],
                "records": [idx - total for idx in o["abs_records"]],
                "meas_ids": o["abs_records"],
            }
            for o in self._obs_json
        ]

        for label in self._injection_ancillas:
            if label not in self._injection_readouts:
                msg = f"Injection ancilla '{label}' has no logical operator for its readout"
                raise ValueError(msg)

        self.tc.set_meta("detectors", json.dumps(det_out))
        self.tc.set_meta("observables", json.dumps(obs_out))
        self.tc.set_meta("num_measurements", str(total))
        # Semantic provenance of every measurement ordinal, for cross-form
        # comparison against Guppy sidebands without reaching into the generator.
        self.tc.set_meta(
            "measurement_keys",
            json.dumps(
                {
                    "stabilizer": [[*key, ordinal] for key, ordinal in self.stab_meas.items()],
                    "data": [[label, qubit, ordinal] for (label, qubit), ordinal in self.data_meas.items()],
                },
            ),
        )
        self.tc.set_meta(
            "injection_readouts",
            json.dumps(
                [
                    {
                        "data_patch": op.patches[0],
                        "ancilla_patch": op.patches[1],
                        "injection_type": op.injection_type,
                        **self._injection_readouts[op.patches[1]],
                        "records": [idx - total for idx in self._injection_readouts[op.patches[1]]["meas_ids"]],
                    }
                    for op in self._injection_ops
                ],
            ),
        )
        return self.tc

    def _first_memory_basis(self, patch_label: str | None = None) -> str:
        """Basis of this patch's first memory segment (preparation)."""
        for op in self.operations:
            if op.gate_type == LogicalGateType.MEMORY and (patch_label is None or patch_label in op.patches):
                return op.per_patch_basis.get(patch_label, op.basis)
        return "Z"

    def _last_memory_basis(self, patch_label: str | None = None) -> str:
        """Basis of this patch's last memory segment (readout)."""
        for op in reversed(self.operations):
            if op.gate_type == LogicalGateType.MEMORY and (patch_label is None or patch_label in op.patches):
                return op.per_patch_basis.get(patch_label, op.basis)
        return "Z"

    def _emit_meas(self, qubits: list[int]) -> list[int]:
        self._tick().mz(qubits)
        indices = list(range(self.meas_count, self.meas_count + len(qubits)))
        self.meas_count += len(qubits)
        return indices

    def _allocation(self, patch_label: str) -> QubitAllocation:
        ps = self.patches[patch_label]
        allocation = gadgets.default_allocation(ps.patch)
        return QubitAllocation(
            [q + ps.qubit_offset for q in allocation.data_qubits],
            [q + ps.qubit_offset for q in allocation.x_ancilla_qubits],
            [q + ps.qubit_offset for q in allocation.z_ancilla_qubits],
        )

    def _emit_steps(self, step_lists: list[tuple[SurfaceCircuitStep, ...]]) -> dict[tuple[int, int], int]:
        """Split different-type conflicts within TICK groups, then zip with padding.

        Each TICK-delimited group, including a comment-only group, produces
        at least one tick; trailing empty groups at the end of the merged
        stream are omitted. Physical steps must contain at least one qubit.

        Patch and step order determine measurement indices. Ancillas retain
        allocation across rounds, unlike TickCircuitRenderer's freeing readout.
        Repeated same-type operations on a qubit in one group are invalid
        parallel layers and must not be silently serialized.
        TickCircuitRenderer retains its silent same-type split until the
        non-rotated schedule is fixed.
        """
        streams = []
        for position, steps in enumerate(step_lists):
            groups = []
            subticks = []
            current = []
            used = set()
            group_used = set()
            layer_label = "unnamed"
            for step in steps:
                if step.op_type == OpType.COMMENT:
                    layer_label = step.label
                    continue
                if step.op_type == OpType.TICK:
                    if current:
                        subticks.append(current)
                    groups.append(subticks or [[]])
                    subticks, current, used = [], [], set()
                    group_used = set()
                    continue
                layer = f"Gadget layer {position}:{len(groups)} ({layer_label})"
                if not step.qubits:
                    msg = f"{layer}: {step.op_type.name} requires at least one qubit"
                    raise ValueError(msg)
                if step.op_type == OpType.MEASURE and len(step.qubits) != 1:
                    msg = f"{layer}: MEASURE requires exactly one qubit per labeled step"
                    raise ValueError(msg)
                for qubit in step.qubits:
                    key = (step.op_type, qubit)
                    if key in group_used:
                        msg = f"{layer}: repeated {step.op_type.name} on qubit {qubit} in a parallel layer"
                        raise ValueError(msg)
                    group_used.add(key)
                if used.intersection(step.qubits):
                    subticks.append(current)
                    current, used = [], set()
                current.append(step)
                used.update(step.qubits)
            if current:
                subticks.append(current)
            if subticks:
                groups.append(subticks)
            streams.append(groups)

        measurements = {}
        merged_groups = list(zip_longest(*streams, fillvalue=()))
        while merged_groups and not any(step for group in merged_groups[-1] for subtick in group for step in subtick):
            merged_groups.pop()
        for groups in merged_groups:
            for subticks in zip_longest(*groups, fillvalue=()):
                merged = [(position, step) for position, steps in enumerate(subticks) for step in steps]
                t = self._new_tick()
                batches = {}
                for position, step in merged:
                    batches.setdefault(step.op_type, []).append((position, step))
                for op_type, batch in batches.items():
                    qubits = [q for _, step in batch for q in step.qubits]
                    if op_type == OpType.ALLOC:
                        # Preserve patch order when fresh allocations and resets
                        # share a tick. TickCircuit coalesces adjacent gate calls.
                        for _, step in batch:
                            self._emit_qalloc_or_reset(step.qubits)
                    elif op_type == OpType.CX:
                        t.cx([tuple(step.qubits) for _, step in batch])
                    elif op_type in {OpType.H, OpType.SZ, OpType.SZDG, OpType.X, OpType.Z}:
                        getattr(t, op_type.name.lower())(qubits)
                    elif op_type == OpType.MEASURE:
                        indices = iter(self._emit_meas(qubits))
                        for position, step in batch:
                            measurements[position, step.qubits[0]] = next(indices)
                    else:
                        msg = f"Unsupported gadget operation: {op_type.name}"
                        raise NotImplementedError(msg)
                self._end_tick()
        return measurements

    def _emit_memory_segment(
        self,
        op: LogicalOp,
        *,
        is_last: bool,
        last_patches: set[str] | None = None,
    ) -> None:
        """Compose preparation and orientation-aware rounds for each patch."""
        first_patches = set(op.patches) - self._prepared
        allocations = {label: self._allocation(label) for label in op.patches}
        self._emit_steps(
            [
                (
                    gadgets.prep_gadget(
                        self.patches[label].patch,
                        allocations[label],
                        basis=op.per_patch_basis.get(label, op.basis),
                    ).steps
                    if label in first_patches
                    else ()
                )
                for label in op.patches
            ],
        )
        self._prepared.update(first_patches)
        for rnd in range(op.rounds):
            measurements = self._emit_steps(
                [
                    gadgets.syndrome_round_gadget(
                        self.patches[label].patch,
                        allocations[label],
                        round_index=rnd,
                        x_z_swapped=self.patches[label].x_z_swapped,
                    ).steps
                    for label in op.patches
                ],
            )
            for (position, qubit), index in measurements.items():
                label = op.patches[position]
                allocation = allocations[label]
                if qubit in allocation.x_ancilla_qubits:
                    family, register = "X", allocation.x_ancilla_qubits
                elif qubit in allocation.z_ancilla_qubits:
                    family, register = "Z", allocation.z_ancilla_qubits
                else:
                    msg = f"Measured qubit {qubit} is not an ancilla of patch '{label}'"
                    raise ValueError(msg)
                if self.patches[label].x_z_swapped:
                    family = "Z" if family == "X" else "X"
                self.stab_meas[label, family, register.index(qubit), self.segment_idx, rnd] = index
            if hasattr(self, "_last_round_cache"):
                del self._last_round_cache
            for label in op.patches:
                self._emit_round_detectors(label, rnd, is_first_segment=label in first_patches)
            self.round_time += 1.0

        # Read every final patch before constructing cross-patch observables.
        if is_last and last_patches:
            final_patches = [label for label in op.patches if label in last_patches]
            for label in final_patches:
                self._emit_final_data_measurements(label)
            for label in final_patches:
                self._emit_final_detectors_and_observables(label)

    def _emit_round_detectors(
        self,
        patch_label: str,
        round_idx: int,
        *,
        is_first_segment: bool,
    ) -> None:
        """Emit detectors for one syndrome round.

        Handles three cases:
        1. First round of first segment: only basis-matching stabs are deterministic
        2. First round after a gate boundary: cross-type comparison needed
        3. Normal round: compare same-type measurements in consecutive rounds
        """
        ps = self.patches[patch_label]
        geom = ps.patch.geometry
        seg = self.segment_idx

        for stab_type in ["X", "Z"]:
            stabs = geom.x_stabilizers if stab_type == "X" else geom.z_stabilizers
            for s in stabs:
                curr_key = (patch_label, stab_type, s.index, seg, round_idx)
                curr_idx = self.stab_meas.get(curr_key)
                if curr_idx is None:
                    continue

                if round_idx == 0 and is_first_segment:
                    # First round of this patch's first segment:
                    # Only stabilizers matching the prep basis are deterministic.
                    # Find the prep basis from the first memory operation.
                    init_basis = self._first_memory_basis(patch_label)
                    det_type = init_basis
                    # Account for X/Z swap
                    effective_type = stab_type
                    if ps.x_z_swapped:
                        effective_type = "Z" if stab_type == "X" else "X"
                    if effective_type == det_type:
                        self._add_detector(
                            patch_label,
                            stab_type,
                            s.index,
                            [curr_idx],
                        )

                elif round_idx == 0 and seg > 0:
                    # First round after a gate boundary.
                    # Need to find the matching measurement from the previous segment.
                    self._emit_boundary_detector(patch_label, stab_type, s.index, curr_idx)

                elif round_idx > 0:
                    # Normal: compare with previous round in same segment
                    prev_key = (patch_label, stab_type, s.index, seg, round_idx - 1)
                    prev_idx = self.stab_meas.get(prev_key)
                    if prev_idx is not None:
                        self._add_detector(
                            patch_label,
                            stab_type,
                            s.index,
                            [curr_idx, prev_idx],
                        )

    def _emit_boundary_detector(
        self,
        patch_label: str,
        stab_type: str,
        stab_index: int,
        curr_meas_idx: int,
    ) -> None:
        """Emit a detector at a gate boundary.

        After transversal H: an X-check in the new segment corresponds to what
        was a Z-check in the previous segment (and vice versa). The detector
        compares the current measurement with the last measurement of the
        *conjugated* type from the previous segment.
        """
        self.patches[patch_label]
        prev_seg = self.segment_idx - 1

        # Find the gate that affects this specific patch at this boundary
        gate_op = self._find_gate_before_segment(self.segment_idx, patch_label)

        if (
            gate_op is not None
            and gate_op.gate_type == LogicalGateType.TRANSVERSAL_H
            and patch_label in gate_op.patches
        ):
            # After H on THIS patch: X-stabs were Z-stabs, Z-stabs were X-stabs
            conjugated_type = "Z" if stab_type == "X" else "X"
            # Find the last round of the previous segment
            prev_last_round = self._last_round_of_segment(patch_label, conjugated_type, prev_seg)
            if prev_last_round is not None:
                prev_key = (patch_label, conjugated_type, stab_index, prev_seg, prev_last_round)
                prev_idx = self.stab_meas.get(prev_key)
                if prev_idx is not None:
                    self._add_detector(
                        patch_label,
                        stab_type,
                        stab_index,
                        [curr_meas_idx, prev_idx],
                    )
            # If no previous measurement found, this stabilizer wasn't measured before
            # (e.g., it's the non-deterministic type). No detector.

        elif (
            gate_op is not None
            and gate_op.gate_type == LogicalGateType.TRANSVERSAL_CX
            and patch_label in gate_op.patches
        ):
            # After CX(control, target):
            #   Control X-stabs: propagated to target → 3-body detector
            #     post_ctrl_X XOR pre_ctrl_X XOR pre_tgt_X
            #   Target Z-stabs: propagated back to control → 3-body detector
            #     post_tgt_Z XOR pre_tgt_Z XOR pre_ctrl_Z
            #   Control Z-stabs: unchanged → normal 2-body detector
            #   Target X-stabs: unchanged → normal 2-body detector
            ctrl_label = gate_op.patches[0]
            tgt_label = gate_op.patches[1]
            is_control = patch_label == ctrl_label

            prev_last_round = self._last_round_of_segment(patch_label, stab_type, prev_seg)
            if prev_last_round is None:
                return  # No previous measurement

            prev_key = (patch_label, stab_type, stab_index, prev_seg, prev_last_round)
            prev_idx = self.stab_meas.get(prev_key)
            if prev_idx is None:
                return

            needs_cross_patch = (is_control and stab_type == "X") or (not is_control and stab_type == "Z")

            if needs_cross_patch:
                # 3-body detector: also include the other patch's measurement
                other_label = tgt_label if is_control else ctrl_label
                other_last_round = self._last_round_of_segment(other_label, stab_type, prev_seg)
                if other_last_round is not None:
                    other_key = (other_label, stab_type, stab_index, prev_seg, other_last_round)
                    other_idx = self.stab_meas.get(other_key)
                    if other_idx is not None:
                        self._add_detector(
                            patch_label,
                            stab_type,
                            stab_index,
                            [curr_meas_idx, prev_idx, other_idx],
                        )
                        return
                # Fall through to 2-body if cross-patch measurement not found
            self._add_detector(
                patch_label,
                stab_type,
                stab_index,
                [curr_meas_idx, prev_idx],
            )

        else:
            # No gate boundary — normal comparison with previous segment
            prev_last_round = self._last_round_of_segment(patch_label, stab_type, prev_seg)
            if prev_last_round is not None:
                prev_key = (patch_label, stab_type, stab_index, prev_seg, prev_last_round)
                prev_idx = self.stab_meas.get(prev_key)
                if prev_idx is not None:
                    self._add_detector(
                        patch_label,
                        stab_type,
                        stab_index,
                        [curr_meas_idx, prev_idx],
                    )

    def _find_gate_before_segment(
        self,
        segment_idx: int,
        patch_label: str | None = None,
    ) -> LogicalOp | None:
        """Find the gate operation that precedes a memory segment.

        If patch_label is given, returns the gate that affects that specific
        patch (checking gate_op.patches). This handles the case where multiple
        gates are stacked between segments (e.g., H on A then H on B).
        """
        mem_count = 0
        for i, op in enumerate(self.operations):
            if op.gate_type == LogicalGateType.MEMORY:
                if mem_count == segment_idx:
                    # Look backwards for gates
                    for j in range(i - 1, -1, -1):
                        if self.operations[j].gate_type == LogicalGateType.MEMORY:
                            break
                        if patch_label is None:
                            return self.operations[j]
                        if patch_label in self.operations[j].patches:
                            return self.operations[j]
                    return None
                mem_count += 1
        return None

    def _last_round_of_segment(self, patch_label: str, stab_type: str, seg_idx: int) -> int | None:
        """Find the last round index for a stabilizer type in a segment.

        Uses a cached index built on first call, then O(1) lookups.
        """
        if not hasattr(self, "_last_round_cache"):
            # Build cache from stab_meas keys: (patch, type, seg) → max_round
            cache: dict[tuple[str, str, int], int] = {}
            for patch, stype, _sidx, seg, rnd in self.stab_meas:
                key = (patch, stype, seg)
                if key not in cache or rnd > cache[key]:
                    cache[key] = rnd
            self._last_round_cache = cache
        return self._last_round_cache.get((patch_label, stab_type, seg_idx))

    def _ancilla_spatial_coords(
        self,
        patch_label: str,
        stab_type: str,
        stab_index: int,
    ) -> tuple[float, float]:
        """Compute the spatial position of a stabilizer's ancilla.

        Returns (x, y) including the patch's coord_offset, using the
        average position of the stabilizer's data qubits.
        """
        ps = self.patches[patch_label]
        geom = ps.patch.geometry
        cx, cy = ps.coord_offset
        s = ps.stabilizers_by_index[stab_type].get(stab_index)
        if s is None:
            msg = f"Patch '{patch_label}' has no {stab_type} stabilizer with index {stab_index}"
            raise ValueError(msg)
        positions = [geom.id_to_pos[q] for q in s.data_qubits]
        avg_row = sum(r for r, c in positions) / len(positions)
        avg_col = sum(c for r, c in positions) / len(positions)
        return (avg_col * 2 + cx, avg_row * 2 + cy)

    def _add_detector(
        self,
        patch_label: str,
        stab_type: str,
        stab_index: int,
        meas_indices: list[int],
    ) -> None:
        anc_x, anc_y = self._ancilla_spatial_coords(patch_label, stab_type, stab_index)
        # Store absolute indices; convert to relative offsets in generate()
        self._det_json.append(
            {
                "id": len(self._det_json),
                "coords": [anc_x, anc_y, self.round_time],
                "abs_records": list(meas_indices),
            },
        )

    def _emit_transversal_layer(self, op: LogicalOp, gate: str) -> None:
        label = op.patches[0]
        self._emit_steps(
            [
                gadgets.transversal_layer_gadget(self.patches[label].patch, self._allocation(label), gate=gate).steps,
            ],
        )

    def _emit_transversal_h(self, op: LogicalOp) -> None:
        self._emit_transversal_layer(op, "H")
        ps = self.patches[op.patches[0]]
        ps.x_z_swapped = not ps.x_z_swapped

    def _emit_transversal_sz(self, op: LogicalOp) -> None:
        self._emit_transversal_layer(op, "SZ")

    def _emit_transversal_szdg(self, op: LogicalOp) -> None:
        self._emit_transversal_layer(op, "SZDG")

    def _emit_transversal_cx(self, op: LogicalOp) -> None:
        ctrl_label, tgt_label = op.patches[0], op.patches[1]
        ctrl_ps = self.patches[ctrl_label]
        tgt_ps = self.patches[tgt_label]

        if ctrl_ps.x_z_swapped != tgt_ps.x_z_swapped:
            msg = (
                f"Transversal CX requires same stabilizer orientation. "
                f"'{ctrl_label}' swapped={ctrl_ps.x_z_swapped}, "
                f"'{tgt_label}' swapped={tgt_ps.x_z_swapped}."
            )
            raise ValueError(msg)

        self._emit_steps(
            [
                gadgets.transversal_cx_gadget(
                    ctrl_ps.patch,
                    self._allocation(ctrl_label),
                    tgt_ps.patch,
                    self._allocation(tgt_label),
                ).steps,
            ],
        )

    def _emit_final_data_measurements(self, patch_label: str) -> None:
        ps = self.patches[patch_label]
        measured = self._emit_steps(
            [
                gadgets.measure_out_gadget(
                    ps.patch,
                    self._allocation(patch_label),
                    basis=self._last_memory_basis(patch_label),
                ).steps,
            ],
        )
        for q in range(ps.patch.geometry.num_data):
            self.data_meas[patch_label, q] = measured[0, ps.qubit_offset + q]

    def _emit_final_detectors_and_observables(self, patch_label: str) -> None:
        ps = self.patches[patch_label]
        geom = ps.patch.geometry
        meas_basis = self._last_memory_basis(patch_label)

        if meas_basis == "Z":
            final_stabs = geom.x_stabilizers if ps.x_z_swapped else geom.z_stabilizers
            lookup_type = "Z"
        else:
            final_stabs = geom.z_stabilizers if ps.x_z_swapped else geom.x_stabilizers
            lookup_type = "X"

        if ps.x_z_swapped:
            logical_op = geom.logical_z if meas_basis == "X" else geom.logical_x
        else:
            logical_op = geom.logical_x if meas_basis == "X" else geom.logical_z

        if logical_op is None:
            role = "Injection ancilla" if patch_label in self._injection_ancillas else "Patch"
            msg = f"{role} '{patch_label}' has no logical operator for {meas_basis} readout"
            raise ValueError(msg)

        seg = self.segment_idx
        last_rnd = self._last_round_of_segment(patch_label, lookup_type, seg)

        if last_rnd is not None:
            for s in final_stabs:
                data_rec = [self.data_meas[(patch_label, dq)] for dq in s.data_qubits]
                syn_key = (patch_label, lookup_type, s.index, seg, last_rnd)
                syn_idx = self.stab_meas.get(syn_key)
                if syn_idx is not None:
                    all_idx = [*data_rec, syn_idx]
                    anc_x, anc_y = self._ancilla_spatial_coords(patch_label, lookup_type, s.index)
                    self._det_json.append(
                        {
                            "id": len(self._det_json),
                            "coords": [anc_x, anc_y, self.round_time],
                            "abs_records": list(all_idx),
                        },
                    )

        obs_indices = [self.data_meas[(patch_label, q)] for q in logical_op.data_qubits]
        if patch_label in self._injection_ancillas:
            # A consumed ancilla's random logical readout controls a
            # correction; it is not a deterministic DEM observable.
            self._injection_readouts[patch_label] = {"basis": meas_basis, "meas_ids": obs_indices}
            return
        if not _logical_readout_is_deterministic(self.operations, self.segment_idx, patch_label, meas_basis):
            # Skip non-reliable observables — they're physically
            # non-deterministic and would cause Stim DEM errors.
            self.next_observable_idx += 1
            return

        obs_idx = self.next_observable_idx
        self.next_observable_idx += 1
        self._obs_json.append(
            {
                "id": obs_idx,
                "abs_records": list(obs_indices),
            },
        )
