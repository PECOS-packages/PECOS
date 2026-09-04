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
from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pecos.qec.surface.patch import Stabilizer, SurfacePatch

PatchSnapshot = dict[str, tuple[bool, list[str], list[str], list[str], list[str]]]

# Kept in sync with pecos_qec::DEM_SLICE_ROUND_ATTRIBUTE. The TickCircuit ->
# DagCircuit conversion copies batch metadata to every split DAG gate, letting
# the structured DEM frontend assign each physical fault location to a round.
DEM_SLICE_ROUND_ATTRIBUTE = "dem_slice_round"


@dataclass(frozen=True)
class _CachedSurfaceMemoryDemTemplates:
    """Noise-weighted bounded templates for one physical memory family."""

    output_model: object
    initialization: object
    bulk: object
    pre_terminal: object
    terminal: object


@cache
def _cached_surface_memory_dem_templates(
    dx: int,
    dz: int,
    orientation_name: str,
    rotated: bool,
    basis: str,
    p1: float,
    p2: float,
    p_meas: float,
    p_prep: float,
) -> _CachedSurfaceMemoryDemTemplates:
    """Compile the constant-depth surface-memory template on a cache miss."""
    from pecos.qec.surface.patch import PatchOrientation, SurfacePatch

    patch = SurfacePatch.create(
        dx=dx,
        dz=dz,
        orientation=PatchOrientation[orientation_name],
        rotated=rotated,
    )
    builder = LogicalCircuitBuilder()
    builder.add_patch(patch, "template", coord_offset=(0.0, 0.0))
    builder.add_memory("template", rounds=3, basis=basis)
    model, influence_map, dag_circuit = builder._build_structured_dem(  # noqa: SLF001
        p1=p1,
        p2=p2,
        p_meas=p_meas,
        p_prep=p_prep,
    )
    schedule = model.round_schedule(influence_map, dag_circuit)
    return _CachedSurfaceMemoryDemTemplates(
        output_model=model,
        initialization=schedule.template(0),
        bulk=schedule.template(1),
        pre_terminal=schedule.template(2),
        terminal=schedule.template(3),
    )


@dataclass(frozen=True)
class _CachedSurfaceBoundaryDemTemplates:
    """Noise-weighted templates around one bounded logical-gate boundary."""

    output_model: object
    initialization: object
    pre_gate_bulk: object
    pre_gate_boundary: object
    gate_boundary: object
    post_gate_bulk: object
    pre_terminal: object
    terminal: object


@cache
def _cached_surface_h_dem_templates(
    dx: int,
    dz: int,
    orientation_name: str,
    rotated: bool,
    initial_basis: str,
    final_basis: str,
    p1: float,
    p2: float,
    p_meas: float,
    p_prep: float,
) -> _CachedSurfaceBoundaryDemTemplates:
    """Compile a bounded memory-H-memory template on a cache miss."""
    from pecos.qec.surface.patch import PatchOrientation, SurfacePatch

    patch = SurfacePatch.create(
        dx=dx,
        dz=dz,
        orientation=PatchOrientation[orientation_name],
        rotated=rotated,
    )
    builder = LogicalCircuitBuilder()
    builder.add_patch(patch, "template", coord_offset=(0.0, 0.0))
    builder.add_memory("template", rounds=3, basis=initial_basis)
    builder.add_transversal_h("template")
    builder.add_memory("template", rounds=3, basis=final_basis)
    model, influence_map, dag_circuit = builder._build_structured_dem(  # noqa: SLF001
        p1=p1,
        p2=p2,
        p_meas=p_meas,
        p_prep=p_prep,
    )
    schedule = model.round_schedule(influence_map, dag_circuit)
    return _CachedSurfaceBoundaryDemTemplates(
        output_model=model,
        initialization=schedule.template(0),
        pre_gate_bulk=schedule.template(1),
        pre_gate_boundary=schedule.template(2),
        gate_boundary=schedule.template(3),
        post_gate_bulk=schedule.template(4),
        pre_terminal=schedule.template(5),
        terminal=schedule.template(6),
    )


@dataclass(frozen=True)
class _CachedSurfaceCxDemTemplates:
    """Bounded transversal-CX templates and their canonical stream layout."""

    templates: _CachedSurfaceBoundaryDemTemplates
    control_stream_count: int
    target_stream_count: int
    target_coordinate_origin: tuple[float, float]


@cache
def _cached_surface_cx_dem_templates(
    control_dx: int,
    control_dz: int,
    control_orientation_name: str,
    control_rotated: bool,
    target_dx: int,
    target_dz: int,
    target_orientation_name: str,
    target_rotated: bool,
    initial_control_basis: str,
    initial_target_basis: str,
    final_control_basis: str,
    final_target_basis: str,
    p1: float,
    p2: float,
    p_meas: float,
    p_prep: float,
) -> _CachedSurfaceCxDemTemplates:
    """Compile a bounded two-patch memory-CX-memory template."""
    from pecos.qec.surface.patch import PatchOrientation, SurfacePatch

    control = SurfacePatch.create(
        dx=control_dx,
        dz=control_dz,
        orientation=PatchOrientation[control_orientation_name],
        rotated=control_rotated,
    )
    target = SurfacePatch.create(
        dx=target_dx,
        dz=target_dz,
        orientation=PatchOrientation[target_orientation_name],
        rotated=target_rotated,
    )
    target_origin = (float(max(control_dz, target_dz) * 2 + 2), 0.0)
    builder = LogicalCircuitBuilder()
    builder.add_patch(control, "control", coord_offset=(0.0, 0.0))
    builder.add_patch(
        target,
        "target",
        qubit_offset=control.geometry.num_qubits,
        coord_offset=target_origin,
    )
    builder.add_memory(
        ["control", "target"],
        rounds=3,
        basis={"control": initial_control_basis, "target": initial_target_basis},
    )
    builder.add_transversal_cx("control", "target")
    builder.add_memory(
        ["control", "target"],
        rounds=3,
        basis={"control": final_control_basis, "target": final_target_basis},
    )
    model, influence_map, dag_circuit = builder._build_structured_dem(  # noqa: SLF001
        p1=p1,
        p2=p2,
        p_meas=p_meas,
        p_prep=p_prep,
    )
    schedule = model.round_schedule(influence_map, dag_circuit)
    templates = _CachedSurfaceBoundaryDemTemplates(
        output_model=model,
        initialization=schedule.template(0),
        pre_gate_bulk=schedule.template(1),
        pre_gate_boundary=schedule.template(2),
        gate_boundary=schedule.template(3),
        post_gate_bulk=schedule.template(4),
        pre_terminal=schedule.template(5),
        terminal=schedule.template(6),
    )
    control_stream_count = len(control.geometry.x_stabilizers) + len(control.geometry.z_stabilizers)
    target_stream_count = len(target.geometry.x_stabilizers) + len(target.geometry.z_stabilizers)
    return _CachedSurfaceCxDemTemplates(
        templates=templates,
        control_stream_count=control_stream_count,
        target_stream_count=target_stream_count,
        target_coordinate_origin=target_origin,
    )


def _boundary_template_instances(
    templates: _CachedSurfaceBoundaryDemTemplates,
    before_rounds: int,
    after_rounds: int,
) -> tuple[list[tuple[object, int]], int]:
    """Place one cached logical boundary between two memory segments."""
    boundary_round = before_rounds
    terminal_round = before_rounds + after_rounds
    instances = [(templates.initialization, 0)]
    instances.extend((templates.pre_gate_bulk, round_) for round_ in range(1, boundary_round - 1))
    instances.extend(
        [
            (templates.pre_gate_boundary, boundary_round - 1),
            (templates.gate_boundary, boundary_round),
        ],
    )
    instances.extend((templates.post_gate_bulk, round_) for round_ in range(boundary_round + 1, terminal_round - 1))
    instances.extend(
        [
            (templates.pre_terminal, terminal_round - 1),
            (templates.terminal, terminal_round),
        ],
    )
    return instances, terminal_round


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
    # Teleportation corrections: ancilla Z measurements to XOR into this
    # patch's observable (from CX teleportation gates).
    z_obs_includes: list[str] = field(default_factory=list)
    x_obs_includes: list[str] = field(default_factory=list)
    # After CX, some observables become non-reliable depending on the
    # measurement basis. Track which bases are "entangled" and with whom.
    # x_entangled_with: this patch's X observable is entangled with these
    # patches (measuring X on this patch alone is non-deterministic).
    x_entangled_with: list[str] = field(default_factory=list)
    z_entangled_with: list[str] = field(default_factory=list)

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
    # For teleportation CX: the target's Z measurement should be included
    # in the control's observable for the Pauli frame correction.
    teleportation: bool = False
    # Type of magic state injection: "T" for T-gate, "SZ" for SZ, or None.
    # Used by build_algorithm_descriptor() to emit the correct boundary gate.
    injection_type: str | None = None


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
            if label not in self._patches:
                msg = f"Unknown patch '{label}'"
                raise ValueError(msg)

        if isinstance(basis, str):
            default_basis = basis.upper()
            per_patch = {}
        else:
            default_basis = "Z"
            per_patch = {k: v.upper() for k, v in basis.items()}

        self._operations.append(
            LogicalOp(
                gate_type=LogicalGateType.MEMORY,
                patches=list(patch_labels),
                rounds=rounds,
                basis=default_basis,
                per_patch_basis=per_patch,
            ),
        )

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
        if patch_label not in self._patches:
            msg = f"Unknown patch '{patch_label}'"
            raise ValueError(msg)
        self._require_square(patch_label, "Transversal H")
        self._operations.append(
            LogicalOp(
                gate_type=LogicalGateType.TRANSVERSAL_H,
                patches=[patch_label],
            ),
        )

    def add_transversal_sz(self, patch_label: str) -> None:
        """Add a fold-transversal SZ gate on a patch.

        Implements the fold-transversal S gate using the Bravyi et al.
        half-cycle trick (arXiv:2412.01391). The fold operation is
        inserted at mid-cycle of a syndrome extraction round:
        - On-diagonal data qubits (r + c = d-1): SZ gate
        - On-diagonal X-ancilla qubits: SZdg gate
        - Off-diagonal: CZ between each qubit and its mirror

        After this gate:
        - Z-stabilizers are unchanged
        - X-stabilizers pick up Z-stabilizer partners:
          X_stab -> X_stab * Z_stab_mirror

        The patch must be square (dx=dz).

        Args:
            patch_label: Label of the patch.
        """
        if patch_label not in self._patches:
            msg = f"Unknown patch '{patch_label}'"
            raise ValueError(msg)
        self._require_square(patch_label, "Transversal SZ")
        self._operations.append(
            LogicalOp(
                gate_type=LogicalGateType.TRANSVERSAL_SZ,
                patches=[patch_label],
            ),
        )

    def add_transversal_szdg(self, patch_label: str) -> None:
        """Add SZdg (S-dagger) on all data qubits of a patch.

        Inverse of add_transversal_sz. Used for mirroring circuits
        that contain SZ gates.
        """
        if patch_label not in self._patches:
            msg = f"Unknown patch '{patch_label}'"
            raise ValueError(msg)
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
        The Z correction is a Pauli frame update tracked by the decoder.

        Note: The |+Y> injection is non-fault-tolerant (distance-1).
        For fault-tolerant SZ, use magic state distillation on the
        injected state before consumption.

        Args:
            data_label: Label of the data patch (receives the SZ gate).
            ancilla_label: Label of the ancilla patch (consumed).
            rounds_before: Syndrome rounds before CX.
            rounds_after: Syndrome rounds after CX.
        """
        # Step 1: Init both patches — data continues in Z, ancilla in |+Y>.
        # Per-patch basis lets us do this in a single parallel segment.
        self.add_memory(
            [data_label, ancilla_label],
            rounds=rounds_before,
            basis={data_label: "Z", ancilla_label: "Y"},
        )
        # Step 2: CX(data=control, ancilla=target) — teleports S onto data.
        # Marked as teleportation so observable propagation includes the
        # ancilla's Z measurement as a Pauli frame correction.
        self._operations.append(
            LogicalOp(
                gate_type=LogicalGateType.TRANSVERSAL_CX,
                patches=[data_label, ancilla_label],
                teleportation=True,
            ),
        )
        # Step 3: Post-CX extraction. Ancilla measured in Z-basis at final round.
        # If ancilla measures logical -1, apply Z correction (Pauli frame update).
        self.add_memory([data_label, ancilla_label], rounds=rounds_after, basis="Z")

    def add_t_via_injection(
        self,
        data_label: str,
        ancilla_label: str,
        rounds_before: int = 3,
        rounds_after: int = 3,
    ) -> None:
        """Apply logical T gate via magic state injection.

        Complete protocol:
        1. Prepare ancilla in |T> = T|+> (non-fault-tolerant injection)
        2. Syndrome rounds to project ancilla into code space
        3. Transversal CX(data=control, ancilla=target)
        4. Syndrome rounds
        5. Ancilla measured in Z-basis (final round)

        After CX, data has T|psi> (up to S correction from ancilla outcome).
        The S correction is a conditional feed-forward operation — this is
        a DECISION POINT where the decoder must provide the Pauli frame.

        The corrected measurement outcome determines:
          corrected = raw_measurement XOR frame[z_obs_bit]
          if corrected == 1: apply S gate on data

        Note: The |T> injection is non-fault-tolerant (distance-1).
        For fault-tolerant T, use magic state distillation on the
        injected state before consumption.

        Args:
            data_label: Label of the data patch (receives the T gate).
            ancilla_label: Label of the ancilla patch (consumed).
            rounds_before: Syndrome rounds before CX.
            rounds_after: Syndrome rounds after CX.
        """
        # Step 1: Init both patches — data continues in Z, ancilla gets |+>.
        # The real protocol prepares |T> = T|+> on the ancilla, but T is
        # non-Clifford and invisible to the fault analyzer. We prepare
        # |+> as a Clifford stand-in: same error structure (H gate noise
        # via p1, prep noise via p_prep), just missing the non-Clifford
        # phase which doesn't affect error correlations in the DEM.
        self.add_memory(
            [data_label, ancilla_label],
            rounds=rounds_before,
            basis={data_label: "Z", ancilla_label: "X"},
        )
        # Step 2: CX(data=control, ancilla=target) — teleports T onto data.
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
        if control_label not in self._patches:
            msg = f"Unknown patch '{control_label}'"
            raise ValueError(msg)
        if target_label not in self._patches:
            msg = f"Unknown patch '{target_label}'"
            raise ValueError(msg)
        ctrl = self._patches[control_label]
        tgt = self._patches[target_label]
        if ctrl.patch.geometry.num_data != tgt.patch.geometry.num_data:
            msg = (
                f"Patches must have same geometry for transversal CX. "
                f"'{control_label}' has {ctrl.patch.geometry.num_data} data qubits, "
                f"'{target_label}' has {tgt.patch.geometry.num_data} data qubits."
            )
            raise ValueError(msg)
        self._operations.append(
            LogicalOp(
                gate_type=LogicalGateType.TRANSVERSAL_CX,
                patches=[control_label, target_label],
            ),
        )

    def _snapshot_and_reset(self) -> PatchSnapshot:
        """Snapshot patch states and reset for generation."""
        saved = {
            label: (
                ps.x_z_swapped,
                list(ps.z_obs_includes),
                list(ps.x_obs_includes),
                list(ps.x_entangled_with),
                list(ps.z_entangled_with),
            )
            for label, ps in self._patches.items()
        }
        for ps in self._patches.values():
            ps.x_z_swapped = False
            ps.z_obs_includes = []
            ps.x_obs_includes = []
            ps.x_entangled_with = []
            ps.z_entangled_with = []
        return saved

    def _restore(self, saved: PatchSnapshot) -> None:
        """Restore patch states from snapshot."""
        for label, (swapped, z_obs, x_obs, x_ent, z_ent) in saved.items():
            ps = self._patches[label]
            ps.x_z_swapped = swapped
            ps.z_obs_includes = z_obs
            ps.x_obs_includes = x_obs
            ps.x_entangled_with = x_ent
            ps.z_entangled_with = z_ent

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
        tc = gen.generate()
        self._restore(saved)
        return tc

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

    def _build_structured_dem(
        self,
        *,
        p1: float = 0.001,
        p2: float = 0.001,
        p_meas: float = 0.001,
        p_prep: float = 0.0,
    ) -> tuple[object, object, object]:
        """Build the native DEM together with its source-tracking context."""
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

        return dem_builder.build(), influence_map, dc

    def _build_structured_dem_from_cached_templates(
        self,
        *,
        p1: float,
        p2: float,
        p_meas: float,
        p_prep: float,
    ) -> tuple[object, object] | None:
        """Assemble an eligible surface DEM from bounded template caches."""
        if len(self._operations) == 3:
            if len(self._patches) == 1:
                return self._build_structured_h_dem_from_cached_templates(
                    p1=p1,
                    p2=p2,
                    p_meas=p_meas,
                    p_prep=p_prep,
                )
            if len(self._patches) == 2:
                return self._build_structured_cx_dem_from_cached_templates(
                    p1=p1,
                    p2=p2,
                    p_meas=p_meas,
                    p_prep=p_prep,
                )
            return None
        if len(self._patches) != 1 or len(self._operations) != 1:
            return None

        operation = self._operations[0]
        if operation.gate_type != LogicalGateType.MEMORY or len(operation.patches) != 1 or operation.rounds < 2:
            return None

        patch_label = operation.patches[0]
        patch_state = self._patches[patch_label]
        geometry = patch_state.patch.geometry
        basis = operation.per_patch_basis.get(patch_label, operation.basis).upper()
        coord_x, coord_y = patch_state.coord_offset
        templates = _cached_surface_memory_dem_templates(
            geometry.dx,
            geometry.dz,
            geometry.orientation.name,
            geometry.rotated,
            basis,
            p1,
            p2,
            p_meas,
            p_prep,
        )

        from pecos_rslib.qec import DemSliceRoundSchedule

        instances = [(templates.initialization, 0)]
        instances.extend((templates.bulk, round_) for round_ in range(1, operation.rounds - 1))
        instances.extend(
            [
                (templates.pre_terminal, operation.rounds - 1),
                (templates.terminal, operation.rounds),
            ],
        )
        schedule = DemSliceRoundSchedule.from_templates(
            templates.output_model,
            instances,
            coordinate_offset=(float(coord_x), float(coord_y)),
        )
        model = schedule.stitch(
            start_round=0,
            commit_rounds=operation.rounds + 1,
            buffer_rounds=0,
            forward_boundary="hard",
        )
        return model, schedule

    def _build_structured_cx_dem_from_cached_templates(
        self,
        *,
        p1: float,
        p2: float,
        p_meas: float,
        p_prep: float,
    ) -> tuple[object, object] | None:
        """Assemble an eligible two-patch memory-CX-memory DEM."""
        before, gate, after = self._operations
        if (
            before.gate_type != LogicalGateType.MEMORY
            or gate.gate_type != LogicalGateType.TRANSVERSAL_CX
            or after.gate_type != LogicalGateType.MEMORY
            or before.rounds < 2
            or after.rounds < 2
            or len(before.patches) != 2
            or len(gate.patches) != 2
            or len(after.patches) != 2
        ):
            return None

        control_label, target_label = gate.patches
        patch_order = list(self._patches)
        if before.patches != patch_order or gate.patches != patch_order or after.patches != patch_order:
            return None

        control_state = self._patches[control_label]
        target_state = self._patches[target_label]
        control_geometry = control_state.patch.geometry
        target_geometry = target_state.patch.geometry
        if (
            control_geometry.dx != target_geometry.dx
            or control_geometry.dz != target_geometry.dz
            or control_geometry.rotated != target_geometry.rotated
        ):
            return None
        initial_control_basis = before.per_patch_basis.get(control_label, before.basis).upper()
        initial_target_basis = before.per_patch_basis.get(target_label, before.basis).upper()
        final_control_basis = after.per_patch_basis.get(control_label, after.basis).upper()
        final_target_basis = after.per_patch_basis.get(target_label, after.basis).upper()
        cached = _cached_surface_cx_dem_templates(
            control_geometry.dx,
            control_geometry.dz,
            control_geometry.orientation.name,
            control_geometry.rotated,
            target_geometry.dx,
            target_geometry.dz,
            target_geometry.orientation.name,
            target_geometry.rotated,
            initial_control_basis,
            initial_target_basis,
            final_control_basis,
            final_target_basis,
            p1,
            p2,
            p_meas,
            p_prep,
        )
        templates = cached.templates

        from pecos_rslib.qec import DemSliceRoundSchedule

        instances, terminal_round = _boundary_template_instances(templates, before.rounds, after.rounds)

        control_x, control_y = control_state.coord_offset
        target_x, target_y = target_state.coord_offset
        target_origin_x, target_origin_y = cached.target_coordinate_origin
        detector_coordinate_offsets = {
            stream: (float(control_x), float(control_y)) for stream in range(cached.control_stream_count)
        }
        detector_coordinate_offsets.update(
            {
                stream: (float(target_x) - target_origin_x, float(target_y) - target_origin_y)
                for stream in range(
                    cached.control_stream_count,
                    cached.control_stream_count + cached.target_stream_count,
                )
            },
        )
        schedule = DemSliceRoundSchedule.from_templates(
            templates.output_model,
            instances,
            detector_coordinate_offsets=detector_coordinate_offsets,
        )
        model = schedule.stitch(
            start_round=0,
            commit_rounds=terminal_round + 1,
            buffer_rounds=0,
            forward_boundary="hard",
        )
        return model, schedule

    def _build_structured_h_dem_from_cached_templates(
        self,
        *,
        p1: float,
        p2: float,
        p_meas: float,
        p_prep: float,
    ) -> tuple[object, object] | None:
        """Assemble an eligible memory-H-memory DEM from bounded templates."""
        before, gate, after = self._operations
        if (
            before.gate_type != LogicalGateType.MEMORY
            or gate.gate_type != LogicalGateType.TRANSVERSAL_H
            or after.gate_type != LogicalGateType.MEMORY
            or before.rounds < 2
            or after.rounds < 2
            or len(before.patches) != 1
            or len(gate.patches) != 1
            or len(after.patches) != 1
            or before.patches[0] != gate.patches[0]
            or gate.patches[0] != after.patches[0]
        ):
            return None

        patch_label = gate.patches[0]
        patch_state = self._patches[patch_label]
        geometry = patch_state.patch.geometry
        initial_basis = before.per_patch_basis.get(patch_label, before.basis).upper()
        final_basis = after.per_patch_basis.get(patch_label, after.basis).upper()
        coord_x, coord_y = patch_state.coord_offset
        templates = _cached_surface_h_dem_templates(
            geometry.dx,
            geometry.dz,
            geometry.orientation.name,
            geometry.rotated,
            initial_basis,
            final_basis,
            p1,
            p2,
            p_meas,
            p_prep,
        )

        from pecos_rslib.qec import DemSliceRoundSchedule

        instances, terminal_round = _boundary_template_instances(templates, before.rounds, after.rounds)
        schedule = DemSliceRoundSchedule.from_templates(
            templates.output_model,
            instances,
            coordinate_offset=(float(coord_x), float(coord_y)),
        )
        model = schedule.stitch(
            start_round=0,
            commit_rounds=terminal_round + 1,
            buffer_rounds=0,
            forward_boundary="hard",
        )
        return model, schedule

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
        No Stim dependency. Eligible single-patch memories, transversal-H, and
        two-patch transversal-CX algorithms reuse bounded physical-template
        compiles across requested memory lengths.

        Args:
            p1: Single-qubit depolarizing error rate.
            p2: Two-qubit depolarizing error rate.
            p_meas: Measurement error rate.
            p_prep: Preparation error rate.

        Returns:
            DEM string in Stim-compatible format.
        """
        cached = self._build_structured_dem_from_cached_templates(
            p1=p1,
            p2=p2,
            p_meas=p_meas,
            p_prep=p_prep,
        )
        if cached is None:
            dem, _, _ = self._build_structured_dem(p1=p1, p2=p2, p_meas=p_meas, p_prep=p_prep)
        else:
            dem, _ = cached
        return str(dem)

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
        from pecos_rslib.qec import LogicalSubgraphDecoder

        cached = self._build_structured_dem_from_cached_templates(
            p1=p1,
            p2=p2,
            p_meas=p_meas,
            p_prep=p_prep,
        )
        if cached is None:
            dem, _, _ = self._build_structured_dem(p1=p1, p2=p2, p_meas=p_meas, p_prep=p_prep)
        else:
            dem, _ = cached
        sampler = dem.to_sampler()
        dem_str = str(dem)

        sc = self.stab_coords()
        decoder = LogicalSubgraphDecoder(dem_str, sc, inner_decoder)

        return sampler, decoder, dem_str

    def build_algorithm_descriptor(
        self,
        *,
        p1: float = 0.001,
        p2: float = 0.001,
        p_meas: float = 0.001,
        p_prep: float = 0.0,
        buffer: int | None = None,
    ) -> dict:
        """Extract per-segment DEMs and boundary gates for LogicalAlgorithmDecoder.

        Splits the structured circuit DEM at gate boundaries. Each memory operation
        becomes a segment; each transversal gate becomes a boundary gate with
        Pauli frame propagation rules. By default, each non-terminal segment
        derives the minimum safe look-ahead from the source-tracked model.
        Passing ``buffer`` requests that exact amount of look-behind and
        look-ahead; a value below the model's required look-ahead is rejected.
        Eligible single-patch memories, memory-H-memory algorithms, and
        two-patch memory-CX-memory algorithms are assembled directly from
        bounded physical-template caches; other circuits retain full-model
        fallback.

        Returns:
            Dict with keys: segments, boundary_gates, num_observables, full_dem.
        """
        if buffer is not None and buffer < 0:
            msg = "buffer must be non-negative or None"
            raise ValueError(msg)

        # Eligible memory, memory-H-memory, and two-patch memory-CX-memory
        # algorithms are assembled entirely from bounded physical-template
        # caches. Other algorithms retain the full structured path as an
        # equivalence oracle and fallback until their logical-operation families
        # are available.
        cached = self._build_structured_dem_from_cached_templates(
            p1=p1,
            p2=p2,
            p_meas=p_meas,
            p_prep=p_prep,
        )
        if cached is None:
            structured_dem, influence_map, dag_circuit = self._build_structured_dem(
                p1=p1,
                p2=p2,
                p_meas=p_meas,
                p_prep=p_prep,
            )
            round_schedule = structured_dem.round_schedule(influence_map, dag_circuit)
        else:
            structured_dem, round_schedule = cached
        full_dem = str(structured_dem)
        sc = self.stab_coords()

        # Compute segment time boundaries from operations.
        # Each MEMORY op has a number of rounds. Time coordinates are
        # sequential round indices across all segments.
        segments = []
        boundary_gates = []
        # Gates accumulate between consecutive MEMORY ops.
        pending_gates = []
        time_cursor = 0
        patch_labels = list(self._patches.keys())
        num_patches = len(patch_labels)

        # Track X/Z swap state per patch for stab_coords.
        # After transversal H, the X and Z stabilizer types swap.
        x_z_swapped = dict.fromkeys(patch_labels, False)

        for op in self._operations:
            if op.gate_type == LogicalGateType.MEMORY:
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
                        "z_obs_bit": idx * 2 + 1,
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
                            "z_obs_bit": ctrl_idx * 2 + 1,
                            "ancilla_z_bit": tgt_idx * 2 + 1,
                        },
                    )
                else:
                    pending_gates.append(
                        {
                            "type": "Cnot",
                            "ctrl_x_bit": ctrl_idx * 2,
                            "ctrl_z_bit": ctrl_idx * 2 + 1,
                            "tgt_x_bit": tgt_idx * 2,
                            "tgt_z_bit": tgt_idx * 2 + 1,
                        },
                    )

            elif op.gate_type in (LogicalGateType.TRANSVERSAL_SZ, LogicalGateType.TRANSVERSAL_SZdg):
                label = op.patches[0]
                idx = patch_labels.index(label)
                pending_gates.append(
                    {
                        "type": "SGate",
                        "x_obs_bit": idx * 2,
                        "z_obs_bit": idx * 2 + 1,
                    },
                )

        # Assemble each sub-DEM through the native slice schedule. This keeps
        # independent source contributions, decomposition metadata, logical
        # outputs, hyperedges, and cross-round correlations intact.
        seg_dems = []
        segment_detector_counts = []
        detector_rounds = [int(coords[2]) for _, coords in structured_dem.detector_coordinates() if coords is not None]
        explicit_buffer = 0 if buffer is None else buffer
        for segment_index, seg in enumerate(segments):
            start_round = max(0, int(seg["time_start"]) - explicit_buffer)
            is_last = segment_index == len(segments) - 1
            commit_end = time_cursor + 1 if is_last else int(seg["time_end"])
            commit_rounds = commit_end - start_round
            forward_buffer = 0 if is_last else buffer
            forward_boundary = "hard" if is_last else "soft"

            if not is_last and buffer is not None:
                required = round_schedule.required_buffer_rounds(
                    start_round,
                    commit_rounds,
                )
                if buffer < required:
                    msg = (
                        f"buffer={buffer} is too small for logical segment {segment_index}; "
                        f"the source-tracked DEM requires at least {required} look-ahead rounds"
                    )
                    raise ValueError(msg)

            segment_dem = round_schedule.stitch(
                start_round=start_round,
                commit_rounds=commit_rounds,
                buffer_rounds=forward_buffer,
                forward_boundary=forward_boundary,
            )
            seg_dems.append(str(segment_dem))
            # Segment metadata partitions the incoming full-circuit syndrome;
            # it therefore counts only this segment's commit detectors, not the
            # look-behind/look-ahead detectors duplicated in its local DEM.
            segment_detector_counts.append(
                sum(int(seg["time_start"]) <= round_ < commit_end for round_ in detector_rounds),
            )

        # Physical code distance for latency/windowing decisions. With multiple
        # patches use the minimum (the weakest bound governs latency). This is the
        # real surface-code distance, NOT a count of logical patches.
        distance = min(
            (min(ps.patch.geometry.dx, ps.patch.geometry.dz) for ps in self._patches.values()),
            default=0,
        )

        return {
            "segments": [
                {
                    "dem": seg_dems[i],
                    "num_detectors": segment_detector_counts[i],
                    "stab_coords": segments[i]["stab_coords"],
                }
                for i in range(len(segments))
            ],
            "boundary_gates": boundary_gates,
            "num_observables": num_patches * 2,
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

        self.tc = TickCircuit()
        self._current_tick = None
        self._allocated: set[int] = set()
        self.meas_count = 0

        self.stab_meas: dict[tuple[str, str, int, int, int], int] = {}
        self.data_meas: dict[tuple[str, int], int] = {}

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
        if self._current_tick is not None:
            round_owner = int(self.round_time)
            if float(round_owner) != self.round_time:
                msg = f"DEM slice round owner must be integral, got {self.round_time}"
                raise ValueError(msg)
            tick_idx = self._current_tick.index()
            tick = self.tc.get_tick(tick_idx)
            if tick is not None:
                for gate_idx in range(tick.gate_batch_count()):
                    self.tc.set_gate_meta(
                        tick_idx,
                        gate_idx,
                        DEM_SLICE_ROUND_ATTRIBUTE,
                        round_owner,
                    )
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
        is_first = True
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
                    is_first=is_first,
                    is_last=bool(last_patches),
                    last_patches=last_patches,
                )
                is_first = False
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

        self.tc.set_meta("detectors", json.dumps(det_out))
        self.tc.set_meta("observables", json.dumps(obs_out))
        self.tc.set_meta("num_measurements", str(total))
        return self.tc

    def _first_memory_basis(self, patch_label: str | None = None) -> str:
        """Basis of the first memory segment (prep basis)."""
        for op in self.operations:
            if op.gate_type == LogicalGateType.MEMORY:
                if patch_label and patch_label in op.per_patch_basis:
                    return op.per_patch_basis[patch_label]
                return op.basis
        return "Z"

    def _last_memory_basis(self, patch_label: str | None = None) -> str:
        """Basis of the last memory segment (measurement basis)."""
        last = "Z"
        for op in self.operations:
            if op.gate_type == LogicalGateType.MEMORY:
                if patch_label and patch_label in op.per_patch_basis:
                    last = op.per_patch_basis[patch_label]
                else:
                    last = op.basis
        return last

    def _emit_meas(self, qubits: list[int]) -> list[int]:
        self._tick().mz(qubits)
        indices = list(range(self.meas_count, self.meas_count + len(qubits)))
        self.meas_count += len(qubits)
        return indices

    def _emit_final_meas(self, qubits: list[int]) -> list[int]:
        self._tick().mz(qubits)
        indices = list(range(self.meas_count, self.meas_count + len(qubits)))
        self.meas_count += len(qubits)
        return indices

    def _rec(self, abs_idx: int) -> int:
        return abs_idx - self.meas_count

    def _data_qubits(self, patch_label: str) -> list[int]:
        ps = self.patches[patch_label]
        return [ps.qubit_offset + i for i in range(ps.patch.geometry.num_data)]

    def _emit_memory_segment(
        self,
        op: LogicalOp,
        *,
        is_first: bool,
        is_last: bool,
        last_patches: set[str] | None = None,
    ) -> None:
        """Emit syndrome extraction rounds for one or more patches."""
        from pecos.qec.surface.schedule import compute_cnot_schedule

        num_rounds = op.rounds

        # Precompute per-patch data
        patch_info = []
        for patch_label in op.patches:
            ps = self.patches[patch_label]
            patch = ps.patch
            geom = patch.geometry
            offset = ps.qubit_offset
            num_x = len(geom.x_stabilizers)
            num_z = len(geom.z_stabilizers)
            anc_base = offset + geom.num_data

            if ps.x_z_swapped:
                x_anc_qs = [anc_base + num_x + i for i in range(num_z)]
                z_anc_qs = [anc_base + i for i in range(num_x)]
                current_x_stabs = geom.z_stabilizers
                current_z_stabs = geom.x_stabilizers
            else:
                x_anc_qs = [anc_base + i for i in range(num_x)]
                z_anc_qs = [anc_base + num_x + i for i in range(num_z)]
                current_x_stabs = geom.x_stabilizers
                current_z_stabs = geom.z_stabilizers

            patch_info.append(
                {
                    "label": patch_label,
                    "ps": ps,
                    "geom": geom,
                    "offset": offset,
                    "num_x": num_x,
                    "anc_base": anc_base,
                    "data_qs": [offset + i for i in range(geom.num_data)],
                    "x_anc_qs": x_anc_qs,
                    "z_anc_qs": z_anc_qs,
                    "current_x_stabs": current_x_stabs,
                    "current_z_stabs": current_z_stabs,
                    "schedule": compute_cnot_schedule(patch),
                },
            )

        # Initialization — per-patch basis
        if is_first:
            t = self._new_tick()
            for pi in patch_info:
                self._emit_qalloc_or_reset(pi["data_qs"])
            self._end_tick()

            need_h = []
            need_hs = []
            for pi in patch_info:
                pb = op.per_patch_basis.get(pi["label"], op.basis)
                if pb == "X":
                    need_h.extend(pi["data_qs"])
                elif pb == "Y":
                    need_hs.extend(pi["data_qs"])

            if need_h or need_hs:
                t = self._new_tick()
                if need_h:
                    t.h(need_h)
                if need_hs:
                    t.h(need_hs)
                self._end_tick()
            if need_hs:
                t = self._new_tick()
                t.sz(need_hs)
                self._end_tick()

        # Syndrome extraction rounds
        for rnd in range(num_rounds):
            # Reset ancillas
            t = self._new_tick()
            for pi in patch_info:
                self._emit_qalloc_or_reset(pi["x_anc_qs"] + pi["z_anc_qs"])
            self._end_tick()

            # H on X-type ancillas
            all_x_anc = [q for pi in patch_info for q in pi["x_anc_qs"]]
            t = self._new_tick()
            t.h(all_x_anc)
            self._end_tick()

            # CX rounds
            num_cx_rounds = max(len(pi["schedule"]) for pi in patch_info)
            for cx_round_idx in range(num_cx_rounds):
                all_pairs = []
                for pi in patch_info:
                    if cx_round_idx >= len(pi["schedule"]):
                        continue
                    for phys_type, stab_idx, data_idx in pi["schedule"][cx_round_idx]:
                        data_q = pi["offset"] + data_idx
                        if phys_type == "X":
                            anc_q = pi["anc_base"] + stab_idx
                        else:
                            anc_q = pi["anc_base"] + pi["num_x"] + stab_idx
                        currently_x = (phys_type == "X") != pi["ps"].x_z_swapped
                        if currently_x:
                            all_pairs.append((anc_q, data_q))
                        else:
                            all_pairs.append((data_q, anc_q))
                t = self._new_tick()
                if all_pairs:
                    t.cx(all_pairs)
                self._end_tick()

            # H on X-type ancillas
            t = self._new_tick()
            t.h(all_x_anc)
            self._end_tick()

            # Measure ancillas
            t = self._new_tick()
            for pi in patch_info:
                x_meas = self._emit_meas(pi["x_anc_qs"])
                z_meas = self._emit_meas(pi["z_anc_qs"])
                for i, s in enumerate(pi["current_x_stabs"]):
                    self.stab_meas[(pi["label"], "X", s.index, self.segment_idx, rnd)] = x_meas[i]
                for i, s in enumerate(pi["current_z_stabs"]):
                    self.stab_meas[(pi["label"], "Z", s.index, self.segment_idx, rnd)] = z_meas[i]
            # Invalidate _last_round_cache since stab_meas changed
            if hasattr(self, "_last_round_cache"):
                del self._last_round_cache
            self._end_tick()

            # Detectors
            for pi in patch_info:
                self._emit_round_detectors(pi["label"], rnd, is_first_segment=is_first)

            self.round_time += 1.0

        # Final measurement: two phases so cross-patch observable
        # references work (all data measurements must exist before
        # any observable is emitted).
        if is_last and last_patches:
            final_patches = [pi for pi in patch_info if pi["label"] in last_patches]
            for pi in final_patches:
                self._emit_final_data_measurements(pi["label"])
            for pi in final_patches:
                self._emit_final_detectors_and_observables(pi["label"])

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

                if round_idx == 0 and is_first_segment and seg == 0:
                    # First round of very first segment:
                    # Only stabilizers matching the prep basis are deterministic.
                    # Find the prep basis from the first memory operation.
                    init_basis = self._first_memory_basis(patch_label)
                    det_type = "Z" if init_basis == "Z" else "X"
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
        stabs = geom.x_stabilizers if stab_type == "X" else geom.z_stabilizers
        s = stabs[stab_index]
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

    def _emit_transversal_h(self, op: LogicalOp) -> None:
        ps = self.patches[op.patches[0]]
        t = self._new_tick()
        t.h(self._data_qubits(op.patches[0]))
        self._end_tick()
        ps.x_z_swapped = not ps.x_z_swapped

    def _emit_transversal_sz(self, op: LogicalOp) -> None:
        t = self._new_tick()
        t.sz(self._data_qubits(op.patches[0]))
        self._end_tick()

    def _emit_transversal_szdg(self, op: LogicalOp) -> None:
        t = self._new_tick()
        t.szdg(self._data_qubits(op.patches[0]))
        self._end_tick()

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

        pairs = list(zip(self._data_qubits(ctrl_label), self._data_qubits(tgt_label), strict=False))
        t = self._new_tick()
        t.cx(pairs)
        self._end_tick()

        if op.teleportation:
            ctrl_ps.z_obs_includes.append(tgt_label)

        # Track entanglement: CX spreads X on control to target,
        # and Z on target to control.
        ctrl_ps.x_entangled_with.append(tgt_label)
        tgt_ps.z_entangled_with.append(ctrl_label)

    def _emit_final_data_measurements(self, patch_label: str) -> None:
        ps = self.patches[patch_label]
        geom = ps.patch.geometry
        data_qs = self._data_qubits(patch_label)
        meas_basis = self._last_memory_basis(patch_label)

        if meas_basis == "X":
            t = self._new_tick()
            t.h(data_qs)
            self._end_tick()

        t = self._new_tick()
        meas_indices = self._emit_final_meas(data_qs)
        self._end_tick()
        for i, q in enumerate(range(geom.num_data)):
            self.data_meas[(patch_label, q)] = meas_indices[i]

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

        if logical_op is not None:
            # Check if this observable is reliable given entanglement.
            # After CX(ctrl, tgt): ctrl's X is entangled with tgt,
            # tgt's Z is entangled with ctrl.
            # An observable is reliable if:
            # - Not entangled, OR
            # - The entangled partner is measured in the same basis
            entangled_with = ps.x_entangled_with if meas_basis == "X" else ps.z_entangled_with
            is_reliable = True
            for other_label in entangled_with:
                other_basis = self._last_memory_basis(other_label)
                if other_basis != meas_basis:
                    is_reliable = False
                    break

            if not is_reliable:
                # Skip non-reliable observables — they're physically
                # non-deterministic and would cause Stim DEM errors.
                # The decoder handles these through the 3-body detectors.
                self.next_observable_idx += 1
                return

            obs_idx = self.next_observable_idx
            self.next_observable_idx += 1
            obs_indices = [self.data_meas[(patch_label, q)] for q in logical_op.data_qubits]

            # Teleportation corrections
            for other_label in ps.z_obs_includes:
                other_logical = self.patches[other_label].patch.geometry.logical_z
                if other_logical is not None:
                    for q in other_logical.data_qubits:
                        key = (other_label, q)
                        if key in self.data_meas:
                            obs_indices.append(self.data_meas[key])

            self._obs_json.append(
                {
                    "id": obs_idx,
                    "abs_records": list(obs_indices),
                },
            )
