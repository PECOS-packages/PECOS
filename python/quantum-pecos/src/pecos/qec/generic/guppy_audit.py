"""Inspect a compiled Guppy implementation without inferring physical timing."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pecos.qec.generic.guppy_ordering import GuppyOrdering

if TYPE_CHECKING:
    from pecos.qec.dem_spec import GuppyDemBuild
    from pecos.qec.generic.guppy_ordering import GuppyOrderingPlan
    from pecos.qec.generic.scheduled import ScheduledCircuit


@dataclass(frozen=True)
class GuppyImplementationAudit:
    """Source schedule alongside the measured runtime implementation.

    Analysis replay ticks and reference physical moments are different units.
    The existing DEM builder normalizes Clifford rotations and simplifies chains;
    its circuit is a runtime-derived analysis representation, not a raw pulse log. A complete
    measurement binding does not certify timing or noise-model equivalence.
    ``build`` exposes the annotated runtime circuit and measurement ledger.
    """

    reference_moments: int
    reference_carriers: int
    reference_peak_live: int
    reference_gate_counts: tuple[tuple[str, int], ...]
    analysis_replay_ticks: int
    analysis_gate_counts: tuple[tuple[str, int], ...]
    build: GuppyDemBuild = field(repr=False, compare=False)
    ordering_plan: GuppyOrderingPlan | None = None

    def to_dict(self) -> dict:
        """Return a JSON-compatible receipt with explicit evidence boundaries."""
        return {
            "reference_moments": self.reference_moments,
            "reference_carriers": self.reference_carriers,
            "reference_peak_live": self.reference_peak_live,
            "reference_gate_counts": dict(self.reference_gate_counts),
            "analysis_replay_ticks": self.analysis_replay_ticks,
            "analysis_gate_counts": dict(self.analysis_gate_counts),
            "measurement_binding": self.build.audit,
            "physical_schedule_verified": False,
            "noise_equivalence_verified": False,
            "scope": "Clifford-normalized runtime analysis and compiler-bound identities; no timing certificate",
            "runtime_circuit_representation": "clifford_normalized_for_dem_analysis",
            "requested_ordering": None if self.ordering_plan is None else self.ordering_plan.to_dict(),
        }

    def describe(self) -> str:
        """Summarize the implementation in human-readable units."""
        return "\n".join(
            (
                f"Reference: {self.reference_moments} physical moments, {self.reference_carriers} carrier roles, "
                f"{self.reference_peak_live} peak live qubits",
                f"Normalized runtime analysis: {self.analysis_replay_ticks} replay ticks, "
                f"{len(self.build.measurement_ledger)} measurements",
                "Reference operations: " + ", ".join(f"{name}={count}" for name, count in self.reference_gate_counts),
                "Normalized operations: " + ", ".join(f"{name}={count}" for name, count in self.analysis_gate_counts),
                f"Named result binding: {self.build.named_result_binding}",
                "Requested ordering: "
                + ("unspecified" if self.ordering_plan is None else self.ordering_plan.describe()),
                "Physical schedule and noise equivalence: not established by this trace",
            ),
        )


def audit_guppy_implementation(
    circuit: ScheduledCircuit,
    *,
    runtime: object | None = None,
    seed: int = 0,
    ordering: GuppyOrdering = GuppyOrdering.FUNCTIONAL,
) -> GuppyImplementationAudit:
    """Compile and trace a plan, retaining the runtime circuit and typed parities.

    The existing Guppy DEM builder validates straight-line control flow and
    compiler-bound named measurement identities. Noise rates are explicitly zero;
    no source SI1000 profile is implicitly attached to compiler-lowered pulses.
    Runtime idles are retained, while Clifford rotations/chains undergo the
    standard DEM analysis normalization. Use capture_qis_operation_trace() when
    inspecting the original emitted pulse stream.
    """
    from pecos.qec.dem import build_dem_from_guppy  # noqa: PLC0415 - optional Guppy backend

    detectors, observables = circuit.dem_spec()
    build = build_dem_from_guppy(
        circuit.guppy_program(ordering=ordering),
        num_qubits=circuit.num_qubits,
        detectors=detectors,
        observables=observables,
        runtime=runtime,
        seed=seed,
        p1=0.0,
        p2=0.0,
        p_meas=0.0,
        p_prep=0.0,
        strip_traced_idles=False,
    )
    if len(build.measurement_ledger) != circuit.num_measurements:
        msg = "Compiled Guppy measurement count differs from the scheduled contract"
        raise ValueError(msg)
    reference = Counter(op.name for moment in circuit.moments for op in moment)
    # Native counts expand batches into individual operations.
    lowered = build.circuit.gate_counts_by_type()
    return GuppyImplementationAudit(
        len(circuit.moments),
        circuit.num_qubits,
        circuit.resources.peak_live,
        tuple(sorted(reference.items())),
        build.circuit.num_ticks(),
        tuple(sorted(lowered.items())),
        build,
        circuit.guppy_ordering_plan(ordering),
    )
