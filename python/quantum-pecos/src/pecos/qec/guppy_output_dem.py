"""Empirical DEM annotations inferred from Guppy parity outputs.

This module is deliberately code-agnostic.  It treats a Guppy program as the
owner of its detector/observable post-processing and learns the corresponding
affine GF(2) functions from PECOS coin-toss executions.  A QIS trace then binds
the program's raw-measurement output to stable ``MeasId`` values.

The inference is empirical, not a compiler proof.  It therefore checks the
fit using additional independent consistency equations and fails unless the raw tag
covers every physical measurement with a unique result-ID binding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pecos_rslib.qec import DemInferenceSettings, _prepare_guppy_dem, _SeleneRuntimeSpec

from pecos._engine_builders import _configure_selene_runtime
from pecos.programs import Guppy, Hugr
from pecos.qec import dem_spec
from pecos.quantum.hosted import _hosted_tick_separation_limit

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from pecos_rslib import Qis
    from pecos_rslib.qec import DetectorErrorModel, _PreparedDem
    from pecos_rslib.quantum import TickCircuit


@dataclass(frozen=True, slots=True)
class InferredGuppyDemAnnotations:
    """Inference evidence backed by a shared Rust preparation.

    In mixed builds, ``circuit``, ``detectors_json``, ``observables_json``, and
    ``build_dem()`` cover the whole prepared schema, including explicit
    definitions. Supports and labels describe the inferred definitions only.
    Circuit access returns a detached snapshot.
    """

    _prepared: _PreparedDem

    @property
    def circuit(self) -> TickCircuit:
        """Return a detached snapshot of the prepared circuit."""
        return self._prepared.circuit

    @property
    def detectors_json(self) -> str:
        """Return detector metadata with computed references."""
        return self._prepared.detectors_json

    @property
    def observables_json(self) -> str:
        """Return observable metadata with computed references."""
        return self._prepared.observables_json

    @property
    def raw_measurement_ids(self) -> tuple[int, ...]:
        """Return the prepared inference raw measurement ids."""
        data = self._prepared.inference["raw_measurement_ids"]
        return tuple(data)

    @property
    def detector_supports(self) -> tuple[tuple[int, ...], ...]:
        """Return the prepared inference detector supports."""
        data = self._prepared.inference["detector_supports"]
        return tuple(tuple(support) for support in data)

    @property
    def observable_supports(self) -> tuple[tuple[int, ...], ...]:
        """Return the prepared inference observable supports."""
        data = self._prepared.inference["observable_supports"]
        return tuple(tuple(support) for support in data)

    @property
    def observable_labels(self) -> tuple[tuple[str, int], ...]:
        """Return the prepared inference observable labels."""
        data = self._prepared.inference["observable_labels"]
        return tuple(data or ())

    @property
    def detector_labels(self) -> tuple[tuple[str, int], ...]:
        """Return the prepared inference detector labels."""
        data = self._prepared.inference["detector_labels"]
        return tuple(data or ())

    @property
    def probe_shots(self) -> int:
        """Return the prepared inference probe shots."""
        return self._prepared.inference["probe_shots"]

    @property
    def validation_rows(self) -> int:
        """Return the prepared inference validation rows."""
        return self._prepared.inference["validation_rows"]

    @property
    def provenance_shots(self) -> int:
        """Return the prepared inference provenance shots."""
        return self._prepared.inference["provenance_shots"]

    @property
    def schedule_validation_shots(self) -> int:
        """Return the prepared inference schedule validation shots."""
        return self._prepared.inference["schedule_validation_shots"]

    @property
    def raw_binding(self) -> str:
        """Return the prepared inference raw binding."""
        return self._prepared.inference["raw_binding"]

    @property
    def raw_tag(self) -> str:
        """Return the prepared inference raw tag."""
        return self._prepared.inference["raw_tag"]

    @property
    def schedule_validation(self) -> str:
        """Return the prepared inference schedule validation."""
        return self._prepared.inference["schedule_validation"]

    @property
    def result_traces(self) -> tuple[Mapping[str, Any], ...]:
        """Return the prepared inference result traces."""
        data = self._prepared.inference["result_traces"]
        return tuple(data)

    @property
    def source_result_traces(self) -> tuple[Mapping[str, Any], ...]:
        """Return the prepared inference source result traces."""
        data = self._prepared.inference["source_result_traces"]
        return tuple(data)

    @property
    def audit(self) -> dict[str, Any]:
        """Describe the empirical evidence without claiming a compiler proof."""
        return self._prepared.audit["inference"]

    def build_dem(self, **noise: object) -> DetectorErrorModel:
        """Build a DEM with the full ``from_circuit`` noise keyword set."""
        return self._prepared.build_dem(**noise).dem


def infer_guppy_dem_annotations(
    program: object,
    *,
    num_qubits: int,
    raw_tag: str = "raw measurements",
    detector_tag: str = "DETECTOR",
    observable_tags: Sequence[str] = ("obs",),
    probe_shots: int | None = None,
    provenance_shots: int = 32,
    validation_rows: int = 32,
    seed: int = 0,
    runtime: object | None = None,
    require_raw_provenance: bool = True,
    require_hosted_operation_order: bool = False,
    max_hosted_tick_separation: int | None = None,
    detector_outputs: Iterable[tuple[str, int]] | None = None,
    observable_outputs: Iterable[tuple[str, int]] | None = None,
) -> InferredGuppyDemAnnotations:
    """Infer parity annotations from an untouched Guppy program.

    The program must emit every physical measurement exactly once
    through one or more ``output(raw_tag, ...)`` calls. Detector and
    observable outputs may be computed XOR expressions.  Coin-toss execution
    makes the physical results independent GF(2) variables; Gaussian
    elimination recovers each emitted parity and extra rows validate it.

    This prototype is suitable only when measurement values do not alter the
    quantum operation schedule.  PECOS captures one QIS path for the returned
    circuit; the affine checks certify classical parity processing, not static
    quantum control flow.  ``require_raw_provenance`` defaults to true.  Its
    opt-in false setting assumes raw output order equals QIS measurement order
    and records that weaker binding in the returned object and circuit. When
    direct runtime result IDs are unavailable, the default mode correlates raw
    output columns with result-ID keyed physical outcomes across independent
    trace probes and requires a unique complete bijection.
    Hosted-order validation runs after all probes, so failures incur the probe
    run cost and raw-tag errors take precedence.
    """
    settings = (
        DemInferenceSettings()
        .with_raw_tag(raw_tag)
        .with_detector_tag(detector_tag)
        .with_observable_tags(observable_tags)
        .with_provenance_shots(provenance_shots)
        .with_validation_rows(validation_rows)
        .with_require_raw_provenance(require_raw_provenance)
    )
    if probe_shots is not None:
        settings = settings.with_probe_shots(probe_shots)
    settings.validate()

    # The binding takes (tag, element) tuples; read each selector once.
    selections = []
    for selectors in (detector_outputs, observable_outputs):
        if selectors is None:
            selections.append(None)
            continue
        pairs = [(tag, element) for tag, element in selectors]
        for tag, element in pairs:
            dem_spec.InferredOutput(tag, element)
        selections.append(pairs)
    detector_outputs, observable_outputs = selections
    if (
        detector_outputs is not None
        and observable_outputs is not None
        and not detector_outputs
        and not observable_outputs
    ):
        msg = "no outputs selected for inference"
        raise ValueError(msg)

    runtime_spec = _configure_selene_runtime(_SeleneRuntimeSpec(), runtime)
    max_hosted_tick_separation = _hosted_tick_separation_limit(
        max_hosted_tick_separation,
        context="infer_guppy_dem_annotations",
    )
    definitions = {
        role: None if outputs is None else [(None, None, None, (tag, element), None, "{}") for tag, element in outputs]
        for role, outputs in (("detectors", detector_outputs), ("observables", observable_outputs))
    }
    prepared = _prepare_guppy_dem(
        _guppy_dem_program(program),
        num_qubits,
        seed,
        runtime_spec,
        settings=settings,
        require_hosted_operation_order=require_hosted_operation_order,
        max_hosted_tick_separation=max_hosted_tick_separation,
        hosted_context="infer_guppy_dem_annotations",
        **definitions,
    )
    return InferredGuppyDemAnnotations(prepared)


def _guppy_dem_program(program: object) -> Qis:
    """Compile boundary program carriers through the shared HUGR/QIS path."""
    if isinstance(program, (bytes, bytearray)):
        program = Hugr(bytes(program))
    elif hasattr(program, "compile") and not hasattr(program, "_to_program"):
        program = Guppy(program)
    to_program = getattr(program, "_to_program", None)
    return to_program() if to_program is not None else program


__all__ = ["DemInferenceSettings", "InferredGuppyDemAnnotations", "infer_guppy_dem_annotations"]
