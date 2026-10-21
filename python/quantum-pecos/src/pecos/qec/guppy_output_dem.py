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

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pecos_rslib.qec import DemInferenceSettings, _capture_dem_evidence, _SeleneRuntimeSpec

from pecos._engine_builders import _configure_selene_runtime
from pecos._qis_trace_replay import _validate_trace_hosted_operations_if_requested
from pecos._traced_circuit import normalize_traced_tick_circuit
from pecos.programs import Guppy

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from pecos_rslib.qec import DetectorErrorModel


@dataclass(frozen=True, slots=True)
class InferredGuppyDemAnnotations:
    """An annotated QIS trace and its inferred detector/observable schema."""

    circuit: Any
    detectors_json: str
    observables_json: str
    raw_measurement_ids: tuple[int, ...]
    detector_supports: tuple[tuple[int, ...], ...]
    observable_supports: tuple[tuple[int, ...], ...]
    observable_labels: tuple[tuple[str, int], ...]
    probe_shots: int
    raw_binding: str
    validation_rows: int = 32
    provenance_shots: int = 32
    raw_tag: str = "raw measurements"
    schedule_validation: str = "unspecified"
    schedule_validation_shots: int = 0
    result_traces: tuple[Mapping[str, Any], ...] = ()
    source_result_traces: tuple[Mapping[str, Any], ...] = ()
    detector_labels: tuple[tuple[str, int], ...] = ()

    @property
    def audit(self) -> dict[str, Any]:
        """Describe the empirical evidence without claiming a compiler proof."""
        return {
            "method": "coin_toss_affine_inference_v2",
            "raw_binding": self.raw_binding,
            "probe_shots": self.probe_shots,
            "provenance_shots": self.provenance_shots,
            "validation_rows": self.validation_rows,
            "validation_kind": "additional_consistency_equations",
            "schedule_validation": self.schedule_validation,
            "schedule_validation_shots": self.schedule_validation_shots,
            "raw_measurement_ids": list(self.raw_measurement_ids),
        }

    def build_dem(self, **noise: object) -> DetectorErrorModel:
        """Build a PECOS DEM from the annotated trace, without Stim."""
        from pecos.qec.dem import DetectorErrorModel  # noqa: PLC0415

        return DetectorErrorModel.from_circuit(self.circuit, **noise)


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
    detector_outputs: Sequence[tuple[str, int]] | None = None,
    observable_outputs: Sequence[tuple[str, int]] | None = None,
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

    for selectors in (detector_outputs, observable_outputs):
        if selectors is not None:
            from pecos.qec.dem_spec import InferredOutput  # noqa: PLC0415

            for tag, element in selectors:
                InferredOutput(tag, element)
    if (
        detector_outputs is not None
        and observable_outputs is not None
        and not detector_outputs
        and not observable_outputs
    ):
        msg = "no outputs selected for inference"
        raise ValueError(msg)

    if hasattr(program, "compile") and not hasattr(program, "_to_program"):
        program = Guppy(program)
    to_program = getattr(program, "_to_program", None)
    if to_program is not None:
        program = to_program()
    runtime_spec = _configure_selene_runtime(_SeleneRuntimeSpec(), runtime)
    evidence = _capture_dem_evidence(
        program,
        num_qubits,
        seed,
        runtime_spec,
        settings,
        detector_outputs,
        observable_outputs,
    )
    circuit = evidence["circuit"]
    _validate_trace_hosted_operations_if_requested(
        circuit,
        require_hosted_operation_order=require_hosted_operation_order,
        max_hosted_tick_separation=max_hosted_tick_separation,
        context="infer_guppy_dem_annotations",
    )
    normalize_traced_tick_circuit(circuit, context="infer_guppy_dem_annotations")
    raw_ids = list(evidence["raw_measurement_ids"])
    detector_labels = tuple((tag, element) for tag, element in evidence["detector_outputs"])
    observable_labels = tuple((tag, element) for tag, element in evidence["observable_outputs"])
    probe_shots = evidence["probe_shots"]
    correlation_shots = evidence["correlation_shots"]
    total_shots = evidence["schedule_shots"]
    raw_binding = evidence["binding"]
    source_result_traces = tuple(dict(item) for item in evidence["named_result_traces"])
    detector_supports, observable_supports = settings.infer(
        raw_ids,
        evidence["raw_rows"],
        evidence["detector_rows"],
        evidence["observable_rows"],
    )
    detector_supports = tuple(tuple(support) for support in detector_supports)
    observable_supports = tuple(tuple(support) for support in observable_supports)

    detectors = [
        {
            "id": index,
            "meas_ids": list(support),
            "inferred_from_result_tag": tag,
            **({"result_element": element} if detector_outputs is not None else {}),
        }
        for index, (support, (tag, element)) in enumerate(zip(detector_supports, detector_labels, strict=True))
    ]
    # Coin-toss probes identify classical XOR support, not the noiseless quantum
    # value or its determinism. Even an affine offset of zero cannot establish a
    # reference. Leave it absent; the constant-one refusal above still protects
    # the separate contract of representing the program's actual output parity.
    observables = [
        {
            "id": index,
            "meas_ids": list(support),
            "inferred_from_result_tag": tag,
            "result_element": element,
        }
        for index, (support, (tag, element)) in enumerate(zip(observable_supports, observable_labels, strict=True))
    ]
    detectors_json = json.dumps(detectors, separators=(",", ":"))
    observables_json = json.dumps(observables, separators=(",", ":"))
    circuit.set_meta("detectors", detectors_json)
    circuit.set_meta("observables", observables_json)
    circuit.set_meta("num_measurements", str(len(raw_ids)))
    circuit.set_meta("guppy_dem_annotation_method", "coin_toss_affine_inference_v2")
    circuit.set_meta("guppy_raw_measurement_binding", raw_binding)

    bound_traces = []
    offset = 0
    for item in source_result_traces:
        if item.get("name") == raw_tag:
            width = len(item["values"])
            bound_traces.append({**item, "result_ids": raw_ids[offset : offset + width]})
            offset += width

    return InferredGuppyDemAnnotations(
        circuit=circuit,
        detectors_json=detectors_json,
        observables_json=observables_json,
        raw_measurement_ids=tuple(raw_ids),
        detector_supports=detector_supports,
        observable_supports=observable_supports,
        observable_labels=observable_labels,
        probe_shots=probe_shots,
        raw_binding=raw_binding,
        validation_rows=validation_rows,
        provenance_shots=correlation_shots,
        raw_tag=raw_tag,
        result_traces=tuple(bound_traces),
        source_result_traces=source_result_traces,
        detector_labels=detector_labels,
        schedule_validation="probe_compared",
        schedule_validation_shots=total_shots,
    )


__all__ = ["DemInferenceSettings", "InferredGuppyDemAnnotations", "infer_guppy_dem_annotations"]
