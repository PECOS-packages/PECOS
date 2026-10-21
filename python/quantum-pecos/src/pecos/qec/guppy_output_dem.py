"""Empirical DEM annotations inferred from Guppy parity outputs.

This module is deliberately code-agnostic.  It treats a Guppy program as the
owner of its detector/observable post-processing and learns the corresponding
affine GF(2) functions from PECOS coin-toss executions.  A QIS trace then binds
the program's raw-measurement output to stable ``MeasId`` values.

The inference is empirical, not a compiler proof.  It therefore checks the
fit using additional independent consistency equations and fails unless the raw tag
covers every physical measurement with a unique result-ID binding.
"""

# Dynamic Guppy/runtime values and local fail-loud messages are intentional at
# this experimental Python boundary.
# ruff: noqa: ANN401, EM101, EM102, TRY003

from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pecos_rslib.qec import DemInferenceSettings, _correlate_measurement_ids

if TYPE_CHECKING:
    from collections.abc import Sequence


def _bit(value: Any, *, context: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    raise ValueError(f"{context} must be bool, 0, or 1; got {value!r}")


def _named_trace_items(trace: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [item for chunk in trace for item in (chunk.get("named_result_traces") or []) if isinstance(item, Mapping)]


def _trace_shots(trace: Sequence[Mapping[str, Any]]) -> list[list[Mapping[str, Any]]]:
    shots: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for chunk in trace:
        engine_id = chunk.get("engine_trace_id")
        shot_index = chunk.get("shot_index")
        if isinstance(engine_id, bool) or not isinstance(engine_id, int):
            raise TypeError("QIS provenance trace is missing a valid engine_trace_id")
        if isinstance(shot_index, bool) or not isinstance(shot_index, int):
            raise TypeError("QIS provenance trace is missing a valid shot_index")
        shots.setdefault((engine_id, shot_index), []).append(chunk)
    from pecos._qis_trace_replay import _validate_audited_trace_stream  # noqa: PLC0415

    result = []
    for _, chunks in sorted(shots.items()):
        if any(isinstance(c.get("chunk_index"), bool) or not isinstance(c.get("chunk_index"), int) for c in chunks):
            raise TypeError("QIS probe trace has an invalid chunk_index")
        ordered = sorted(chunks, key=lambda item: item["chunk_index"])
        _validate_audited_trace_stream(ordered)
        if any(c.get("lowered_quantum_ops_complete") is not True for c in ordered):
            raise ValueError("QIS probe trace does not attest a complete lowered gate stream")
        result.append(ordered)
    return result


def _lowered_schedule(shot: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    gates = []
    for chunk in shot:
        lowered = chunk.get("lowered_quantum_ops")
        if not isinstance(lowered, list) or any(not isinstance(gate, Mapping) for gate in lowered):
            raise TypeError("QIS probe trace has invalid lowered_quantum_ops")
        gates.extend(json.dumps(gate, sort_keys=True, separators=(",", ":")) for gate in lowered)
    return tuple(gates)


def _trace_output_row(shot: Sequence[Mapping[str, Any]], tag: str) -> list[bool | int | float]:
    row = []
    if sum(1 for chunk in shot if chunk.get("named_results")) > 1:
        raise ValueError("QIS trace shot carries complete named results in more than one chunk")
    for chunk in shot:
        result = (chunk.get("named_results") or {}).get(tag)
        if result is None:
            continue
        values = result["values"]
        kind = result["type"]
        if kind == "f64":
            row.extend(struct.unpack("!d", struct.pack("!Q", bits))[0] for bits in values)
        elif kind in {"bool", "i64", "u64"}:
            row.extend(values)
        else:
            raise ValueError(f"result tag {tag!r} has unknown NamedResult type {kind!r}")
    if not row:
        raise ValueError(f"missing or empty required result tag {tag!r}")
    return row


def _raw_row(shot: Sequence[Mapping[str, Any]], tag: str) -> list[int]:
    message = f"raw tag {tag!r}: raw measurements must be emitted as bool results"
    values = _trace_output_row(shot, tag)
    try:
        row = [_bit(value, context=f"raw tag {tag!r}") for value in values]
    except ValueError as error:
        raise ValueError(message) from error
    traced = [value for item in _named_trace_items(shot) if item.get("name") == tag for value in item["values"]]
    if row != traced:
        raise ValueError(message)
    return row


def _selected_row(rows: Mapping[str, Sequence[Any]], labels: Sequence[tuple[str, int]]) -> list[int]:
    return [_bit(rows[tag][element], context=f"result tag {tag!r} element {element}") for tag, element in labels]


def _physical_row(shot: Sequence[Mapping[str, Any]], ids: Sequence[int]) -> list[int]:
    """Read a fully framed shot's physical outcomes in the requested ID order."""
    raw_results = shot[-1].get("measurement_results")
    if not isinstance(raw_results, Mapping):
        raise TypeError("QIS terminal trace lacks result-ID keyed measurement outcomes")
    try:
        outcomes = {
            int(identity): _bit(value, context="QIS measurement outcome") for identity, value in raw_results.items()
        }
    except (TypeError, ValueError) as error:
        raise ValueError("QIS terminal trace contains invalid measurement outcomes") from error
    if len(outcomes) != len(raw_results) or set(outcomes) != set(ids):
        raise ValueError("QIS probe measurement ids differ from the source trace")
    return [outcomes[identity] for identity in ids]


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

    def build_dem(self, **noise: Any) -> Any:
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
    from pecos._qis_trace_replay import source_measurement_ids_from_operation_trace  # noqa: PLC0415
    from pecos._traced_circuit import normalize_traced_tick_circuit  # noqa: PLC0415
    from pecos.tracing import (  # noqa: PLC0415
        _capture_qis_operation_traces,
        capture_qis_operation_trace,
        qis_operation_trace_to_tick_circuit,
    )

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
        raise ValueError("no outputs selected for inference")

    trace = capture_qis_operation_trace(program, num_qubits, seed=seed, runtime=runtime)
    circuit = qis_operation_trace_to_tick_circuit(
        trace,
        require_hosted_operation_order=require_hosted_operation_order,
        max_hosted_tick_separation=max_hosted_tick_separation,
    )
    normalize_traced_tick_circuit(circuit, context="infer_guppy_dem_annotations")

    _raw_row(trace, raw_tag)
    raw_ids: list[int] = []
    raw_value_count = 0
    provenance_complete = True
    for item in _named_trace_items(trace):
        if item.get("name") != raw_tag:
            continue
        values = item.get("values")
        result_ids = item.get("result_ids")
        if not isinstance(values, list):
            raise TypeError(f"raw result tag {raw_tag!r} has an invalid runtime trace value list")
        raw_value_count += len(values)
        if not isinstance(result_ids, list) or len(values) != len(result_ids):
            provenance_complete = False
            continue
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in result_ids):
            raise ValueError(f"raw result tag {raw_tag!r} contains an invalid measurement id")
        raw_ids.extend(result_ids)

    source_ids_json = circuit.get_meta("qis_source_measurement_ids")
    source_ids = json.loads(source_ids_json) if source_ids_json else []
    if raw_value_count == 0 or raw_value_count != len(source_ids):
        raise ValueError(f"result tag {raw_tag!r} must expose every physical measurement exactly once")
    if provenance_complete:
        if len(set(raw_ids)) != len(source_ids) or set(raw_ids) != set(source_ids):
            raise ValueError(
                f"result tag {raw_tag!r} must expose every physical measurement exactly once; "
                f"tag ids={raw_ids[:12]}, source ids={source_ids[:12]}",
            )
        if raw_ids != source_ids:
            raise ValueError(
                f"result tag {raw_tag!r} must expose physical measurements in certified source order; "
                f"tag ids={raw_ids[:12]}, source ids={source_ids[:12]}",
            )
    selected_tags = {raw_tag}
    selected_tags.update([detector_tag] if detector_outputs is None else [tag for tag, _ in detector_outputs])
    selected_tags.update(observable_tags if observable_outputs is None else [tag for tag, _ in observable_outputs])
    baseline_rows = {tag: _trace_output_row(trace, tag) for tag in selected_tags}

    def select_labels(
        selectors: Sequence[tuple[str, int]] | None,
        default_tags: Sequence[str],
    ) -> tuple[tuple[str, int], ...]:
        labels = (
            tuple(selectors)
            if selectors is not None
            else tuple((tag, element) for tag in default_tags for element in range(len(baseline_rows[tag])))
        )
        for tag, element in labels:
            if element < 0 or element >= len(baseline_rows[tag]):
                raise ValueError(f"inferred output {tag!r} element {element} is out of range")
        return labels

    detector_labels = select_labels(detector_outputs, [detector_tag])
    observable_labels = select_labels(observable_outputs, observable_tags)
    _selected_row(baseline_rows, detector_labels)
    _selected_row(baseline_rows, observable_labels)
    probe_shots = settings.resolved_probe_shots(raw_value_count)
    correlation_shots = provenance_shots if not provenance_complete and require_raw_provenance else 0
    total_shots = probe_shots + correlation_shots
    probe_trace = _capture_qis_operation_traces(
        program,
        num_qubits,
        shots=total_shots,
        seed=(seed + 1) % (1 << 64),
        runtime=runtime,
    )
    shots = _trace_shots(probe_trace)
    if len(shots) != total_shots:
        raise ValueError(f"QIS probe trace contains {len(shots)} shots; expected {total_shots}")
    expected_schedule = _lowered_schedule(trace)
    for shot in shots:
        if (
            _lowered_schedule(shot) != expected_schedule
            or source_measurement_ids_from_operation_trace(shot) != source_ids
        ):
            raise ValueError(
                "quantum operation schedule changed during inference/provenance probing; "
                "a single static DEM cannot represent this program",
            )
    raw_probe_rows = [_raw_row(shot, raw_tag) for shot in shots]
    if provenance_complete:
        raw_binding = "runtime_result_ids"
    elif require_raw_provenance:
        provenance = shots[:correlation_shots]
        raw_ids = _correlate_measurement_ids(
            raw_probe_rows[:correlation_shots],
            [_physical_row(shot, source_ids) for shot in provenance],
            source_ids,
        )
        raw_binding = "probe_correlated_result_ids"
    else:
        raw_ids = list(source_ids)
        raw_binding = "assumed_canonical_result_order"

    raw_rows = []
    detector_rows = []
    observable_rows = []
    # Fit and validate the exact shots whose schedules and physical bindings
    # were checked. Correlation rows are separate from the fitted rows.
    for shot, raw_row in zip(shots[correlation_shots:], raw_probe_rows[correlation_shots:], strict=True):
        rows = {tag: _trace_output_row(shot, tag) for tag in selected_tags}
        for tag, row in rows.items():
            if len(row) != len(baseline_rows[tag]):
                raise ValueError(f"result {tag!r} has inconsistent shot widths")
        if require_raw_provenance and raw_row != _physical_row(shot, raw_ids):
            raise ValueError("raw measurement binding changed across inference probes")
        raw_rows.append(raw_row)
        detector_rows.append(_selected_row(rows, detector_labels))
        observable_rows.append(_selected_row(rows, observable_labels))

    # Shared Rust inference validates both output families with one elimination.
    detector_supports, observable_supports = settings.infer(
        raw_ids,
        raw_rows,
        detector_rows,
        observable_rows,
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
    for item in _named_trace_items(trace):
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
        source_result_traces=tuple(_named_trace_items(trace)),
        detector_labels=detector_labels,
        schedule_validation="probe_compared",
        schedule_validation_shots=total_shots,
    )


__all__ = ["DemInferenceSettings", "InferredGuppyDemAnnotations", "infer_guppy_dem_annotations"]
