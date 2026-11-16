"""Empirical DEM annotations inferred from Guppy parity outputs.

This module is deliberately code-agnostic.  It treats a Guppy program as the
owner of its detector/observable post-processing and learns the corresponding
affine GF(2) functions from PECOS coin-toss executions.  A QIS trace then binds
the program's raw-measurement output to stable ``MeasId`` values.

The inference is empirical, not a compiler proof.  It therefore validates the
learned functions on additional independent rows and fails unless the raw tag
covers every physical measurement with a unique result-ID binding.
"""

# Dynamic Guppy/runtime values and local fail-loud messages are intentional at
# this experimental Python boundary.
# ruff: noqa: ANN401, EM101, EM102, TRY003

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pecos_rslib.qec import DemInferenceSettings, _correlate_measurement_ids


def _bit(value: Any, *, context: str) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) and value in (0, 1):
        return value
    raise ValueError(f"{context} must be bool, 0, or 1; got {value!r}")


def _rows(values: Sequence[Any], *, tag: str) -> list[list[int]]:
    rows: list[list[int]] = []
    width: int | None = None
    for shot, value in enumerate(values):
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            row = [_bit(item, context=f"result {tag!r}, shot {shot}") for item in value]
        else:
            row = [_bit(value, context=f"result {tag!r}, shot {shot}")]
        if not row:
            raise ValueError(f"result {tag!r}, shot {shot} is empty")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError(f"result {tag!r} has inconsistent shot widths: {width} and {len(row)}")
        rows.append(row)
    if not rows:
        raise ValueError(f"result {tag!r} contains no shots")
    return rows


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
    return [sorted(chunks, key=lambda item: int(item.get("chunk_index", -1))) for _, chunks in sorted(shots.items())]


def _lowered_schedule(shot: Sequence[Mapping[str, Any]]) -> tuple[str, ...]:
    return tuple(
        json.dumps(gate, sort_keys=True, separators=(",", ":"))
        for chunk in shot
        for gate in (chunk.get("lowered_quantum_ops") or [])
        if isinstance(gate, Mapping)
    )


def _correlate_raw_measurement_ids(
    trace: Sequence[Mapping[str, Any]],
    *,
    raw_tag: str,
    source_ids: Sequence[int],
) -> list[int]:
    """Recover aggregate raw-output identity by independent probe signatures."""
    shots = _trace_shots(trace)
    if len(shots) < 2:
        raise ValueError("measurement provenance correlation needs at least two trace shots")
    expected_schedule = _lowered_schedule(shots[0])
    raw_rows: list[list[int]] = []
    physical_rows: list[list[int]] = []
    for shot_number, shot in enumerate(shots):
        if _lowered_schedule(shot) != expected_schedule:
            raise ValueError(
                f"quantum operation schedule changed during provenance probing at shot {shot_number}; "
                "a single static DEM cannot represent this program",
            )
        raw_row = [
            _bit(value, context=f"raw result tag {raw_tag!r}, provenance shot {shot_number}")
            for item in _named_trace_items(shot)
            if item.get("name") == raw_tag
            for value in (item.get("values") or [])
        ]
        terminal = [chunk for chunk in shot if chunk.get("stage") == "trace_complete"]
        if len(terminal) != 1:
            raise ValueError(f"provenance shot {shot_number} has {len(terminal)} terminal trace chunks")
        raw_results = terminal[0].get("measurement_results")
        if not isinstance(raw_results, Mapping):
            raise TypeError("QIS terminal trace lacks result-ID keyed measurement outcomes")
        try:
            outcomes = {
                int(result_id): _bit(value, context="QIS measurement outcome")
                for result_id, value in raw_results.items()
            }
        except (TypeError, ValueError) as error:
            raise ValueError("QIS terminal trace contains invalid measurement outcomes") from error
        if set(outcomes) != set(source_ids):
            raise ValueError(
                f"provenance shot {shot_number} measurement ids differ from the source trace: "
                f"shot={sorted(outcomes)[:12]}, source={list(source_ids)[:12]}",
            )
        if len(raw_row) != len(source_ids):
            raise ValueError(
                f"result tag {raw_tag!r} emits {len(raw_row)} values during provenance probing, "
                f"but the QIS trace has {len(source_ids)} measurements",
            )
        raw_rows.append(raw_row)
        physical_rows.append([outcomes[result_id] for result_id in source_ids])

    return _correlate_measurement_ids(raw_rows, physical_rows, source_ids)


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
    result_traces: tuple[Mapping[str, Any], ...] = ()

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
) -> InferredGuppyDemAnnotations:
    """Infer parity annotations from an untouched Guppy program.

    The program must emit every physical measurement, in QIS measurement
    through one or more ``result(raw_tag, ...)`` calls. Detector and
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
    import pecos_rslib  # noqa: PLC0415

    import pecos  # noqa: PLC0415
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

    trace = capture_qis_operation_trace(program, num_qubits, seed=seed, runtime=runtime)
    circuit = qis_operation_trace_to_tick_circuit(
        trace,
        require_hosted_operation_order=require_hosted_operation_order,
        max_hosted_tick_separation=max_hosted_tick_separation,
    )
    normalize_traced_tick_circuit(circuit, context="infer_guppy_dem_annotations")

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
    # Schedule comparison is required even when runtime IDs already bind raw
    # outputs. Compare to the circuit-producing shot, not only between probes.
    provenance_trace = _capture_qis_operation_traces(
        program,
        num_qubits,
        shots=provenance_shots,
        seed=(seed + 1) % (1 << 64),
        runtime=runtime,
    )
    expected_schedule = _lowered_schedule(trace)
    for shot in _trace_shots(provenance_trace):
        if _lowered_schedule(shot) != expected_schedule:
            raise ValueError(
                "quantum operation schedule changed during provenance probing; "
                "a single static DEM cannot represent this program",
            )
    if not provenance_complete:
        if require_raw_provenance:
            raw_ids = _correlate_raw_measurement_ids(
                provenance_trace,
                raw_tag=raw_tag,
                source_ids=source_ids,
            )
            raw_binding = "probe_correlated_result_ids"
        else:
            if raw_value_count != len(source_ids):
                raise ValueError(
                    f"result tag {raw_tag!r} emits {raw_value_count} traced values, but the QIS trace has "
                    f"{len(source_ids)} measurements",
                )
            raw_ids = list(source_ids)
            raw_binding = "assumed_canonical_result_order"
    else:
        raw_binding = "runtime_result_ids"

    if probe_shots is None:
        probe_shots = settings.resolved_probe_shots(raw_value_count)

    results = (
        pecos.sim(program)
        .classical(pecos.selene_engine(runtime))
        .quantum(pecos_rslib.coin_toss())
        .qubits(num_qubits)
        .seed(seed)
        .run(probe_shots)
        .to_dict()
    )
    missing = [tag for tag in (raw_tag, detector_tag, *observable_tags) if tag not in results]
    if missing:
        raise ValueError(f"Guppy results are missing required tag(s): {missing}")

    raw_rows = _rows(results[raw_tag], tag=raw_tag)
    if len(raw_rows[0]) != len(raw_ids):
        raise ValueError(
            f"result tag {raw_tag!r} emits {len(raw_rows[0])} values per shot, "
            f"but the QIS trace has {len(raw_ids)} measurements",
        )
    detector_rows = _rows(results[detector_tag], tag=detector_tag)
    observable_parts = [(tag, _rows(results[tag], tag=tag)) for tag in observable_tags]
    observable_rows = [[value for _, rows in observable_parts for value in rows[shot]] for shot in range(probe_shots)]
    observable_labels = tuple((tag, element) for tag, rows in observable_parts for element in range(len(rows[0])))

    # Shared Rust inference validates both output families with one elimination.
    detector_supports, observable_supports = settings._infer(  # noqa: SLF001
        raw_ids,
        raw_rows,
        detector_rows,
        observable_rows,
    )
    detector_supports = tuple(tuple(support) for support in detector_supports)
    observable_supports = tuple(tuple(support) for support in observable_supports)

    detectors = [
        {"id": index, "meas_ids": list(support), "inferred_from_result_tag": detector_tag}
        for index, support in enumerate(detector_supports)
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
        provenance_shots=provenance_shots,
        raw_tag=raw_tag,
        result_traces=tuple(bound_traces),
        schedule_validation="probe_compared",
    )


__all__ = ["DemInferenceSettings", "InferredGuppyDemAnnotations", "infer_guppy_dem_annotations"]
