# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Pre-decision error model for raw-resource surface-code T teleportation.

Only the first decision in a shot is covered, including in programs with several
T/TDG decisions: its window precedes every correction branch. Later windows
include executed correction rounds and require a different model. Inputs must
be canonical, with no pending frame. Decoding is external: XOR the raw decision
with the decoder's predicted observable flip.

The DEM uses a raw native trace of a straight-line Z-seed stand-in, preserving
noise on every native gate. It omits the sign-fix Z Paulis: these change no fault
propagation, but their p1 noise is missing (about 2.4% of single-qubit locations
at d = 3). The resource seed's T and input seed's single-qubit gates are logical,
not detectable: their X parts flip the observable undetectably, so DEM-sampled
observable rates miss about 2 p1 of floor; decoding is unaffected. Idle noise,
hook resources and their acceptance conditioning are excluded.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from importlib import import_module
from typing import TYPE_CHECKING

from pecos_rslib.qec import SampleBatch

from pecos._traced_circuit import measurement_ids_in_execution_order
from pecos.qec.dem import DetectorErrorModel
from pecos.qec.surface._detection_events import extract_detection_events_and_observables
from pecos.qec.surface.gadgets import default_allocation
from pecos.qec.surface.logical_circuit import LogicalCircuitBuilder
from pecos.tracing import _trace_program_to_tick_circuit_with_result_traces

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pecos_rslib.quantum import TickCircuit

    from pecos.decoders import DecoderSpec
    from pecos.qec.surface.patch import SurfacePatch


_TRACE_SEED = 712  # Any value works: the stand-in is straight-line, with no measurement-dependent branch.
_SYNDROME_TAG = re.compile(r"(data|anc):s([xz])(\d+):meas:(\d+)")


def _decision_stand_in_source(patch: SurfacePatch, rounds_before: int) -> str:
    # Guppy generation imports qec.surface; load it only when a DEM is needed.
    renderer = import_module("pecos.guppy_gen.surface_teleportation")
    lines = ["", "@guppy", "def decision_stand_in() -> None:"]
    for register, scope in (("data", "data"), ("resource", "anc")):
        lines.extend(
            [
                f"    {register} = prep_injection_z_seed()",
                f"    syn = syndrome_extraction_{scope}({register})",
                '    output("injection_synx", syn.synx)',
                '    output("injection_synz", syn.synz)',
            ],
        )
    for _ in range(rounds_before):
        lines.extend(
            [
                "    syn_data = syndrome_extraction_data(data)",
                "    syn_resource = syndrome_extraction_anc(resource)",
                '    output("data_synx", syn_data.synx)',
                '    output("data_synz", syn_data.synz)',
                '    output("resource_synx", syn_resource.synx)',
                '    output("resource_synz", syn_resource.synz)',
            ],
        )
    lines.extend(["    consume_t_resource(data, resource)", '    output("final_data", measure_z_basis(data))', ""])
    return renderer.render_surface_t_teleportation_module(patch) + "\n".join(lines)


def _decision_measurement_map(
    patch: SurfacePatch,
    rounds_before: int,
    circuit: TickCircuit,
    result_traces: Sequence[dict],
) -> dict[int, int]:
    rounds = []
    readouts = []
    for result in result_traces:
        match = _SYNDROME_TAG.fullmatch(result["name"])
        if match:
            scope, family, index, ordinal = match.groups()
            index, ordinal = int(index), int(ordinal)
            if ordinal == 0:
                rounds.append((scope, {"x": {}, "z": {}}))
            if not rounds or rounds[-1][0] != scope:
                msg = "Unexpected syndrome scope or missing :meas:0 round boundary"
                raise ValueError(msg)
            checks = rounds[-1][1]
            if ordinal != sum(map(len, checks.values())) or index in checks[family] or len(result["result_ids"]) != 1:
                msg = "Syndrome round must contain each check exactly once with one result id"
                raise ValueError(msg)
            checks[family][index] = result["result_ids"][0]
        elif result["name"] == "teleportation_resource_readout":
            readouts.append(result["result_ids"])
    if [scope for scope, _ in rounds] != ["data", "anc"] * (rounds_before + 1):
        msg = "Unexpected syndrome scope sequence in decision trace"
        raise ValueError(msg)
    for _, checks in rounds:
        for family in ("x", "z"):
            if set(checks[family]) != set(range(len(getattr(patch.geometry, f"{family}_stabilizers")))):
                msg = f"Syndrome round must contain every {family} check exactly once"
                raise ValueError(msg)
    if len(readouts) != 1 or len(readouts[0]) != patch.geometry.num_data:
        msg = "Decision trace must contain one complete resource readout"
        raise ValueError(msg)
    keys = json.loads(circuit.get_meta("measurement_keys"))
    mapping = {}
    for label, family, index, _segment, occurrence, ordinal in keys["stabilizer"]:
        scope_index = 0 if label == "D" else 1
        # Keep X1: the program's feed-forward makes X2 sensitive to X1 errors.
        # Without the fix, X2 XOR X1 has exactly that same sensitivity.
        mapping[ordinal] = rounds[2 * occurrence + scope_index][1][family.lower()][index]
    data_indices = {qubit: index for index, qubit in enumerate(default_allocation(patch).data_qubits)}
    for label, qubit, ordinal in keys["data"]:
        if label == "A":
            mapping[ordinal] = readouts[0][data_indices[qubit]]
    if len(set(mapping.values())) != len(mapping):
        msg = "Decision trace result ids must be unique"
        raise ValueError(msg)
    return mapping


@dataclass(frozen=True)
class DecodedDecisionRun:
    """Shot records selected by decoded and raw first-correction decisions."""

    results: dict[str, list]
    raw_results: dict[str, list]
    raw: list[bool]
    decoded: list[bool]


@dataclass(frozen=True)
class TeleportationDecisionModel:
    """Decision-time detectors and one resource logical-Z flip observable.

    ``circuit`` is the builder stand-in defining detector ids and program record
    transcription. ``traced_circuit`` is the lazily cached native DEM source.
    """

    patch: SurfacePatch
    rounds_before: int
    circuit: TickCircuit

    @cached_property
    def traced_circuit(self) -> TickCircuit:
        """Raw native trace of a Clifford stand-in with this model's detectors and observable; the DEM source.

        The straight-line Z-seed program omits sign-fix Z Paulis. This changes
        no fault propagation and avoids sampling a random correction path, but
        under-counts their p1 noise. Native Clifford chains remain unsimplified.
        """
        loader = import_module("pecos.guppy_gen._module_loader")
        module = loader.load_cached_guppy_source(
            "teleportation_decision",
            _decision_stand_in_source(self.patch, self.rounds_before),
        )
        traced, results = _trace_program_to_tick_circuit_with_result_traces(
            module["decision_stand_in"],
            self.patch.geometry.num_data + self.patch.geometry.num_qubits,
            seed=_TRACE_SEED,
        )
        mapping = _decision_measurement_map(self.patch, self.rounds_before, self.circuit, results)
        emitted = measurement_ids_in_execution_order(traced)
        positions = {mid: position for position, mid in enumerate(emitted)}
        if len(positions) != len(emitted) or not set(mapping.values()) <= positions.keys():
            msg = "Decision result ids must name unique emitted measurements"
            raise ValueError(msg)
        for name in ("detectors", "observables"):
            entries = json.loads(self.circuit.get_meta(name))
            for entry in entries:
                entry["meas_ids"] = [mapping[mid] for mid in entry["meas_ids"]]
                entry["records"] = [positions[mid] - len(emitted) for mid in entry["meas_ids"]]
            traced.set_meta(name, json.dumps(entries))
        traced.set_meta("num_measurements", str(len(emitted)))
        return traced

    def build_dem(
        self,
        *,
        p1: float = 0.001,
        p2: float = 0.001,
        p_meas: float = 0.001,
        p_prep: float = 0.0,
    ) -> DetectorErrorModel:
        """Build the truncated DEM with native-gate noise rates."""
        return DetectorErrorModel.from_circuit(self.traced_circuit, p1=p1, p2=p2, p_meas=p_meas, p_prep=p_prep)

    def detection_events(self, results: Mapping[str, Sequence]) -> tuple[list[list[int]], list[bool]]:
        """Return fired detector ids and raw decision parity for each shot.

        Accept the output of ``sim(program).run(shots).to_dict()`` for a raw
        resource program with this patch and pre-decision round count. Repeated
        tags may contain later records; only the first decision's prefix is used.
        """
        rows = self._transcribe(results)
        events, _ = extract_detection_events_and_observables(self.circuit, rows)
        observable = json.loads(self.circuit.get_meta("observables"))[0]
        raw = [bool(sum(row[mid] for mid in observable["meas_ids"]) % 2) for row in rows]
        return events, raw

    def decode_decisions(
        self,
        results: Mapping[str, Sequence],
        decoder: DecoderSpec,
        *,
        dem: DetectorErrorModel,
    ) -> tuple[list[bool], list[bool]]:
        """Return raw decisions and decisions XORed with predicted observable flips.

        Rows use this window's detector count so SampleBatch.decode validates
        compatibility with the supplied DEM, including for mismatched windows.
        """
        events, raw = self.detection_events(results)
        width = len(json.loads(self.circuit.get_meta("detectors")))
        rows = [[0] * width for _ in events]
        for row, fired in zip(rows, events, strict=True):
            for detector in fired:
                row[detector] = 1
        predictions = (
            SampleBatch(rows, [0] * len(raw), num_observables=1)
            .decode(
                str(dem),
                decoder,
                predictions=True,
            )
            .predictions
        )
        decoded = [bit ^ bool(prediction & 1) for bit, prediction in zip(raw, predictions, strict=True)]
        return raw, decoded

    def run_decoded(
        self,
        run: Callable[[bool], Mapping[str, Sequence]],
        decoder: DecoderSpec,
        *,
        dem: DetectorErrorModel,
    ) -> DecodedDecisionRun:
        """Run both forced variants at full shot count (2x cost), decoding once.

        Results and raw_results are exact: programs share pre-branch source,
        each shot's seed depends only on (run seed, shot index), and QIS
        operations run in program order. Window equality checks the pre-branch
        half at runtime. The caller must bind identical seeds and noise.
        """
        variants = tuple(run(forced) for forced in (False, True))
        if variants[0].keys() != variants[1].keys():
            msg = "Forced variants must have the same tags"
            raise ValueError(msg)
        counts = {len(values) for results in variants for values in results.values()}
        if len(counts) != 1:
            msg = "Every tag in both forced variants must have the same shot count"
            raise ValueError(msg)
        for forced, results in zip((False, True), variants, strict=True):
            if "t_correction" not in results:
                msg = "Missing t_correction; the first decision must be forced"
                raise ValueError(msg)
            for records in results["t_correction"]:
                first = records[0] if isinstance(records, Sequence) and records else records
                if first != forced:
                    msg = f"First t_correction must equal forced decision {forced}"
                    raise ValueError(msg)
        if self.detection_events(variants[0]) != self.detection_events(variants[1]):
            msg = "Pre-decision windows differ: runs did not share a seed or differ before the decision"
            raise ValueError(msg)
        raw, decoded = self.decode_decisions(variants[0], decoder, dem=dem)
        return DecodedDecisionRun(
            results={tag: [variants[bit][tag][k] for k, bit in enumerate(decoded)] for tag in variants[0]},
            raw_results={tag: [variants[bit][tag][k] for k, bit in enumerate(raw)] for tag in variants[0]},
            raw=raw,
            decoded=decoded,
        )

    def _transcribe(self, results: Mapping[str, Sequence]) -> list[list[int]]:
        if "hook_accepted" in results:
            msg = "hook_accepted results require a hook-resource model; only raw resources are supported"
            raise ValueError(msg)
        widths = {
            "x": len(self.patch.geometry.x_stabilizers),
            "z": len(self.patch.geometry.z_stabilizers),
        }
        required = {"teleportation_resource_readout": self.patch.geometry.num_data}
        for family, width in widths.items():
            required[f"injection_syn{family}"] = 2 * width
            if self.rounds_before:
                for role in ("data", "resource"):
                    required[f"{role}_syn{family}"] = self.rounds_before * width
        shots = None
        for tag, width in required.items():
            if tag not in results:
                msg = f"Missing {tag}; expected at least {width} records per shot"
                raise ValueError(msg)
            values = results[tag]
            if shots is None:
                shots = len(values)
            if len(values) != shots:
                msg = f"{tag} has {len(values)} shots; expected {shots}"
                raise ValueError(msg)
            for shot, bits in enumerate(values):
                if not isinstance(bits, Sequence) or len(bits) < width:
                    msg = f"{tag} shot {shot} must contain at least {width} records"
                    raise ValueError(msg)

        keys = json.loads(self.circuit.get_meta("measurement_keys"))
        num_measurements = int(self.circuit.get_meta("num_measurements"))
        data_indices = {qubit: index for index, qubit in enumerate(default_allocation(self.patch).data_qubits)}
        rows = []
        for shot in range(shots):
            row = [0] * num_measurements
            for label, check_family, index, _segment, occurrence, ordinal in keys["stabilizer"]:
                family = check_family.lower()
                if occurrence == 0:
                    # The physical sign fix sets the initial X checks to +1.
                    # A faulty first measurement survives in the next record.
                    if family == "x":
                        continue
                    tag = f"injection_syn{family}"
                    block = 0 if label == "D" else 1
                else:
                    tag = f"{'data' if label == 'D' else 'resource'}_syn{family}"
                    block = occurrence - 1
                row[ordinal] = int(bool(results[tag][shot][block * widths[family] + index]))
            for label, qubit, ordinal in keys["data"]:
                if label == "A":
                    row[ordinal] = int(bool(results["teleportation_resource_readout"][shot][data_indices[qubit]]))
            rows.append(row)
        return rows


def t_teleportation_decision_model(patch: SurfacePatch, rounds_before: int = 0) -> TeleportationDecisionModel:
    """Build a raw-resource T/TDG first-decision window from a Clifford stand-in.

    The patch must be odd, square and rotated, with distance at least three.
    ``rounds_before`` counts full rounds after the two injection rounds.
    """
    if isinstance(rounds_before, bool) or not isinstance(rounds_before, int) or rounds_before < 0:
        msg = "rounds_before must be a nonnegative integer"
        raise ValueError(msg)
    if not patch.rotated or patch.dx != patch.dz or patch.dx < 3 or patch.dx % 2 == 0:
        msg = "Surface T teleportation requires an odd square rotated patch of distance >= 3"
        raise ValueError(msg)
    builder = LogicalCircuitBuilder()
    builder.add_patch(patch, "D")
    builder.add_patch(patch, "A", qubit_offset=patch.geometry.num_qubits)
    builder.add_memory(["D", "A"], rounds=rounds_before + 1, basis="Z")
    builder.add_transversal_cx("D", "A")
    builder.add_memory(["D", "A"], rounds=0, basis="Z")
    circuit = builder.to_tick_circuit()
    keys = json.loads(circuit.get_meta("measurement_keys"))
    data_readout = {ordinal for label, _qubit, ordinal in keys["data"] if label == "D"}
    detectors = [
        det for det in json.loads(circuit.get_meta("detectors")) if not data_readout.intersection(det["meas_ids"])
    ]
    observables = [
        obs for obs in json.loads(circuit.get_meta("observables")) if not data_readout.intersection(obs["meas_ids"])
    ]
    if len(observables) != 1 or observables[0]["reference"] != 0:
        msg = "The decision stand-in must have exactly one resource observable with reference 0"
        raise ValueError(msg)
    for index, detector in enumerate(detectors):
        detector["id"] = index
    observables[0]["id"] = 0
    circuit.set_meta("detectors", json.dumps(detectors))
    circuit.set_meta("observables", json.dumps(observables))
    return TeleportationDecisionModel(patch, rounds_before, circuit)
