# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""First-decision transcription, truncation and matched-channel statistics."""

import inspect
import json
import math
import statistics
from collections import defaultdict
from itertools import combinations

import numpy as np
import pecos_rslib as prs
import pytest
from pecos import sim, stab_vec
from pecos.decoders import tesseract
from pecos.guppy_gen import make_surface_t_teleportation
from pecos.guppy_gen._module_loader import _get_temp_dir, load_guppy_source
from pecos.guppy_gen.surface_teleportation import (
    load_surface_t_teleportation_module,
    render_surface_t_teleportation_module,
)
from pecos.qec.dem import DetectorErrorModel
from pecos.qec.surface import LogicalCircuitBuilder, SurfacePatch, t_teleportation_decision_model
from pecos.qec.surface.gadgets import default_allocation
from pecos_rslib.qec import DemSampler


def _run(program, patch, shots=8, noise=None, seed=712):
    engine = prs.qis_engine().selene_runtime().interface(prs.qis_helios_interface())
    runner = (
        sim(program)
        .classical(engine)
        .quantum(stab_vec())
        .qubits(patch.geometry.num_data + patch.geometry.num_qubits)
        .seed(seed)
    )
    if noise is not None:
        runner = runner.noise(noise)
    return runner.run(shots).to_dict()


def _stand_in(patch, rounds_before):
    builder = LogicalCircuitBuilder()
    builder.add_patch(patch, "D")
    builder.add_patch(patch, "A", qubit_offset=patch.geometry.num_qubits)
    builder.add_memory(["D", "A"], rounds=rounds_before + 1, basis="Z")
    builder.add_transversal_cx("D", "A")
    builder.add_memory(["D", "A"], rounds=0, basis="Z")
    return builder.to_tick_circuit()


def _zero_results(patch, rounds_before):
    results = {"teleportation_resource_readout": [[0] * patch.geometry.num_data]}
    for family, checks in (("x", patch.geometry.x_stabilizers), ("z", patch.geometry.z_stabilizers)):
        results[f"injection_syn{family}"] = [[0] * (2 * len(checks))]
        if rounds_before:
            for role in ("data", "resource"):
                results[f"{role}_syn{family}"] = [[0] * (rounds_before * len(checks))]
    return results


@pytest.mark.parametrize("dagger", [False, True])
@pytest.mark.parametrize(
    ("distance", "state", "rounds_before"),
    [(3, state, rounds) for state in ("X", "-X", "Y", "-Y", "Z", "-Z") for rounds in (0, 1, 2)]
    + [(5, state, rounds) for state in ("X", "-Z", "Y") for rounds in (0, 2)],
)
def test_noiseless_program_fires_no_detector(distance, state, rounds_before, dagger):
    patch = SurfacePatch.create(distance=distance)
    model = t_teleportation_decision_model(patch, rounds_before)
    program = make_surface_t_teleportation(patch, rounds_before, input_state=state, dagger=dagger)
    results = _run(program, patch)
    events, raw = model.detection_events(results)
    assert events == [[] for _ in range(8)]
    assert raw == [bool(bit) for bit in results["t_correction"]]


def test_noiseless_program_fires_no_detector_round_trip():
    patch = SurfacePatch.create(distance=3)
    source = render_surface_t_teleportation_module(patch) + """
@guppy
def decision_round_trip() -> None:
    data = prepare_injected_y()
    resource = prepare_injected_t()
    apply_t_teleportation(data, resource, False)
    resource = prepare_injected_tdg()
    apply_t_teleportation(data, resource, True)
    syn = syndrome_extraction_fold_szdg_data(data)
    output("readout_synx", syn.synx)
    output("readout_synz", syn.synz)
    output("final_data", measure_x_basis(data))
"""
    name = "test_decision_round_trip"
    module = load_guppy_source(source, _get_temp_dir() / f"{name}.py", name)
    results = _run(module["decision_round_trip"], patch)
    events, raw = t_teleportation_decision_model(patch).detection_events(results)
    assert events == [[] for _ in range(8)]
    assert raw == [bool(bits[0]) for bits in results["t_correction"]]


def _measurement_owners(circuit, patch):
    # Independent oracle: physical gates and allocations, not measurement_keys.
    allocation = default_allocation(patch)
    owners = {}
    for label, offset in (("D", 0), ("A", patch.geometry.num_qubits)):
        for family, qubits in (
            ("data", allocation.data_qubits),
            ("x", allocation.x_ancilla_qubits),
            ("z", allocation.z_ancilla_qubits),
        ):
            for index, qubit in enumerate(qubits):
                owners[qubit + offset] = (label, family, index)
    measurements = {}
    for tick in range(circuit.num_ticks()):
        for batch in circuit.get_tick(tick).gate_batches():
            if batch.meas_ids:
                for mid, qubit in zip(batch.meas_ids, batch.qubits, strict=True):
                    measurements[mid] = owners[qubit]
    assert sorted(measurements) == list(range(int(circuit.get_meta("num_measurements"))))
    occurrences = defaultdict(int)
    by_owner = {}
    for mid, owner in sorted(measurements.items()):
        by_owner[*owner, occurrences[owner]] = mid
        occurrences[owner] += 1
    return by_owner


def test_each_record_fires_exactly_its_owners_detectors():
    patch = SurfacePatch.create(distance=3)
    model = t_teleportation_decision_model(patch, 1)
    owners = _measurement_owners(model.circuit, patch)
    detectors = json.loads(model.circuit.get_meta("detectors"))
    results = _zero_results(patch, 1)
    checked = 0
    for (label, family, index, occurrence), mid in owners.items():
        if family == "data":
            if label == "D":
                continue
            tag, position = "teleportation_resource_readout", index
        elif occurrence == 0:
            tag = f"injection_syn{family}"
            width = len(getattr(patch.geometry, f"{family}_stabilizers"))
            position = index + (width if label == "A" else 0)
        else:
            tag = f"{'data' if label == 'D' else 'resource'}_syn{family}"
            position = index
        results[tag][0][position] = 1
        events, raw = model.detection_events(results)
        expected = {det["id"] for det in detectors if mid in det["meas_ids"]}
        if family == "x" and occurrence == 0:
            expected = set()
        assert events == [sorted(expected)], (tag, position, label, family, occurrence)
        assert raw == [family == "data" and index in patch.geometry.logical_z.data_qubits]
        results[tag][0][position] = 0
        checked += 1
    assert checked == 41


@pytest.mark.parametrize(
    ("distance", "rounds_before", "before", "after"),
    [(3, 0, 16, 12), (3, 1, 32, 28), (5, 0, 48, 36), (5, 1, 96, 84)],
)
def test_truncation_drops_exactly_the_data_readout_detectors(distance, rounds_before, before, after):
    patch = SurfacePatch.create(distance=distance)
    full = _stand_in(patch, rounds_before)
    circuit = t_teleportation_decision_model(patch, rounds_before).circuit
    owners = _measurement_owners(full, patch)
    data = {mid for (label, family, _index, _round), mid in owners.items() if label == "D" and family == "data"}
    resource = {mid for (label, family, _index, _round), mid in owners.items() if label == "A" and family == "data"}
    full_dets = json.loads(full.get_meta("detectors"))
    kept = json.loads(circuit.get_meta("detectors"))
    assert len(full_dets) == before
    assert len(kept) == after
    expected = [det for det in full_dets if data.isdisjoint(det["meas_ids"])]
    assert kept == [dict(det, id=index) for index, det in enumerate(expected)]
    assert all(data.isdisjoint(det["meas_ids"]) for det in kept)
    assert all(data.intersection(det["meas_ids"]) for det in full_dets if det not in expected)
    observables = json.loads(circuit.get_meta("observables"))
    assert len(observables) == 1
    assert observables[0]["id"] == 0
    assert observables[0]["reference"] == 0
    assert set(observables[0]["meas_ids"]) <= resource


def _posterior(count, total):
    alpha, beta = count + 0.5, total - count + 0.5
    return alpha / (alpha + beta), alpha * beta / ((alpha + beta) ** 2 * (alpha + beta + 1))


def _rate_z(left, right):
    a, v = _posterior(sum(left), len(left))
    b, w = _posterior(sum(right), len(right))
    return abs(a - b) / math.sqrt(v + w)


def test_invalid_inputs():
    for kwargs in ({"distance": 1}, {"distance": 2}, {"dx": 3, "dz": 5}, {"distance": 3, "rotated": False}):
        with pytest.raises(ValueError, match="odd square rotated"):
            t_teleportation_decision_model(SurfacePatch.create(**kwargs))
    patch = SurfacePatch.create(distance=3)
    for rounds in (-1, 1.0, True):
        with pytest.raises(ValueError, match="rounds_before"):
            t_teleportation_decision_model(patch, rounds)
    model = t_teleportation_decision_model(patch, 1)
    with pytest.raises(ValueError, match="hook_accepted"):
        model.detection_events({**_zero_results(patch, 1), "hook_accepted": [True]})
    for tag in _zero_results(patch, 1):
        short = _zero_results(patch, 1)
        short[tag][0].pop()
        with pytest.raises(ValueError, match=tag):
            model.detection_events(short)
        missing = _zero_results(patch, 1)
        del missing[tag]
        with pytest.raises(ValueError, match=tag):
            model.detection_events(missing)
        inconsistent = _zero_results(patch, 1)
        inconsistent[tag].append(inconsistent[tag][0][:])
        with pytest.raises(ValueError, match="shots"):
            model.detection_events(inconsistent)
    with pytest.raises(ValueError, match="teleportation_resource_readout"):
        model.detection_events({**_zero_results(patch, 1), "teleportation_resource_readout": [0]})
    assert model.detection_events({tag: [] for tag in _zero_results(patch, 1)}) == ([], [])


def test_raw_decision_is_parity_even_with_changed_reference():
    model = t_teleportation_decision_model(SurfacePatch.create(distance=3))
    results = _zero_results(model.patch, 0)
    observables = json.loads(model.circuit.get_meta("observables"))
    observables[0]["reference"] = 1
    model.circuit.set_meta("observables", json.dumps(observables))
    assert model.detection_events(results)[1] == [False]
    results["teleportation_resource_readout"][0][model.patch.geometry.logical_z.data_qubits[0]] = 1
    assert model.detection_events(results)[1] == [True]


def test_nonzero_stand_in_reference_is_rejected(monkeypatch):
    patch = SurfacePatch.create(distance=3)
    circuit = _stand_in(patch, 0)
    resource = {
        mid
        for (label, family, _index, _round), mid in _measurement_owners(circuit, patch).items()
        if label == "A" and family == "data"
    }
    observables = json.loads(circuit.get_meta("observables"))
    (resource_observable,) = [obs for obs in observables if set(obs["meas_ids"]) <= resource]
    resource_observable["reference"] = 1
    circuit.set_meta("observables", json.dumps(observables))
    monkeypatch.setattr(LogicalCircuitBuilder, "to_tick_circuit", lambda _self: circuit)
    with pytest.raises(ValueError, match="reference 0"):
        t_teleportation_decision_model(patch)


def test_model_matches_program_detector_statistics():
    patch = SurfacePatch.create(distance=3)
    model = t_teleportation_decision_model(patch, 1)
    p = 0.005
    # Guppy/QIS p1 acts on every Helios-native gate inside CX, preparation and
    # measurement. The logical-gate model has no counterpart: a stated omission,
    # not a statistical tolerance. Match p2, measurement and preparation only.
    noise = prs.depolarizing_noise().with_uniform_probability(0.0).with_p2(p).with_p_meas(p).with_p_prep(p)
    results = _run(make_surface_t_teleportation(patch, 1, input_state="X"), patch, shots=400, noise=noise)
    events, _ = model.detection_events(results)
    dem = model.build_dem(p1=0.0, p2=p, p_meas=p, p_prep=p)
    sampled = DemSampler.from_dem_string(str(dem)).sample_batch(20000, seed=42).detector_events()
    width = len(json.loads(model.circuit.get_meta("detectors")))
    program_columns = [[i in row for row in events] for i in range(width)]
    model_columns = list(zip(*sampled, strict=True))
    z_scores = [_rate_z(a, b) for a, b in zip(program_columns, model_columns, strict=True)]
    program_totals, model_totals = list(map(len, events)), list(map(sum, sampled))
    mean_z = abs(statistics.mean(program_totals) - statistics.mean(model_totals)) / math.sqrt(
        statistics.variance(program_totals) / len(program_totals)
        + statistics.variance(model_totals) / len(model_totals),
    )
    # Pairwise co-firing is also an oracle in the design of record. Check all
    # pairs in these same samples with the same Jeffreys posterior calculation.
    pair_z = [
        _rate_z(
            [a and b for a, b in zip(program_columns[i], program_columns[j], strict=True)],
            [a and b for a, b in zip(model_columns[i], model_columns[j], strict=True)],
        )
        for i, j in combinations(range(width), 2)
    ]
    assert max(z_scores) <= 6
    assert mean_z <= 3
    assert max(pair_z) <= 6
    # At 400 shots this checks mapping and algebra under matched channels; it
    # cannot resolve the localized seed-chain/sign-fix omissions (~10000 pooled
    # shots would be needed for their expected ~10% effect).


def test_decoded_decision_beats_raw_on_model_samples():
    model = t_teleportation_decision_model(SurfacePatch.create(distance=3), 1)
    dem = str(model.build_dem(p1=0.001, p2=0.001, p_meas=0.001, p_prep=0.001))
    batch = DemSampler.from_dem_string(dem).sample_batch(4000, seed=42)
    raw_errors = sum(row[0] for row in batch.observable_flips())
    decoded_errors = batch.decode(dem, tesseract()).num_errors
    assert raw_errors > 0
    assert raw_errors >= 5 * decoded_errors


def test_later_records_do_not_change_first_decision():
    patch = SurfacePatch.create(distance=3)
    model = t_teleportation_decision_model(patch, 2)
    results = _zero_results(patch, 2)
    results["data_synz"][0][-1] = 1
    results["resource_synx"][0][-1] = 1
    results["teleportation_resource_readout"][0][patch.geometry.logical_z.data_qubits[0]] = 1
    expected = model.detection_events(results)
    for values in results.values():
        values[0].extend([1] * len(values[0]))
    results["t_correction"] = [[False, True]]
    assert model.detection_events(results) == expected


def test_build_dem_defaults_match_explicit_builder_rates():
    model = t_teleportation_decision_model(SurfacePatch.create(distance=3), 1)
    builder_defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(LogicalCircuitBuilder.build_dem).parameters.items()
        if name != "self"
    }
    assert str(model.build_dem()) == str(DetectorErrorModel.from_circuit(model.circuit, **builder_defaults))
    # p1 and p_meas share from_circuit's defaults, so check they are forwarded too.
    rates = {"p1": 0.03, "p2": 0.001, "p_meas": 0.02, "p_prep": 0.0}
    assert str(model.build_dem(p1=0.03, p_meas=0.02)) == str(DetectorErrorModel.from_circuit(model.circuit, **rates))


@pytest.mark.parametrize("dagger", [False, True])
@pytest.mark.parametrize("rounds_before", [0, 1])
def test_forced_variants_noiseless(rounds_before, dagger):
    patch = SurfacePatch.create(distance=3)
    model = t_teleportation_decision_model(patch, rounds_before)
    for forced in (None, False, True):
        program = make_surface_t_teleportation(
            patch,
            rounds_before,
            input_state="X",
            dagger=dagger,
            forced_decision=forced,
        )
        results = _run(program, patch)
        events, raw = model.detection_events(results)
        assert events == [[] for _ in range(8)]
        assert results["t_correction"] == (raw if forced is None else [forced] * 8)


def _decoded_setup(p=0.005):
    patch = SurfacePatch.create(distance=3)
    model = t_teleportation_decision_model(patch, 1)
    noise = prs.depolarizing_noise().with_uniform_probability(0.0).with_p2(p).with_p_meas(p).with_p_prep(p)
    dem = model.build_dem(p1=0.0, p2=p, p_meas=p, p_prep=p)
    return patch, model, noise, dem


def test_exact_raw_equivalence_noisy():
    patch, model, noise, dem = _decoded_setup()
    calls = []

    def run(forced):
        calls.append(forced)
        program = make_surface_t_teleportation(patch, 1, input_state="X", forced_decision=forced)
        return _run(program, patch, shots=64, noise=noise, seed=712)

    result = model.run_decoded(run, tesseract(), dem=dem)
    assert calls == [False, True]
    assert result.raw_results == run(None)
    independent = {forced: run(forced) for forced in (False, True)}
    disagreements = sum(a != b for a, b in zip(result.raw, result.decoded, strict=True))
    print(f"Exact raw equivalence: seed=712, shots=64, disagreements={disagreements}")
    assert disagreements > 0
    for tag, values in result.results.items():
        assert values == [independent[bit][tag][k] for k, bit in enumerate(result.decoded)]


def test_noiseless_decode():
    patch, model, noise, dem = _decoded_setup(p=0.0)

    def run(forced):
        return _run(make_surface_t_teleportation(patch, 1, forced_decision=forced), patch, noise=noise)

    result = model.run_decoded(run, tesseract(), dem=dem)
    assert result.decoded == result.raw
    assert result.results == result.raw_results


def _forced_round_trip(patch, forced):
    source = render_surface_t_teleportation_module(patch) + f"""
@guppy
def forced_round_trip_{forced}() -> None:
    data = prepare_injected_y()
    resource = prepare_injected_t()
    syn_data = syndrome_extraction_data(data)
    syn_resource = syndrome_extraction_anc(resource)
    output("data_synx", syn_data.synx)
    output("data_synz", syn_data.synz)
    output("resource_synx", syn_resource.synx)
    output("resource_synz", syn_resource.synz)
    consume_t_resource(data, resource)
    correct_t_teleportation(data, {forced}, False)
    resource = prepare_injected_tdg()
    apply_t_teleportation(data, resource, True)
    syn = syndrome_extraction_fold_szdg_data(data)
    output("readout_synx", syn.synx)
    output("readout_synz", syn.synz)
    output("final_data", measure_x_basis(data))
"""
    name = f"test_forced_round_trip_{forced}"
    module = load_guppy_source(source, _get_temp_dir() / f"{name}.py", name)
    return module[f"forced_round_trip_{forced}"]


def test_decoded_round_trip_paired_win():
    patch, model, noise, dem = _decoded_setup()

    def run(forced):
        return _run(_forced_round_trip(patch, forced), patch, shots=400, noise=noise, seed=712)

    result = model.run_decoded(run, tesseract(), dem=dem)
    raw_errors, decoded_errors = [
        [bool(sum(bits[q] for q in patch.geometry.logical_x.data_qubits) % 2) for bits in records["final_data"]]
        for records in (result.raw_results, result.results)
    ]
    paired = defaultdict(int)
    for raw, decoded, raw_error, decoded_error in zip(
        result.raw,
        result.decoded,
        raw_errors,
        decoded_errors,
        strict=True,
    ):
        if raw != decoded:
            paired[raw_error, decoded_error] += 1
    disagreements = sum(paired.values())
    print(
        f"Round trip: disagreements={disagreements}; "
        f"decoded-right/raw-wrong={paired[True, False]}, raw-right/decoded-wrong={paired[False, True]}, "
        f"both-right={paired[False, False]}, both-wrong={paired[True, True]}; "
        f"raw rate={sum(raw_errors) / 400:.4f}, decoded rate={sum(decoded_errors) / 400:.4f}",
    )
    # Seeded samples are not promised stable across Rust releases. Re-pin only
    # if a release moves this sample, and only if the paired win still holds.
    assert disagreements > 0
    assert paired[True, False] > paired[False, True]


def test_run_decoded_rejects_different_seeds():
    patch, model, noise, dem = _decoded_setup()

    def run(forced):
        return _run(
            make_surface_t_teleportation(patch, 1, forced_decision=forced),
            patch,
            noise=noise,
            seed=713 if forced else 712,
        )

    with pytest.raises(ValueError, match=r"Pre-decision.*seed.*before the decision"):
        model.run_decoded(run, tesseract(), dem=dem)


def test_run_decoded_rejects_ignored_forcing():
    patch, model, noise, dem = _decoded_setup()
    results = _run(make_surface_t_teleportation(patch, 1), patch, noise=noise)
    with pytest.raises(ValueError, match=r"t_correction.*forced"):
        model.run_decoded(lambda _forced: results, tesseract(), dem=dem)


def test_run_decoded_rejects_dropped_tag():
    patch, model, noise, dem = _decoded_setup()

    def run(forced):
        results = _run(make_surface_t_teleportation(patch, 1, forced_decision=forced), patch, noise=noise)
        if forced:
            del results["final_data"]
        return results

    with pytest.raises(ValueError, match="same tags"):
        model.run_decoded(run, tesseract(), dem=dem)


def test_decode_decisions_rejects_incompatible_dem():
    patch, model, noise, _dem = _decoded_setup()
    results = _run(make_surface_t_teleportation(patch, 1), patch, noise=noise)
    events, _raw = model.detection_events(results)
    assert any(events)
    dem = t_teleportation_decision_model(patch, 0).build_dem(p1=0.0, p2=0.005, p_meas=0.005, p_prep=0.005)
    with pytest.raises(ValueError, match=r"SampleBatch has .* detectors"):
        model.decode_decisions(results, tesseract(), dem=dem)


def test_forced_decision_validation():
    patch = SurfacePatch.create(distance=3)
    factories = [
        lambda **kwargs: make_surface_t_teleportation(patch, **kwargs),
        load_surface_t_teleportation_module(patch)["make_t_teleportation"],
    ]
    for factory in factories:
        for invalid in (0, 1, "True", np.bool_(True)):
            with pytest.raises(ValueError, match="forced_decision"):
                factory(forced_decision=invalid)
        for valid in (None, False, True):
            assert factory(forced_decision=valid) is not None
    with pytest.raises(ValueError, match=r"forced_decision.*hook"):
        make_surface_t_teleportation(patch, resource_preparation="hook", forced_decision=False)


@pytest.mark.parametrize("both_variants", [False, True])
def test_run_decoded_rejects_inconsistent_shot_counts(both_variants):
    patch, model, _noise, dem = _decoded_setup(p=0.0)

    def run(forced):
        results = _zero_results(patch, 1)
        results["t_correction"] = [forced]
        results["final_data"] = [] if forced or both_variants else [[0] * patch.geometry.num_data]
        return results

    with pytest.raises(ValueError, match="same shot count"):
        model.run_decoded(run, tesseract(), dem=dem)
