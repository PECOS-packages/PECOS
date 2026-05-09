# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Slow traced-QIS integration tests for the raw-measurement pipeline."""

import json
import math

import numpy as np
import pytest
from pecos.qec.surface import SurfacePatch
from pecos.qec.surface.circuit_builder import tick_circuit_to_stim
from pecos.qec.surface.decode import _build_surface_tick_circuit_for_native_model
from pecos_rslib.qec import DemSampler
from pecos_rslib_exp import depolarizing, fault_catalog, meas_sampling, sim_neo

pymatching = pytest.importorskip("pymatching")
stim = pytest.importorskip("stim")

pytestmark = pytest.mark.slow


def _noise_args(error_rate=0.003):
    return {
        "p1": error_rate * 0.1,
        "p2": error_rate,
        "p_meas": error_rate * 0.5,
        "p_prep": error_rate * 0.5,
    }


def _depolarizing_noise(noise_args):
    return (
        depolarizing()
        .p1(noise_args["p1"])
        .p2(noise_args["p2"])
        .p_meas(noise_args["p_meas"])
        .p_prep(noise_args["p_prep"])
    )


def _build_lowered_traced_qis_surface_code(distance, rounds, basis="Z"):
    patch = SurfacePatch.create(distance=distance)
    circuit = _build_surface_tick_circuit_for_native_model(patch, rounds, basis, circuit_source="traced_qis")
    circuit.lower_clifford_rotations()
    return circuit


def _pymatching_decoder(circuit, noise_args):
    stim_str = tick_circuit_to_stim(circuit, **noise_args)
    dem = stim.Circuit(stim_str).detector_error_model(decompose_errors=True)
    return pymatching.Matching.from_detector_error_model(dem)


def _extract_observable_mask(row, observables, num_measurements):
    mask = 0
    for obs_index, obs in enumerate(observables):
        value = 0
        for rec in obs["records"]:
            idx = num_measurements + rec
            if 0 <= idx < len(row):
                value ^= int(row[idx])
        if value:
            mask |= 1 << obs_index
    return mask


def _decode_raw_measurements(result, circuit, matching, shots):
    detectors = json.loads(circuit.get_meta("detectors"))
    observables = json.loads(circuit.get_meta("observables") or "[]")
    num_measurements = int(circuit.get_meta("num_measurements"))
    syndrome = np.zeros(len(detectors), dtype=np.uint8)

    errors = 0
    for shot_index in range(shots):
        row = result[shot_index]
        syndrome.fill(0)

        for det_index, det in enumerate(detectors):
            value = 0
            for rec in det["records"]:
                idx = num_measurements + rec
                if 0 <= idx < len(row):
                    value ^= int(row[idx])
            syndrome[det_index] = value

        predicted = matching.decode(syndrome)
        predicted_mask = sum(int(bit) << index for index, bit in enumerate(predicted))
        actual_mask = _extract_observable_mask(row, observables, num_measurements)
        errors += predicted_mask != actual_mask

    return errors


def _decode_native_dem_samples(circuit, noise_args, matching, shots, seed):
    sampler = DemSampler.from_circuit(circuit, **noise_args)
    batch = sampler.generate_samples(shots, seed=seed)
    syndrome = np.zeros(sampler.num_detectors, dtype=np.uint8)

    errors = 0
    for shot_index in range(shots):
        sampled_syndrome = batch.get_syndrome(shot_index)
        for det_index in range(sampler.num_detectors):
            syndrome[det_index] = sampled_syndrome[det_index]
        predicted = matching.decode(syndrome)
        predicted_mask = sum(int(bit) << index for index, bit in enumerate(predicted))
        errors += predicted_mask != batch.get_observable_mask(shot_index)

    return errors


def _assert_statistically_consistent(meas_errors, native_errors, shots):
    meas_ler = meas_errors / shots
    native_ler = native_errors / shots
    pooled = (meas_errors + native_errors) / (2 * shots)
    variance = 2 * max(pooled * (1 - pooled), 1 / shots) / shots
    tolerance = max(0.04, 7 * math.sqrt(variance))

    assert abs(meas_ler - native_ler) <= tolerance, (
        "meas_sampling and native DEM LERs differ more than stochastic tolerance: "
        f"meas={meas_errors}/{shots} ({meas_ler:.4f}), "
        f"native={native_errors}/{shots} ({native_ler:.4f}), "
        f"tolerance={tolerance:.4f}"
    )


@pytest.mark.parametrize(
    ("distance", "rounds", "shots"),
    [
        (3, 6, 2_500),
        (5, 10, 2_500),
    ],
)
def test_traced_qis_meas_sampling_ler_tracks_native_dem_pymatching(distance, rounds, shots):
    noise_args = _noise_args()
    circuit = _build_lowered_traced_qis_surface_code(distance, rounds)
    matching = _pymatching_decoder(circuit, noise_args)

    raw_result = (
        sim_neo(circuit).quantum(meas_sampling()).noise(_depolarizing_noise(noise_args)).shots(shots).seed(1234).run()
    )
    meas_errors = _decode_raw_measurements(raw_result, circuit, matching, shots)
    native_errors = _decode_native_dem_samples(circuit, noise_args, matching, shots, seed=5678)

    assert 0 <= meas_errors <= shots
    assert 0 <= native_errors <= shots
    _assert_statistically_consistent(meas_errors, native_errors, shots)


def test_d3_traced_qis_zero_noise_pymatching_pipeline_has_no_logical_errors():
    noise_args = _noise_args(error_rate=0.0)
    circuit = _build_lowered_traced_qis_surface_code(distance=3, rounds=3)
    matching = _pymatching_decoder(circuit, noise_args)
    shots = 64

    raw_result = (
        sim_neo(circuit).quantum(meas_sampling()).noise(_depolarizing_noise(noise_args)).shots(shots).seed(2468).run()
    )
    meas_errors = _decode_raw_measurements(raw_result, circuit, matching, shots)
    native_errors = _decode_native_dem_samples(circuit, noise_args, matching, shots, seed=1357)

    assert meas_errors == 0
    assert native_errors == 0


def test_d3_traced_qis_fault_catalog_builds_with_all_noise_channels_enabled():
    noise_args = _noise_args()
    circuit = _build_lowered_traced_qis_surface_code(distance=3, rounds=9)
    catalog = fault_catalog(circuit, _depolarizing_noise(noise_args))

    alternative_counts = [len(location.faults) for location in catalog]
    assert len(catalog) > 100
    assert 1 in alternative_counts
    assert 3 in alternative_counts
    assert 15 in alternative_counts
    assert sum(alternative_counts) > 1_000

    first_event = next(catalog.fault_configurations(1))
    assert len(first_event.locations) == 1
    assert len(first_event.faults) == 1
    assert first_event.locations[0] is catalog.locations[first_event.location_indices[0]]
    assert first_event.faults[0] is first_event.locations[0].faults[first_event.alternative_indices[0]]
