from __future__ import annotations

import numpy as np
import pytest
from pecos.qec.surface import (
    GuppyRngMaskConfig,
    NoiseParameters,
    SurfacePatch,
    TwirlConfig,
    build_memory_circuit,
    build_native_sampler,
    decode_native_samples,
    demask_pauli_frame_records,
)
from pecos.qec.surface._twirl_sites import (
    mask_col_for,
    mask_col_for_gate_operand,
    num_pauli_sites,
    num_pauli_sites_for_schedule,
    num_two_qubit_gate_twirl_sites,
    pauli_active_gate_tag,
    pauli_active_round_tag,
    pauli_mask_gate_tag,
    pauli_mask_round_tag,
)
from pecos.qec.surface.decode import (
    _extract_pauli_activations_from_results,
    _extract_pauli_masks_from_results,
    generate_circuit_level_dem_from_builder,
)


def test_extract_pauli_masks_packs_bits_in_row_major_site_qubit_order() -> None:
    results = {
        pauli_mask_round_tag(0): [[1, 0, 1, 1]],
        pauli_mask_round_tag(1): [[0, 1, 0, 0]],
    }

    mask = _extract_pauli_masks_from_results(
        results,
        num_rounds=3,
        num_data=2,
        num_shots=1,
    )

    assert mask.dtype == np.uint8
    assert mask.tolist() == [[1, 3, 2, 0]]
    assert mask[0, mask_col_for(0, 0, 2)] == 1
    assert mask[0, mask_col_for(0, 1, 2)] == 3
    assert mask[0, mask_col_for(1, 0, 2)] == 2
    assert mask[0, mask_col_for(1, 1, 2)] == 0


def test_extract_scaled_round_twirl_requires_and_validates_activation_tags() -> None:
    scaled = TwirlConfig(twirl_probability=0.5)
    results = {
        pauli_mask_round_tag(0): [[1, 0, 0, 0]],
        pauli_active_round_tag(0): [[True, False]],
    }

    mask = _extract_pauli_masks_from_results(
        results,
        num_rounds=2,
        num_data=2,
        num_shots=1,
        twirl=scaled,
    )
    active = _extract_pauli_activations_from_results(
        results,
        num_rounds=2,
        num_data=2,
        num_shots=1,
        twirl=scaled,
    )

    assert mask.tolist() == [[1, 0]]
    assert active.tolist() == [[True, False]]

    with pytest.raises(ValueError, match="missing Pauli-activation result tag"):
        _extract_pauli_masks_from_results(
            {pauli_mask_round_tag(0): [[0, 0, 0, 0]]},
            num_rounds=2,
            num_data=2,
            num_shots=1,
            twirl=scaled,
        )

    malformed = {
        pauli_mask_round_tag(0): [[1, 0, 0, 0]],
        pauli_active_round_tag(0): [[False, True]],
    }
    with pytest.raises(ValueError, match="inactive round site recorded a non-identity"):
        _extract_pauli_masks_from_results(
            malformed,
            num_rounds=2,
            num_data=2,
            num_shots=1,
            twirl=scaled,
        )


def test_extract_pauli_masks_rejects_missing_or_misshaped_tags() -> None:
    with pytest.raises(ValueError, match="missing Pauli-mask result tag"):
        _extract_pauli_masks_from_results({}, num_rounds=2, num_data=1, num_shots=1)

    with pytest.raises(ValueError, match=r"expected \(1, 4\)"):
        _extract_pauli_masks_from_results(
            {pauli_mask_round_tag(0): [[1, 0, 1]]},
            num_rounds=2,
            num_data=2,
            num_shots=1,
        )


def test_extract_gate_local_pauli_masks_packs_operand_order() -> None:
    patch = SurfacePatch.create(distance=3)
    results = {
        pauli_mask_gate_tag(site): [[0, 0, 0, 0]]
        for site in range(num_two_qubit_gate_twirl_sites(patch, num_rounds=1, basis="Z"))
    }
    results[pauli_mask_gate_tag(0)] = [[1, 0, 0, 1]]

    mask = _extract_pauli_masks_from_results(
        results,
        num_rounds=1,
        num_data=patch.geometry.num_data,
        num_shots=1,
        patch=patch,
        basis="Z",
        twirl=TwirlConfig(site_schedule="before_two_qubit_gate"),
    )

    assert mask.dtype == np.uint8
    assert mask.shape == (
        1,
        num_pauli_sites_for_schedule(
            patch,
            num_rounds=1,
            basis="Z",
            site_schedule="before_two_qubit_gate",
        ),
    )
    assert mask[0, mask_col_for_gate_operand(0, 0)] == 1
    assert mask[0, mask_col_for_gate_operand(0, 1)] == 2


def test_extract_scaled_gate_local_twirl_validates_activation_tags() -> None:
    patch = SurfacePatch.create(distance=3)
    twirl = TwirlConfig(site_schedule="before_two_qubit_gate", twirl_probability=0.5)
    results = {
        pauli_mask_gate_tag(site): [[0, 0, 0, 0]]
        for site in range(num_two_qubit_gate_twirl_sites(patch, num_rounds=1, basis="Z"))
    }
    results.update(
        {
            pauli_active_gate_tag(site): [[True, True]]
            for site in range(num_two_qubit_gate_twirl_sites(patch, num_rounds=1, basis="Z"))
        },
    )
    results[pauli_mask_gate_tag(0)] = [[1, 0, 0, 0]]
    results[pauli_active_gate_tag(0)] = [[False, True]]

    with pytest.raises(ValueError, match="inactive gate-local site recorded a non-identity"):
        _extract_pauli_masks_from_results(
            results,
            num_rounds=1,
            num_data=patch.geometry.num_data,
            num_shots=1,
            patch=patch,
            basis="Z",
            twirl=twirl,
        )


def test_demask_helper_cancels_known_pauli_frame_xor() -> None:
    patch = SurfacePatch.create(distance=3)
    sampler = build_native_sampler(
        patch,
        num_rounds=2,
        noise=NoiseParameters(),
        basis="Z",
        twirl=TwirlConfig(),
    )
    assert sampler.pauli_frame_lookup is not None

    masks = np.zeros((2, sampler.num_pauli_sites), dtype=np.uint8)
    masks[0, 0] = 1
    masks[1, 0] = 3

    det_xor, obs_xor = sampler.pauli_frame_lookup.compute_mask_xor(masks.astype(np.int64))
    physical_events = np.zeros((2, sampler.num_detectors), dtype=bool)
    physical_obs = np.zeros((2, sampler.num_observables), dtype=bool)
    physical_events[0, 0] = True
    physical_obs[1, 0] = True

    raw_events = physical_events ^ np.asarray(det_xor, dtype=bool)
    raw_obs = physical_obs ^ np.asarray(obs_xor, dtype=bool)

    events, observables = demask_pauli_frame_records(
        sampler.pauli_frame_lookup,
        raw_events,
        raw_obs,
        masks,
    )

    np.testing.assert_array_equal(events, physical_events)
    np.testing.assert_array_equal(observables, physical_obs)


def test_native_sampler_accepts_harvested_uint8_pauli_masks() -> None:
    patch = SurfacePatch.create(distance=3)
    sampler = build_native_sampler(
        patch,
        num_rounds=2,
        noise=NoiseParameters(),
        basis="Z",
        twirl=TwirlConfig(),
    )
    assert sampler.num_pauli_sites > 0

    masks = np.zeros((4, sampler.num_pauli_sites), dtype=np.uint8)
    masks[:, 0] = [0, 1, 2, 3]

    det_events, obs_flips = sampler.sample(4, seed=123, pauli_masks=masks)
    assert det_events.shape == (4, sampler.num_detectors)
    assert obs_flips.shape == (4, sampler.num_observables)

    assert decode_native_samples(sampler, 4, seed=123, pauli_masks=masks) == 0


def test_canonical_frame_output_reuses_raw_abstract_sampler_topology() -> None:
    patch = SurfacePatch.create(distance=3)
    raw = build_native_sampler(
        patch,
        num_rounds=2,
        noise=NoiseParameters(),
        basis="Z",
        twirl=TwirlConfig(),
    )
    canonical = build_native_sampler(
        patch,
        num_rounds=2,
        noise=NoiseParameters(),
        basis="Z",
        twirl=TwirlConfig(frame_output="canonical"),
    )
    scaled = build_native_sampler(
        patch,
        num_rounds=2,
        noise=NoiseParameters(),
        basis="Z",
        twirl=TwirlConfig(twirl_probability=0.5),
    )

    assert canonical.num_detectors == raw.num_detectors
    assert canonical.num_observables == raw.num_observables
    assert canonical.num_pauli_sites == raw.num_pauli_sites
    assert canonical.pauli_frame_lookup is raw.pauli_frame_lookup
    assert canonical.dem_string == raw.dem_string
    assert scaled.num_detectors == raw.num_detectors
    assert scaled.num_observables == raw.num_observables
    assert scaled.num_pauli_sites == raw.num_pauli_sites
    assert scaled.pauli_frame_lookup is raw.pauli_frame_lookup
    assert scaled.dem_string == raw.dem_string


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scheme", "clifford"),
        ("site_schedule", "per_two_qubit_gate"),
        ("result_encoding", "bogus"),
        ("frame_output", "physical"),
        ("twirl_probability", -0.1),
        ("twirl_probability", 1.1),
    ],
)
def test_abstract_twirl_builders_reject_unsupported_config(
    field: str,
    value: object,
) -> None:
    patch = SurfacePatch.create(distance=3)
    kwargs = {field: value}
    twirl = TwirlConfig(**kwargs)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match=field):
        twirl.validate_runtime_supported()

    with pytest.raises(ValueError, match=field):
        build_memory_circuit(
            patch=patch,
            rounds=2,
            basis="Z",
            twirl=twirl,
        )

    with pytest.raises(ValueError, match=field):
        build_native_sampler(
            patch,
            num_rounds=2,
            noise=NoiseParameters(),
            basis="Z",
            twirl=twirl,
        )

    with pytest.raises(ValueError, match=field):
        generate_circuit_level_dem_from_builder(
            patch,
            num_rounds=2,
            noise=NoiseParameters(),
            basis="Z",
            twirl=twirl,
        )


def test_twirl_sine_law_idle_noise_builds_dem_and_sampler() -> None:
    patch = SurfacePatch.create(distance=3)
    noise = NoiseParameters().with_p_idle_sin_squared(0.03, {"X": 1.0})
    twirl = TwirlConfig()

    dem = generate_circuit_level_dem_from_builder(
        patch,
        num_rounds=2,
        noise=noise,
        basis="Z",
        decompose_errors=True,
        twirl=twirl,
    )
    assert "error(" in dem

    sampler = build_native_sampler(
        patch,
        num_rounds=2,
        noise=noise,
        basis="Z",
        twirl=twirl,
    )
    assert sampler.num_pauli_sites == num_pauli_sites(2, patch.geometry.num_data)

    det_events, obs_flips = sampler.sample(2, seed=7)
    assert det_events.shape == (2, sampler.num_detectors)
    assert obs_flips.shape == (2, sampler.num_observables)


@pytest.mark.parametrize(
    ("label", "noise"),
    [
        ("depolarizing", NoiseParameters(p1=0.001, p2=0.01, p_meas=0.001, p_prep=0.001)),
        ("uniform_idle", NoiseParameters(p_idle=0.002)),
        ("t1_t2", NoiseParameters(t1=1000.0, t2=800.0)),
        ("linear_idle", NoiseParameters().with_p_idle_linear(0.001, {"Z": 1.0})),
        ("z_sine_law_idle", NoiseParameters().with_p_idle_sin_squared(0.01, {"Z": 1.0})),
        ("x_sine_law_idle", NoiseParameters().with_p_idle_sin_squared(0.03, {"X": 1.0})),
    ],
)
def test_twirling_does_not_change_canonical_dem(
    label: str,
    noise: NoiseParameters,
) -> None:
    del label
    patch = SurfacePatch.create(distance=3)

    untwirled = generate_circuit_level_dem_from_builder(
        patch,
        num_rounds=2,
        noise=noise,
        basis="Z",
        decompose_errors=True,
    )
    twirled = generate_circuit_level_dem_from_builder(
        patch,
        num_rounds=2,
        noise=noise,
        basis="Z",
        decompose_errors=True,
        twirl=TwirlConfig(),
    )

    assert twirled == untwirled


def test_gate_local_twirling_does_not_change_canonical_dem() -> None:
    patch = SurfacePatch.create(distance=3)
    noise = NoiseParameters(p1=0.001, p2=0.01, p_meas=0.001, p_prep=0.001)

    untwirled = generate_circuit_level_dem_from_builder(
        patch,
        num_rounds=2,
        noise=noise,
        basis="Z",
        decompose_errors=True,
    )
    twirled = generate_circuit_level_dem_from_builder(
        patch,
        num_rounds=2,
        noise=noise,
        basis="Z",
        decompose_errors=True,
        twirl=TwirlConfig(site_schedule="before_two_qubit_gate"),
    )

    assert twirled == untwirled


def test_surface_traced_qis_rejects_twirl_with_semantic_message() -> None:
    patch = SurfacePatch.create(distance=3)

    with pytest.raises(ValueError, match="one concrete mask realization") as exc_info:
        build_memory_circuit(
            patch=patch,
            rounds=2,
            basis="Z",
            circuit_source="traced_qis",
            twirl=TwirlConfig(),
        )

    msg = str(exc_info.value)
    assert "one concrete mask realization" in msg
    assert "result-id provenance" in msg
    assert "circuit_source='abstract'" in msg


@pytest.mark.parametrize("distance", [3, 5])
@pytest.mark.parametrize("num_rounds", [2, 3])
def test_tracked_pauli_label_order_matches_mask_col_for(
    distance: int,
    num_rounds: int,
) -> None:
    patch = SurfacePatch.create(distance=distance)
    num_data = patch.geometry.num_data
    tc = build_memory_circuit(
        patch=patch,
        rounds=num_rounds,
        basis="Z",
        twirl=TwirlConfig(),
    )
    tracked = [a for a in tc.annotations() if a["kind"] == "tracked_pauli"]

    assert len(tracked) == 3 * num_pauli_sites(num_rounds, num_data)
    for site in range(num_rounds - 1):
        for q in range(num_data):
            col = mask_col_for(site, q, num_data)
            base = 3 * col
            for offset, kind in enumerate(("X", "Y", "Z")):
                assert tracked[base + offset]["label"] == f"twirl_s{site}_q{q}_{kind}"


def test_gate_local_tracked_pauli_label_order_matches_gate_operand_cols() -> None:
    from pecos.qec.surface.schedule import compute_cnot_schedule

    patch = SurfacePatch.create(distance=3)
    num_rounds = 2
    tc = build_memory_circuit(
        patch=patch,
        rounds=num_rounds,
        basis="Z",
        twirl=TwirlConfig(site_schedule="before_two_qubit_gate"),
    )
    tracked = [a for a in tc.annotations() if a["kind"] == "tracked_pauli"]
    expected_cols = num_pauli_sites_for_schedule(
        patch,
        num_rounds=num_rounds,
        basis="Z",
        site_schedule="before_two_qubit_gate",
    )
    assert len(tracked) == 3 * expected_cols

    first_init_x_gate = next(
        (stab_idx, data_idx)
        for cx_round in compute_cnot_schedule(patch)
        for stab_type, stab_idx, data_idx in cx_round
        if stab_type == "X"
    )
    stab_idx, data_idx = first_init_x_gate
    first_control = patch.geometry.num_data + stab_idx
    expected_first_labels = [
        *(f"twirl_g0o0_q{first_control}_{kind}" for kind in ("X", "Y", "Z")),
        *(f"twirl_g0o1_q{data_idx}_{kind}" for kind in ("X", "Y", "Z")),
    ]
    assert [tracked[i]["label"] for i in range(6)] == expected_first_labels


def test_raw_twirled_guppy_trace_result_provenance_ignores_sideband_tags() -> None:
    pytest.importorskip("guppylang")
    pytest.importorskip("selene_sim")

    from pecos.guppy_gen import get_num_qubits
    from pecos.guppy_gen.surface import generate_memory_experiment
    from pecos.qec.surface.decode import (
        _index_surface_result_trace_ids,
        trace_guppy_into_tick_circuit_with_result_traces,
    )

    patch = SurfacePatch.create(distance=3)
    num_rounds = 2
    twirl = TwirlConfig(twirl_probability=0.5)
    program = generate_memory_experiment(
        patch,
        num_rounds=num_rounds,
        basis="Z",
        twirl=twirl,
        rng=GuppyRngMaskConfig(seed=0),
    )
    _, result_traces = trace_guppy_into_tick_circuit_with_result_traces(
        program,
        get_num_qubits(patch=patch, twirl=twirl),
        seed=0,
    )

    sideband_names = {trace.get("name") for trace in result_traces if isinstance(trace.get("name"), str)}
    assert any(str(name).startswith("pauli_mask:") for name in sideband_names)
    assert any(str(name).startswith("pauli_active:") for name in sideband_names)
    assert "frame_mode:raw" in sideband_names

    scalar_trace_ids, array_trace_ids = _index_surface_result_trace_ids(result_traces)
    indexed_names = set(scalar_trace_ids) | set(array_trace_ids)
    assert any(name.startswith("sx") and ":meas:" in name for name in indexed_names)
    assert "final" in indexed_names
    assert not any(name.startswith(("pauli_mask:", "pauli_active:", "frame_mode:", "raw:")) for name in indexed_names)
