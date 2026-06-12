from __future__ import annotations

import numpy as np
import pytest
from pecos.qec.surface import (
    GuppyRngMaskConfig,
    NoiseModel,
    SurfacePatch,
    TwirlConfig,
    build_memory_circuit,
    build_native_sampler,
    decode_native_samples,
    demask_pauli_frame_records,
    extract_detection_events_and_observables,
    sample_pauli_masks_from_guppy,
)
from pecos.qec.surface._twirl_sites import num_pauli_sites
from pecos.qec.surface.decode import _extract_pauli_masks_from_results

pytest.importorskip("guppylang")
pytest.importorskip("selene_sim")


@pytest.fixture
def patch_d3() -> SurfacePatch:
    return SurfacePatch.create(distance=3)


def _bool_array_from_indices(rows: list[list[int]], width: int) -> np.ndarray:
    out = np.zeros((len(rows), width), dtype=bool)
    for shot, indices in enumerate(rows):
        for idx in indices:
            out[shot, idx] = True
    return out


def _canonicalize_raw_rows_from_masks(
    patch: SurfacePatch,
    *,
    basis: str,
    num_rounds: int,
    raw_rows: list[list[int]],
    masks: np.ndarray,
) -> list[list[int]]:
    """Apply the same Pauli-frame rule as the generated canonical tracker."""
    num_data = patch.geometry.num_data
    init_count = (
        len(patch.geometry.x_stabilizers)
        if basis.upper() == "Z"
        else len(patch.geometry.z_stabilizers)
    )
    expected_width = init_count + num_rounds * patch.geometry.num_ancilla + num_data
    canonical_rows: list[list[int]] = []

    for shot, raw_row in enumerate(raw_rows):
        assert len(raw_row) == expected_width
        fx = [False] * num_data
        fz = [False] * num_data
        row: list[int] = []
        offset = 0

        for _ in range(init_count):
            row.append(int(raw_row[offset]))
            offset += 1

        for r in range(num_rounds):
            for stab in patch.geometry.x_stabilizers:
                flip = False
                for q in stab.data_qubits:
                    flip ^= fz[q]
                row.append(int(bool(raw_row[offset]) ^ flip))
                offset += 1
            for stab in patch.geometry.z_stabilizers:
                flip = False
                for q in stab.data_qubits:
                    flip ^= fx[q]
                row.append(int(bool(raw_row[offset]) ^ flip))
                offset += 1
            if r < num_rounds - 1:
                site_offset = r * num_data
                for q in range(num_data):
                    code = int(masks[shot, site_offset + q])
                    fx[q] ^= code in (1, 2)
                    fz[q] ^= code in (2, 3)

        final_flip = fx if basis.upper() == "Z" else fz
        for q in range(num_data):
            row.append(int(bool(raw_row[offset]) ^ final_flip[q]))
            offset += 1

        assert offset == expected_width
        canonical_rows.append(row)

    return canonical_rows


def _run_twirled_guppy_rows_masks_and_raw(
    patch: SurfacePatch,
    *,
    basis: str,
    num_rounds: int,
    num_shots: int,
    rng: GuppyRngMaskConfig,
    twirl: TwirlConfig,
) -> tuple[list[list[int]], np.ndarray, list[list[int]], list[str]]:
    from pecos.compilation_pipeline import compile_guppy_to_hugr
    from pecos.guppy.surface import generate_memory_experiment, get_num_qubits
    from selene_sim import SimpleRuntime, Stim, build

    fn = generate_memory_experiment(
        patch,
        num_rounds=num_rounds,
        basis=basis,
        twirl=twirl,
        rng=rng,
    )
    hugr_bytes = compile_guppy_to_hugr(fn)
    instance = build(
        hugr_bytes,
        name=f"pauli_twirl_null_d{patch.geometry.dx}_r{num_rounds}_{basis.lower()}",
    )

    mask_results: dict[str, list[list[int]]] = {}
    measurement_rows: list[list[int]] = []
    raw_rows: list[list[int]] = []
    frame_modes: list[str] = []
    for shot_results in instance.run_shots(
        simulator=Stim(random_seed=int(rng.seed)),
        n_qubits=get_num_qubits(patch=patch, twirl=twirl),
        n_shots=num_shots,
        runtime=SimpleRuntime(),
        n_processes=1,
    ):
        row: list[int] = []
        raw_row: list[int] = []
        saw_final = False
        saw_raw_final = False
        frame_mode: str | None = None

        for name, values in shot_results:
            try:
                shot_value = list(values)
            except TypeError:
                shot_value = [values]

            if name.startswith("pauli_mask:round:"):
                mask_results.setdefault(name, []).append([int(v) for v in shot_value])
            elif name.startswith("frame_mode:"):
                assert len(shot_value) == 1
                assert bool(shot_value[0])
                frame_mode = name.removeprefix("frame_mode:")
            elif name.startswith("raw:") and ":bit:" in name:
                assert len(shot_value) == 1
                raw_row.append(int(shot_value[0]))
            elif name == "raw:final":
                raw_row.extend(int(v) for v in shot_value)
                saw_raw_final = True
            elif ":meas:" in name:
                assert len(shot_value) == 1
                bit = int(shot_value[0])
                row.append(bit)
                if twirl.frame_output == "canonical" and ":init:meas:" in name:
                    raw_row.append(bit)
            elif name == "final":
                row.extend(int(v) for v in shot_value)
                saw_final = True

        assert saw_final
        assert frame_mode == twirl.frame_output
        if twirl.frame_output == "canonical":
            assert saw_raw_final
        measurement_rows.append(row)
        raw_rows.append(raw_row)
        frame_modes.append(frame_mode)

    masks = _extract_pauli_masks_from_results(
        mask_results,
        num_rounds=num_rounds,
        num_data=patch.geometry.num_data,
        num_shots=num_shots,
    )
    return measurement_rows, masks, raw_rows, frame_modes


def _run_twirled_guppy_measurement_rows_and_masks(
    patch: SurfacePatch,
    *,
    basis: str,
    num_rounds: int,
    num_shots: int,
    rng: GuppyRngMaskConfig,
) -> tuple[list[list[int]], np.ndarray]:
    rows, masks, _, _ = _run_twirled_guppy_rows_masks_and_raw(
        patch,
        basis=basis,
        num_rounds=num_rounds,
        num_shots=num_shots,
        rng=rng,
        twirl=TwirlConfig(),
    )
    return rows, masks


def test_runtime_twirl_masks_vary_across_shots(patch_d3: SurfacePatch) -> None:
    masks = sample_pauli_masks_from_guppy(
        patch_d3,
        num_rounds=3,
        num_shots=6,
        basis="Z",
        twirl=TwirlConfig(),
        rng=GuppyRngMaskConfig(seed=11),
    )

    unique_rows = np.unique(masks, axis=0)
    assert unique_rows.shape[0] >= 2


def test_same_seed_is_reproducible(patch_d3: SurfacePatch) -> None:
    kwargs = {
        "num_rounds": 3,
        "num_shots": 6,
        "basis": "Z",
        "twirl": TwirlConfig(),
        "rng": GuppyRngMaskConfig(seed=42),
    }

    masks_a = sample_pauli_masks_from_guppy(patch_d3, **kwargs)
    masks_b = sample_pauli_masks_from_guppy(patch_d3, **kwargs)

    np.testing.assert_array_equal(masks_a, masks_b)


def test_different_seeds_differ(patch_d3: SurfacePatch) -> None:
    common = {
        "num_rounds": 3,
        "num_shots": 8,
        "basis": "Z",
        "twirl": TwirlConfig(),
    }

    masks_seed1 = sample_pauli_masks_from_guppy(
        patch_d3,
        rng=GuppyRngMaskConfig(seed=1),
        **common,
    )
    masks_seed2 = sample_pauli_masks_from_guppy(
        patch_d3,
        rng=GuppyRngMaskConfig(seed=2),
        **common,
    )

    assert not np.array_equal(masks_seed1, masks_seed2)


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_runtime_twirled_theta0_demask_null(
    patch_d3: SurfacePatch,
    basis: str,
) -> None:
    num_rounds = 2
    num_shots = 6
    measurement_rows, masks = _run_twirled_guppy_measurement_rows_and_masks(
        patch_d3,
        basis=basis,
        num_rounds=num_rounds,
        num_shots=num_shots,
        rng=GuppyRngMaskConfig(seed=12345),
    )
    assert masks.any()

    tick_circuit = build_memory_circuit(
        patch=patch_d3,
        rounds=num_rounds,
        basis=basis,
        twirl=TwirlConfig(),
    )
    sampler = build_native_sampler(
        patch_d3,
        num_rounds=num_rounds,
        noise=NoiseModel(),
        basis=basis,
        twirl=TwirlConfig(),
    )
    assert sampler.pauli_frame_lookup is not None

    events_per_shot, obs_per_shot = extract_detection_events_and_observables(
        tick_circuit,
        measurement_rows,
    )
    raw_events = _bool_array_from_indices(events_per_shot, sampler.num_detectors)
    raw_obs = _bool_array_from_indices(obs_per_shot, sampler.num_observables)

    events, observables = demask_pauli_frame_records(
        sampler.pauli_frame_lookup,
        raw_events,
        raw_obs,
        masks,
    )

    assert not events.any()
    assert not observables.any()


def _assert_canonical_frame_output_matches_lookup(
    patch: SurfacePatch,
    *,
    basis: str,
    num_rounds: int,
    num_shots: int,
    seed: int,
) -> None:
    twirl = TwirlConfig(frame_output="canonical")
    measurement_rows, masks, raw_rows, frame_modes = _run_twirled_guppy_rows_masks_and_raw(
        patch,
        basis=basis,
        num_rounds=num_rounds,
        num_shots=num_shots,
        rng=GuppyRngMaskConfig(seed=seed),
        twirl=twirl,
    )

    assert frame_modes == ["canonical"] * num_shots
    assert masks.any()
    assert raw_rows

    expected_rows = _canonicalize_raw_rows_from_masks(
        patch,
        basis=basis,
        num_rounds=num_rounds,
        raw_rows=raw_rows,
        masks=masks,
    )
    assert measurement_rows == expected_rows

    tick_circuit = build_memory_circuit(
        patch=patch,
        rounds=num_rounds,
        basis=basis,
        twirl=TwirlConfig(),
    )
    sampler = build_native_sampler(
        patch,
        num_rounds=num_rounds,
        noise=NoiseModel(),
        basis=basis,
        twirl=TwirlConfig(),
    )
    assert sampler.pauli_frame_lookup is not None

    raw_events_per_shot, raw_obs_per_shot = extract_detection_events_and_observables(
        tick_circuit,
        raw_rows,
    )
    raw_events = _bool_array_from_indices(raw_events_per_shot, sampler.num_detectors)
    raw_obs = _bool_array_from_indices(raw_obs_per_shot, sampler.num_observables)
    assert raw_events.any() or raw_obs.any()

    demasked_events, demasked_obs = demask_pauli_frame_records(
        sampler.pauli_frame_lookup,
        raw_events,
        raw_obs,
        masks,
    )

    events_per_shot, obs_per_shot = extract_detection_events_and_observables(
        tick_circuit,
        measurement_rows,
    )
    events = _bool_array_from_indices(events_per_shot, sampler.num_detectors)
    observables = _bool_array_from_indices(obs_per_shot, sampler.num_observables)

    np.testing.assert_array_equal(events, demasked_events)
    np.testing.assert_array_equal(observables, demasked_obs)
    assert not events.any()
    assert not observables.any()


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_runtime_canonical_frame_output_theta0_null_and_raw_equivalence(
    patch_d3: SurfacePatch,
    basis: str,
) -> None:
    _assert_canonical_frame_output_matches_lookup(
        patch_d3,
        basis=basis,
        num_rounds=2,
        num_shots=6,
        seed=12345,
    )


@pytest.mark.parametrize(
    ("distance", "num_rounds", "basis", "num_shots", "seed"),
    [
        (3, 3, "Z", 4, 2024),
        (3, 3, "X", 4, 2025),
        (5, 3, "Z", 2, 2026),
    ],
)
def test_runtime_canonical_frame_output_matches_lookup_matrix(
    distance: int,
    num_rounds: int,
    basis: str,
    num_shots: int,
    seed: int,
) -> None:
    patch = SurfacePatch.create(distance=distance)
    _assert_canonical_frame_output_matches_lookup(
        patch,
        basis=basis,
        num_rounds=num_rounds,
        num_shots=num_shots,
        seed=seed,
    )


def test_harvested_runtime_masks_drive_fixed_dem_sampler_null(patch_d3: SurfacePatch) -> None:
    num_rounds = 3
    num_shots = 8
    twirl = TwirlConfig()
    masks = sample_pauli_masks_from_guppy(
        patch_d3,
        num_rounds=num_rounds,
        num_shots=num_shots,
        basis="Z",
        twirl=twirl,
        rng=GuppyRngMaskConfig(seed=5150),
    )
    assert masks.any()

    sampler = build_native_sampler(
        patch_d3,
        num_rounds=num_rounds,
        noise=NoiseModel(),
        basis="Z",
        twirl=twirl,
    )
    assert sampler.pauli_frame_lookup is not None

    raw_events, raw_obs = sampler.sample(num_shots, seed=99, pauli_masks=masks)
    assert raw_events.any() or raw_obs.any()

    events, observables = demask_pauli_frame_records(
        sampler.pauli_frame_lookup,
        raw_events,
        raw_obs,
        masks,
    )
    assert not events.any()
    assert not observables.any()
    assert decode_native_samples(sampler, num_shots, seed=99, pauli_masks=masks) == 0


def test_d5_compile_smoke(patch_d3: SurfacePatch) -> None:
    del patch_d3
    patch_d5 = SurfacePatch.create(distance=5)
    masks = sample_pauli_masks_from_guppy(
        patch_d5,
        num_rounds=3,
        num_shots=2,
        basis="Z",
        twirl=TwirlConfig(),
        rng=GuppyRngMaskConfig(seed=11),
    )

    num_data = patch_d5.geometry.num_data
    assert masks.shape == (2, num_pauli_sites(3, num_data))
    assert masks.dtype == np.uint8
    assert masks.min() >= 0
    assert masks.max() <= 3
