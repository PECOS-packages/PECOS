# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Bit-for-bit migration guard for :mod:`pecos.qec.surface.decode`.

The checked-in fixture was generated against the pre-migration NumPy
implementation with::

    uv run --frozen python \
      python/quantum-pecos/tests/qec/surface/test_decode_numpy_migration_equivalence.py \
      --generate

The generator creates physically consistent, fixed-seed syndrome histories for
distance 3 and 5, X and Z memory, and every ``DecoderType`` member.  It records
the raw syndromes, the exact detector/raw-syndrome payload sent to the decoder,
and the correction and logical prediction returned.  Tests replay those inputs;
they do not regenerate random cases.

The same script provides the migration performance smoke over the largest
native batch-decode case used here::

    uv run --frozen python \
      python/quantum-pecos/tests/qec/surface/test_decode_numpy_migration_equivalence.py \
      --benchmark
"""

from __future__ import annotations

import argparse
import builtins
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pecos import array, dtypes
from pecos.qec.surface import NoiseParameters, SurfaceDecoder, SurfacePatch, syndromes_to_detection_events
from pecos.qec.surface.decode import (
    DEM_DECODER_TYPES,
    DecoderType,
    build_native_sampler,
    decode_native_samples,
    generate_circuit_level_dem,
    generate_dem_from_patch,
)

_FIXTURE_PATH = Path(__file__).with_name("fixtures") / "decode_numpy_migration_equivalence.json"
_SEED = 458_004
_NOISE = NoiseParameters.uniform(0.01)
_DEM_MODES = {
    DecoderType.PYMATCHING.value: "native_terminal_graphlike",
    DecoderType.PYMATCHING_CORRELATED.value: "native_terminal_graphlike",
    DecoderType.PYMATCHING_UNCORRELATED.value: "native_terminal_graphlike",
    DecoderType.TESSERACT.value: "native_decomposed",
}


def _dense_check_matrix(patch: SurfacePatch, stabilizer_type: str) -> np.ndarray:
    stabilizers = patch.geometry.x_stabilizers if stabilizer_type == "X" else patch.geometry.z_stabilizers
    check = np.zeros((len(stabilizers), patch.num_data), dtype=np.uint8)
    for stabilizer in stabilizers:
        check[stabilizer.index, list(stabilizer.data_qubits)] = 1
    return check


def _generate_inputs(distance: int, basis: str, rounds: int) -> dict[str, Any]:
    """Generate a sparse, physically consistent syndrome history."""
    patch = SurfacePatch.create(distance=distance)
    rng = np.random.default_rng(_SEED + distance * 100 + ord(basis))
    x_check = _dense_check_matrix(patch, "X")
    z_check = _dense_check_matrix(patch, "Z")
    x_frame = np.zeros(patch.num_data, dtype=np.uint8)
    z_frame = np.zeros(patch.num_data, dtype=np.uint8)
    init_synx = np.zeros(x_check.shape[0], dtype=np.uint8)
    init_synz = np.zeros(z_check.shape[0], dtype=np.uint8)
    init_synx[int(rng.integers(x_check.shape[0]))] = 1
    init_synz[int(rng.integers(z_check.shape[0]))] = 1
    x_error_qubit = int(rng.integers(patch.num_data))
    z_error_qubit = int(rng.integers(patch.num_data))
    x_error_round = int(rng.integers(rounds))
    z_error_round = int(rng.integers(rounds))
    synx_list: list[list[int]] = []
    synz_list: list[list[int]] = []

    for round_index in range(rounds):
        if round_index == x_error_round:
            x_frame[x_error_qubit] ^= 1
        if round_index == z_error_round:
            z_frame[z_error_qubit] ^= 1
        synx = (x_check @ z_frame) % 2
        synz = (z_check @ x_frame) % 2
        if basis == "Z":
            synx ^= init_synx
        else:
            synz ^= init_synz
        if rounds > 1 and round_index == 0:
            synx[int(rng.integers(synx.shape[0]))] ^= 1
            synz[int(rng.integers(synz.shape[0]))] ^= 1
        synx_list.append(synx.tolist())
        synz_list.append(synz.tolist())

    final = x_frame.copy() if basis == "Z" else z_frame.copy()
    return {
        "synx": synx_list,
        "synz": synz_list,
        "final": final.tolist(),
        "init": (init_synx if basis == "Z" else init_synz).tolist(),
    }


def _zero_inputs(distance: int, basis: str, rounds: int = 1) -> dict[str, Any]:
    patch = SurfacePatch.create(distance=distance)
    num_x = len(patch.geometry.x_stabilizers)
    num_z = len(patch.geometry.z_stabilizers)
    return {
        "synx": [[0] * num_x for _round in range(rounds)],
        "synz": [[0] * num_z for _round in range(rounds)],
        "final": [0] * patch.num_data,
        "init": [0] * (num_z if basis == "X" else num_x),
    }


def _single_boundary_error_inputs(distance: int, basis: str, rounds: int) -> dict[str, Any]:
    patch = SurfacePatch.create(distance=distance)
    x_check = _dense_check_matrix(patch, "X")
    z_check = _dense_check_matrix(patch, "Z")
    x_error_qubit = int(np.flatnonzero(z_check.sum(axis=0) == 1)[0])
    z_error_qubit = int(np.flatnonzero(x_check.sum(axis=0) == 1)[0])
    final = [0] * patch.num_data
    final[x_error_qubit if basis == "Z" else z_error_qubit] = 1
    return {
        "synx": [x_check[:, z_error_qubit].tolist() for _round in range(rounds)],
        "synz": [z_check[:, x_error_qubit].tolist() for _round in range(rounds)],
        "final": final,
        "init": [0] * (z_check.shape[0] if basis == "X" else x_check.shape[0]),
    }


def _decoder_for_case(case: dict[str, Any]) -> SurfaceDecoder:
    decoder_type = case["backend"]
    use_dem = DecoderType(decoder_type) in DEM_DECODER_TYPES
    return SurfaceDecoder(
        SurfacePatch.create(distance=case["distance"]),
        num_rounds=case["rounds"],
        noise=_NOISE,
        decoder_type=decoder_type,
        use_circuit_level_dem=use_dem,
        circuit_level_dem_mode=_DEM_MODES.get(decoder_type, "native_full"),
    )


def _numpy_dem_detection_events(
    case: dict[str, Any],
    decoder: SurfaceDecoder,
    synx: list[np.ndarray],
    synz: list[np.ndarray],
    final: np.ndarray,
    init: np.ndarray,
) -> np.ndarray:
    """Compute the exact DEM payload independently with the NumPy oracle."""
    events: list[int] = []
    if case["basis"] == "Z":
        events.extend(np.not_equal(synx[0], init).tolist())
        for round_index in range(1, case["rounds"]):
            events.extend(np.not_equal(synx[round_index], synx[round_index - 1]).tolist())
        events.extend(synz[0].tolist())
        for round_index in range(1, case["rounds"]):
            events.extend(np.not_equal(synz[round_index], synz[round_index - 1]).tolist())
        for stabilizer in decoder.patch.geometry.z_stabilizers:
            parity = sum(int(final[qubit]) for qubit in stabilizer.data_qubits) % 2
            events.append(parity ^ int(synz[-1][stabilizer.index]))
    else:
        events.extend(synx[0].tolist())
        for round_index in range(1, case["rounds"]):
            events.extend(np.not_equal(synx[round_index], synx[round_index - 1]).tolist())
        events.extend(np.not_equal(synz[0], init).tolist())
        for round_index in range(1, case["rounds"]):
            events.extend(np.not_equal(synz[round_index], synz[round_index - 1]).tolist())
        for stabilizer in decoder.patch.geometry.x_stabilizers:
            parity = sum(int(final[qubit]) for qubit in stabilizer.data_qubits) % 2
            events.append(parity ^ int(synx[-1][stabilizer.index]))
    return np.asarray(events, dtype=np.uint8)


def _capture_case(case: dict[str, Any]) -> dict[str, Any]:
    decoder = _decoder_for_case(case)
    inputs = case["inputs"]
    synx = [np.asarray(row, dtype=np.uint8) for row in inputs["synx"]]
    synz = [np.asarray(row, dtype=np.uint8) for row in inputs["synz"]]
    final = np.asarray(inputs["final"], dtype=np.uint8)
    init = np.asarray(inputs["init"], dtype=np.uint8)
    decoder_type = DecoderType(case["backend"])

    if decoder_type in DEM_DECODER_TYPES:
        detector_events = _numpy_dem_detection_events(case, decoder, synx, synz, final, init)
        decoder_input = {"detection_events": np.asarray(detector_events).tolist()}
    elif case["basis"] == "Z":
        detector_events = syndromes_to_detection_events(
            np.asarray(synz, dtype=np.uint8),
            case["rounds"],
            len(decoder.patch.geometry.z_stabilizers),
        )
        decoder_input = {
            "detection_events": np.asarray(detector_events).tolist(),
            "raw_syndrome": synz[-1].tolist(),
        }
    else:
        detector_events = syndromes_to_detection_events(
            np.asarray(synx, dtype=np.uint8),
            case["rounds"],
            len(decoder.patch.geometry.x_stabilizers),
        )
        decoder_input = {
            "detection_events": np.asarray(detector_events).tolist(),
            "raw_syndrome": synx[-1].tolist(),
        }

    if case["basis"] == "Z":
        is_error, result = decoder.decode_memory_z(synx, synz, final, init_synx=init)
    else:
        is_error, result = decoder.decode_memory_x(synx, synz, final, init_synz=init)

    return {
        "decoder_input": decoder_input,
        "output": {
            "is_logical_error": bool(is_error),
            "x_correction": np.asarray(result.x_correction).tolist(),
            "z_correction": np.asarray(result.z_correction).tolist(),
            "logical_x_flip": bool(result.logical_x_flip),
            "logical_z_flip": bool(result.logical_z_flip),
            "decoding_weight_hex": float(result.decoding_weight).hex(),
        },
    }


def _generate_fixture() -> dict[str, Any]:
    cases = []
    for distance in (3, 5):
        for basis in ("X", "Z"):
            for decoder_type in DecoderType:
                # FusionBlossom's supported single-round constructor avoids its
                # pre-existing duplicate-edge panic in the multi-round graph.
                rounds = 1 if decoder_type is DecoderType.FUSION_BLOSSOM else distance
                inputs = (
                    _zero_inputs(distance, basis, rounds)
                    if decoder_type is DecoderType.FUSION_BLOSSOM
                    else (
                        (
                            _zero_inputs(distance, basis, rounds)
                            if basis == "Z"
                            else _single_boundary_error_inputs(distance, basis, rounds)
                        )
                        if decoder_type is DecoderType.BP_LSD
                        else _generate_inputs(distance, basis, rounds)
                    )
                )
                case = {
                    "id": f"d{distance}-{basis.lower()}-{decoder_type.value}",
                    "distance": distance,
                    "rounds": rounds,
                    "basis": basis,
                    "backend": decoder_type.value,
                    "inputs": inputs,
                }
                case.update(_capture_case(case))
                cases.append(case)
    return {
        "schema": 1,
        "seed": _SEED,
        "numpy_version": np.__version__,
        "backends": [decoder_type.value for decoder_type in DecoderType],
        "cases": cases,
    }


def _load_fixture() -> dict[str, Any]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


_FIXTURE = _load_fixture() if _FIXTURE_PATH.is_file() else {"cases": []}


def test_fixture_enumerates_every_supported_backend() -> None:
    assert _FIXTURE["backends"] == [decoder_type.value for decoder_type in DecoderType]
    assert len(_FIXTURE["cases"]) == 2 * 2 * len(DecoderType)


def test_tesseract_check_matrix_all_one_syndromes_match_pre_migration() -> None:
    """Pin the valid mixed-dtype array_equal path reported during review."""
    patch = SurfacePatch.create(distance=3)
    decoder = SurfaceDecoder(
        patch,
        num_rounds=3,
        decoder_type="tesseract",
        use_circuit_level_dem=False,
    )
    synx = [np.ones(len(patch.geometry.x_stabilizers), dtype=np.uint8) for _round in range(3)]
    synz = [np.ones(len(patch.geometry.z_stabilizers), dtype=np.uint8) for _round in range(3)]
    final = np.ones(patch.num_data, dtype=np.uint8)

    is_error, result = decoder.decode_memory_x(synx, synz, final)

    assert is_error
    assert result.x_correction.tolist() == [0] * patch.num_data
    assert result.z_correction.tolist() == [0] * patch.num_data
    assert not result.logical_x_flip
    assert not result.logical_z_flip
    assert float(result.decoding_weight).hex() == "0x1.2616719161d2bp+3"


def test_syndrome_conversion_valid_bits_are_unchanged() -> None:
    syndromes = np.asarray([[1, 0], [1, 1], [0, 1]], dtype=np.uint8)
    events = syndromes_to_detection_events(syndromes, num_rounds=3, num_detectors_per_round=2)
    assert events.tolist() == [[1, 0], [0, 1], [1, 0]]


@pytest.mark.parametrize(
    ("bad_value", "position"),
    [(2, (0, 1)), (255, (2, 0))],
)
def test_syndrome_conversion_rejects_non_bits(bad_value: int, position: tuple[int, int]) -> None:
    syndromes = np.zeros((3, 2), dtype=np.uint8)
    syndromes[position] = bad_value

    with pytest.raises(ValueError, match=rf"found {bad_value} at position") as exc_info:
        syndromes_to_detection_events(syndromes, num_rounds=3, num_detectors_per_round=2)

    assert str(exc_info.value) == (f"syndromes must contain only 0/1 bits; found {bad_value} at position {position}")


@pytest.mark.parametrize("method_name", ["decode_memory_x", "decode_memory_z"])
@pytest.mark.parametrize("syndrome_name", ["synx_list", "synz_list"])
@pytest.mark.parametrize("bad_value", [2, 255])
@pytest.mark.parametrize("round_index", range(3))
def test_memory_decoders_reject_non_bits_in_any_round(
    method_name: str,
    syndrome_name: str,
    bad_value: int,
    round_index: int,
) -> None:
    patch = SurfacePatch.create(distance=3)
    decoder = SurfaceDecoder(patch, num_rounds=3, decoder_type="pymatching", use_circuit_level_dem=False)
    synx = [np.zeros(len(patch.geometry.x_stabilizers), dtype=np.uint8) for _round in range(3)]
    synz = [np.zeros(len(patch.geometry.z_stabilizers), dtype=np.uint8) for _round in range(3)]
    target = synx if syndrome_name == "synx_list" else synz
    target[round_index][1] = bad_value

    with pytest.raises(ValueError, match=rf"found {bad_value} at position") as exc_info:
        getattr(decoder, method_name)(synx, synz, np.zeros(patch.num_data, dtype=np.uint8))

    assert str(exc_info.value) == (
        f"{syndrome_name} must contain only 0/1 bits; found {bad_value} at position ({round_index}, 1)"
    )


@pytest.mark.parametrize("case", _FIXTURE["cases"], ids=lambda case: case["id"])
def test_decode_capture_is_bit_for_bit_identical(case: dict[str, Any]) -> None:
    assert _capture_case(case) == {
        "decoder_input": case["decoder_input"],
        "output": case["output"],
    }


def test_native_decoder_path_does_not_import_stim(monkeypatch: pytest.MonkeyPatch) -> None:
    """Core decoding remains usable when the optional Stim extra is absent."""
    case = next(case for case in _FIXTURE["cases"] if case["id"] == "d3-z-pymatching")
    real_import = builtins.__import__

    def reject_stim(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "stim" or name.startswith("stim."):
            message = "Stim deliberately hidden by equivalence harness"
            raise ModuleNotFoundError(message)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_stim)
    assert _capture_case(case)["output"] == case["output"]


def test_stim_extra_interoperability_boundaries() -> None:
    """Exercise both lazy Stim helpers and an outbound PECOS Array target buffer."""
    stim = pytest.importorskip("stim")
    patch = SurfacePatch.create(distance=3)
    for basis in ("X", "Z"):
        generated = generate_circuit_level_dem(3, 3, _NOISE, basis)
        from_patch = generate_dem_from_patch(patch, 3, _NOISE, basis)
        assert stim.DetectorErrorModel(generated).num_observables == 1
        assert stim.DetectorErrorModel(from_patch).num_observables == 1

    targets = array([0, 2], dtype=dtypes.uint64)
    circuit = stim.Circuit()
    circuit.append("X", targets)
    assert str(circuit) == "X 0 2"


def _benchmark_native_batch() -> dict[str, Any]:
    patch = SurfacePatch.create(distance=5)
    sampler = build_native_sampler(patch, 5, _NOISE, basis="Z")
    shots = 8192
    repeats = 5
    decode_native_samples(sampler, 64, seed=_SEED)
    durations = []
    logical_errors = None
    for _repeat in range(repeats):
        started = time.perf_counter()
        count = decode_native_samples(sampler, shots, seed=_SEED)
        durations.append(time.perf_counter() - started)
        if logical_errors is None:
            logical_errors = count
        else:
            assert count == logical_errors
    return {
        "case": "d5-z-pymatching-native-batch",
        "shots": shots,
        "repeats": repeats,
        "logical_errors": logical_errors,
        "durations_seconds": durations,
        "median_seconds": statistics.median(durations),
    }


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    if args.generate:
        fixture = _generate_fixture()
        _FIXTURE_PATH.write_text(json.dumps(fixture, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(fixture, indent=2))
    if args.benchmark:
        print(json.dumps(_benchmark_native_batch(), indent=2))
    if not args.generate and not args.benchmark:
        parser.error("pass --generate or --benchmark")


if __name__ == "__main__":
    _main()
