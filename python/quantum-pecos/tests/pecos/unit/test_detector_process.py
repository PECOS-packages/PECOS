# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
"""Independent physical and serialization checks for quantum detector processes."""

import json
from dataclasses import replace

import numpy as np
import pecos as pc
import pytest
from pecos.experimental.detector_models.process import (
    ChannelFactor,
    DetectorDef,
    DetectorInstrument,
    DetectorProcess,
    InstrumentFactor,
    KrausMap,
    SystemSpec,
    compile_instrument,
)


def _numpy(value):
    return np.asarray(value.tolist() if hasattr(value, "tolist") else value)


def _qubit(name="q"):
    return SystemSpec(name, ("0", "1"), {"computational": (0, 1)})


def _four_level():
    return SystemSpec("q", ("0", "1", "L", "vac"), {"computational": (0, 1), "leakage": (2,), "vacancy": (3,)})


def _projector(index, dimension=4):
    result = np.zeros((dimension, dimension), complex)
    result[index, index] = 1
    return result


def _rotation(theta):
    result = np.eye(4, dtype=complex)
    result[1:3, 1:3] = [[np.cos(theta / 2), -1j * np.sin(theta / 2)], [-1j * np.sin(theta / 2), np.cos(theta / 2)]]
    return result


def _readout():
    return InstrumentFactor(
        "readout",
        ("q",),
        {"0": [_projector(0)], "1": [_projector(1)], "no_readout": [_projector(2) + _projector(3)]},
        "m",
    )


def _observed_process(theta=0.6):
    return DetectorProcess(
        (_four_level(),),
        (ChannelFactor("rotation", ("q",), [_rotation(theta)]), _readout()),
        (DetectorDef("D0", ("m",)),),
        (),
    )


def test_hidden_coherent_leakage_preserves_phase_without_creating_records():
    theta = 0.7
    process = DetectorProcess((_four_level(),), (ChannelFactor("rotation", ("q",), [_rotation(theta)]),))
    compiled = compile_instrument(process)
    expected = np.array([0, np.cos(theta / 2), -1j * np.sin(theta / 2), 0])
    assert set(compiled.branches) == {()}
    np.testing.assert_allclose(_numpy(compiled.total(_projector(1))), np.outer(expected, expected.conj()), atol=1e-13)
    assert abs(_numpy(compiled.total(_projector(1)))[1, 2].imag) > 0


def test_inverse_coherent_pulses_cancel_while_repeated_pulses_interfere():
    theta = 0.7
    first = ChannelFactor("first", ("q",), [_rotation(theta)])
    cancelled = compile_instrument(
        DetectorProcess((_four_level(),), (first, ChannelFactor("inverse", ("q",), [_rotation(-theta)]))),
    )
    doubled = compile_instrument(
        DetectorProcess((_four_level(),), (first, ChannelFactor("again", ("q",), [_rotation(theta)]))),
    )
    np.testing.assert_allclose(_numpy(cancelled.total(_projector(1))), _projector(1), atol=1e-13)
    assert _numpy(doubled.total(_projector(1)))[2, 2].real == pytest.approx(np.sin(theta) ** 2)


def test_environment_kraus_alternatives_are_summed_as_maps_not_amplitudes():
    p = 0.3
    channel = ChannelFactor("hidden_dephasing", ("q",), [np.sqrt(p) * np.eye(2), np.sqrt(1 - p) * np.diag([1, -1])])
    compiled = compile_instrument(DetectorProcess((_qubit(),), (channel,)))
    plus = np.ones((2, 2)) / 2
    expected = p * plus + (1 - p) * np.diag([1, -1]) @ plus @ np.diag([1, -1])
    assert set(compiled.branches) == {()}
    np.testing.assert_allclose(_numpy(compiled.total(plus)), expected)


def test_loss_persists_until_an_explicit_refill_channel():
    loss = [_projector(3)]
    for source in range(3):
        jump = np.zeros((4, 4), complex)
        jump[3, source] = 1
        loss.append(jump)
    x_gate = np.eye(4, dtype=complex)
    x_gate[:2, :2] = [[0, 1], [1, 0]]
    factors = (ChannelFactor("loss", ("q",), loss), ChannelFactor("ordinary_x", ("q",), [x_gate]))
    lost = compile_instrument(DetectorProcess((_four_level(),), factors))
    np.testing.assert_allclose(_numpy(lost.total(_projector(1))), _projector(3))
    refill = np.zeros((4, 4), complex)
    refill[0, 3] = 1
    present = np.diag([1, 1, 1, 0])
    replaced = compile_instrument(
        DetectorProcess((_four_level(),), (*factors, ChannelFactor("refill", ("q",), [present, refill]))),
    )
    np.testing.assert_allclose(_numpy(replaced.total(_projector(1))), _projector(0))
    assert set(lost.branches) == {()}


def test_ordinary_unavailable_outcome_does_not_identify_leakage_versus_loss():
    compiled = compile_instrument(DetectorProcess((_four_level(),), (_readout(),), (DetectorDef("D0", ("m",)),), ()))
    vector = np.array([0, 0, 1, 1j]) / np.sqrt(2)
    rho = np.outer(vector, vector.conj())
    assert set(compiled.branches) == {(0,), (1,), (None,)}
    np.testing.assert_allclose(_numpy(compiled.apply(rho)[(None,)]), rho)
    assert abs(_numpy(compiled.apply(rho)[(None,)])[2, 3]) > 0


def test_default_retains_only_declared_records_and_explicit_empty_discards_them():
    p0, p1 = _projector(0, 2), _projector(1, 2)
    readout = InstrumentFactor("readout", ("q",), {"0": [p0], "1": [p1]}, "m")
    process = DetectorProcess((_qubit(),), (readout,))
    retained, discarded = compile_instrument(process), compile_instrument(replace(process, retained_records=()))
    assert set(retained.branches) == {("0",), ("1",)}
    assert set(discarded.branches) == {()}
    plus = np.ones((2, 2)) / 2
    np.testing.assert_allclose(_numpy(discarded.total(plus)), np.eye(2) / 2)
    np.testing.assert_allclose(_numpy(retained.total(plus)), _numpy(discarded.total(plus)))


def test_binary_leakage_readout_is_a_physical_instrument_assumption():
    p0, p1, leaked = _projector(0), _projector(1), _projector(2)
    readout = InstrumentFactor(
        "binary_readout",
        ("q",),
        {"0": [p0, leaked / np.sqrt(2)], "1": [p1, leaked / np.sqrt(2)], "no_readout": [_projector(3)]},
        "m",
    )
    instrument = compile_instrument(DetectorProcess((_four_level(),), (readout,), (DetectorDef("D", ("m",)),), ()))
    for key in ((0,), (1,)):
        assert np.trace(_numpy(instrument.apply(leaked)[key])).real == pytest.approx(0.5)
    assert np.trace(_numpy(instrument.apply(leaked)[(None,)])).real == pytest.approx(0)


def test_ordered_nonadjacent_targets_with_heterogeneous_dimensions():
    middle = SystemSpec("b", ("0", "1", "2"), {"all": (0, 1, 2)})
    cx = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    process = DetectorProcess((_qubit("a"), middle, _qubit("c")), (ChannelFactor("cx_c_a", ("c", "a"), [cx]),))
    np.testing.assert_allclose(_numpy(compile_instrument(process).total(_projector(11, 12))), _projector(5, 12))


def test_rectangular_encoding_retains_full_physical_output():
    encoding = np.zeros((4, 2), complex)
    encoding[0, 0], encoding[3, 1] = 1, 1
    process = DetectorProcess((_four_level(),), (), input_isometry=encoding)
    instrument = compile_instrument(process)
    rho = np.array([[0.5, -0.5j], [0.5j, 0.5]])
    assert instrument.input_dimension == 2
    assert instrument.output_dimension == 4
    np.testing.assert_allclose(_numpy(instrument.total(rho)), encoding @ rho @ encoding.conj().T)


@pytest.mark.parametrize("dimension", [2, 3, 8])
def test_trace_nonincreasing_check_matches_independent_complex_psd_oracle(dimension):
    rng = np.random.default_rng(1234 + dimension)
    unitary, _ = np.linalg.qr(rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension)))
    for singular_values in (np.linspace(0, 1, dimension), np.ones(dimension), np.linspace(0.2, 1.2, dimension)):
        operator = np.diag(singular_values) @ unitary
        eigenvalues = np.linalg.eigvalsh(operator.conj().T @ operator)
        if eigenvalues[-1] <= 1 + 1e-12:
            KrausMap([operator])
        else:
            with pytest.raises(ValueError, match="nonincreasing"):
                KrausMap([operator])


def test_trace_nonincreasing_rejects_offdiagonal_effect_violation():
    operator = np.array([[0.8, 0.8], [0, 0]])
    assert np.max(np.diag(operator.conj().T @ operator)) < 1
    with pytest.raises(ValueError, match="nonincreasing"):
        KrausMap([operator])


def test_local_and_compiled_json_roundtrips_preserve_complex_action(tmp_path):
    process = _observed_process()
    source_path, compiled_path = tmp_path / "process.json", tmp_path / "instrument.json"
    manifest = process.save(source_path)
    assert manifest["complex_encoding"] == "shape and flattened real/imaginary pairs"
    assert not list(tmp_path.glob("*.npz"))
    restored = DetectorProcess.load(source_path)
    original, compiled = compile_instrument(process), compile_instrument(restored)
    compiled.save(compiled_path)
    reloaded = DetectorInstrument.load(compiled_path)
    for row in range(4):
        for column in range(4):
            matrix_unit = np.zeros((4, 4), complex)
            matrix_unit[row, column] = 1
            for key in original.branches:
                np.testing.assert_allclose(
                    _numpy(original.branches[key].apply(matrix_unit)),
                    _numpy(reloaded.branches[key].apply(matrix_unit)),
                    atol=1e-14,
                )
    with pytest.raises(FileExistsError):
        restored.save(source_path)


def test_source_roundtrip_preserves_declared_logical_encoding(tmp_path):
    encoding = np.zeros((4, 2), complex)
    encoding[0, 0], encoding[1, 1] = 1, 1
    path = tmp_path / "encoded.json"
    process = replace(_observed_process(), input_isometry=encoding)
    process.save(path)
    restored = DetectorProcess.load(path)
    assert compile_instrument(restored).input_dimension == 2
    np.testing.assert_array_equal(_numpy(restored.input_isometry), encoding)


@pytest.mark.parametrize("compiled", [False, True])
def test_serialized_array_tampering_is_rejected(tmp_path, compiled):
    path = tmp_path / "model.json"
    value = compile_instrument(_observed_process()) if compiled else _observed_process()
    value.save(path)
    payload = json.loads(path.read_text())
    first = next(iter(payload["arrays"].values()))
    first["values"][0][0] += 0.1
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash"):
        type(value).load(path)


def test_serialized_tensor_convention_is_checked(tmp_path):
    path = tmp_path / "model.json"
    _observed_process().save(path)
    payload = json.loads(path.read_text())
    payload["tensor_convention"] = "least-significant first"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="tensor"):
        DetectorProcess.load(path)


def test_unavailable_detector_and_unspecified_reference_are_distinct():
    diagnostic = DetectorDef("raw", ("m",), expected_parity=None)
    relative = DetectorDef("reference_one", ("m",), expected_parity=1)
    assert diagnostic.evaluate({"m": "1"}) == 1
    assert relative.evaluate({"m": "1"}) == 0
    assert relative.evaluate({"m": "no_readout"}) is None


def test_reference_compiler_limits_fail_explicitly():
    with pytest.raises(ValueError, match="dimension"):
        compile_instrument(DetectorProcess((_four_level(),), ()), max_dimension=2)
    with pytest.raises(ValueError, match="path"):
        compile_instrument(_observed_process(), max_paths=2)
    with pytest.raises(ValueError, match="positive integer"):
        compile_instrument(_observed_process(), max_paths=True)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), complex(0, float("inf"))])
def test_nonfinite_operator_entries_are_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        KrausMap([[[value]]])


def test_missing_outcome_is_not_silently_renormalized():
    process = DetectorProcess((_qubit(),), (InstrumentFactor("bad", ("q",), {"0": [_projector(0, 2)]}, "m"),))
    with pytest.raises(ValueError, match="preserve trace"):
        compile_instrument(process)


@pytest.mark.parametrize("change", ["duplicate_system", "unknown_record", "sector_gap", "bad_reference", "bad_target"])
def test_declaration_errors_fail_before_compilation(change):
    process = _observed_process()
    if change == "duplicate_system":
        process = replace(process, systems=(*process.systems, process.systems[0]))
    elif change == "unknown_record":
        process = replace(process, detectors=(DetectorDef("bad", ("hidden_kraus_index",)),))
    elif change == "sector_gap":
        process = replace(process, systems=(replace(_four_level(), sectors={"computational": (0, 1)}),))
    elif change == "bad_reference":
        process = replace(process, detectors=(DetectorDef("bad", ("m",), expected_parity=2),))
    else:
        process = replace(process, factors=(ChannelFactor("bad", ("absent",), [np.eye(4)]),))
    with pytest.raises(ValueError, match=r"duplicate|undefined|partition|reference|targets"):
        compile_instrument(process)


def test_public_outputs_use_pecos_arrays_and_accept_nested_lists():
    instrument = compile_instrument(
        DetectorProcess((_qubit(),), (ChannelFactor("identity", ("q",), [[[1, 0], [0, 1]]]),)),
    )
    result = instrument.total([[1, 0], [0, 0]])
    assert isinstance(result, pc.Array)
    np.testing.assert_array_equal(_numpy(result), [[1, 0], [0, 0]])


def test_kraus_map_snapshots_input_and_defends_validated_operators():
    original = np.eye(2, dtype=complex)[None]
    channel = KrausMap(operators=original)
    original[0, 0, 0] = 2
    exposed = channel.operators
    exposed[0, 0, 0] = 3
    np.testing.assert_array_equal(_numpy(channel.apply(_projector(0, 2))), _projector(0, 2))


def test_instrument_snapshots_branch_mapping_and_prevents_inplace_mutation():
    branches = {(): KrausMap([np.eye(2)])}
    instrument = DetectorInstrument(branches)
    branches.clear()
    assert set(instrument.branches) == {()}
    with pytest.raises(TypeError):
        instrument.branches[()] = KrausMap([np.eye(2) * 0.5])
