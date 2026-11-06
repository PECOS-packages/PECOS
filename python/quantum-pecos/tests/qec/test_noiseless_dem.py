# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Gate-batch noiseless metadata reaches the native DEM source boundary."""

import pytest
from pecos_rslib.qec import DetectorErrorModel
from pecos_rslib.quantum import TickCircuit

_NOISE = {"p1": 0.03, "p2": 0.06, "p_meas": 0.07, "p_prep": 0.08}


def _circuit() -> TickCircuit:
    circuit = TickCircuit()
    circuit.tick().pz([0, 1])
    circuit.tick().h([0, 1])
    circuit.tick().h([0, 1])
    circuit.tick().cx([(0, 1)])
    circuit.tick().mz([0, 1])
    circuit.set_meta("detectors", '[{"id":0,"records":[-2]},{"id":1,"records":[-1]}]')
    circuit.set_meta("observables", '[{"id":0,"records":[-1]}]')
    return circuit


def _mark_tick(circuit: TickCircuit, index: int, value: object) -> None:
    # get_tick() returns a copy. The in-place setter calls Rust Tick::set_gate_attr.
    tick = circuit.get_tick(index)
    for gate_index in range(len(tick)):
        circuit.set_gate_meta(index, gate_index, "noiseless", value)


def test_noiseless_tick_and_fully_noiseless_circuit() -> None:
    circuit = _circuit()
    baseline = DetectorErrorModel.from_circuit(circuit, **_NOISE)
    _mark_tick(circuit, 3, True)
    marked = DetectorErrorModel.from_circuit(circuit, **_NOISE)
    expected = DetectorErrorModel.from_circuit(_circuit(), **(_NOISE | {"p2": 0.0}))
    assert _mechanisms(marked.to_string()) != _mechanisms(baseline.to_string())
    _assert_mechanisms(marked.to_string(), expected.to_string())

    for index in range(circuit.num_ticks()):
        _mark_tick(circuit, index, True)
    empty = DetectorErrorModel.from_circuit(circuit, **_NOISE)
    assert _mechanisms(empty.to_string()) == {}
    assert "error(" not in empty.to_string()
    assert empty.num_detectors == baseline.num_detectors
    assert empty.num_observables == baseline.num_observables


@pytest.mark.parametrize("value", [1, 0.0, "true", []])
def test_noiseless_requires_boolean(value: object) -> None:
    circuit = _circuit()
    _mark_tick(circuit, 1, value)
    with pytest.raises(ValueError, match=r"noiseless.*gate H at tick 1 gate 0 must be a boolean"):
        DetectorErrorModel.from_circuit(circuit, **_NOISE)


def test_autodetected_dem_honours_noiseless() -> None:
    from pecos.qec.surface.circuit_builder import generate_dem_from_tick_circuit_via_autodetection

    circuit = _circuit()
    baseline = generate_dem_from_tick_circuit_via_autodetection(circuit, **_NOISE)
    assert "error(" in baseline
    for index in range(circuit.num_ticks()):
        _mark_tick(circuit, index, True)
    empty = generate_dem_from_tick_circuit_via_autodetection(circuit, **_NOISE)
    assert "error(" not in empty
    assert empty.splitlines() == [line for line in baseline.splitlines() if not line.startswith("error(")]
    _mark_tick(circuit, 1, "true")
    with pytest.raises(ValueError, match=r"noiseless.*gate H at tick 1 gate 0.*must be a boolean"):
        generate_dem_from_tick_circuit_via_autodetection(circuit, **_NOISE)


def test_noiseless_csr_export_preserves_location_numbering() -> None:
    from pecos_rslib.qec import DagFaultAnalyzer

    circuit = _circuit()
    baseline = DagFaultAnalyzer(circuit.to_dag_circuit()).build_influence_map().export_csr()
    assert baseline["detector_data_x"]
    for index in range(circuit.num_ticks()):
        _mark_tick(circuit, index, True)
    empty = DagFaultAnalyzer(circuit.to_dag_circuit()).build_influence_map().export_csr()
    assert empty["num_locations"] == baseline["num_locations"]
    assert empty["num_detectors"] == baseline["num_detectors"]
    for family in ("detector", "dem_output"):
        for pauli in "xyz":
            assert not empty[f"{family}_data_{pauli}"]
            assert empty[f"{family}_offsets_{pauli}"] == [0] * (empty["num_locations"] + 1)
    _mark_tick(circuit, 1, 1)
    malformed = DagFaultAnalyzer(circuit.to_dag_circuit()).build_influence_map()
    with pytest.raises(ValueError, match=r"noiseless.*H.*must be a boolean"):
        malformed.export_csr()


def _mechanisms(text: str) -> dict[tuple[str, ...], float]:
    import stim

    return {
        tuple(str(target) for target in instruction.targets_copy()): instruction.args_copy()[0]
        for instruction in stim.DetectorErrorModel(text).flattened()
        if instruction.type == "error"
    }


def _assert_mechanisms(actual: str, expected: str) -> None:
    assert _mechanisms(actual) == pytest.approx(_mechanisms(expected))


def test_noiseless_stim_native_parity() -> None:
    from pecos.qec.surface.circuit_builder import generate_dem_from_tick_circuit_via_stim, tick_circuit_to_stim

    # Split measurement into two batches in separate ticks to mark just qubit 0.
    circuit = TickCircuit()
    circuit.tick().pz([0, 1])
    circuit.tick().cx([(0, 1)])
    circuit.tick().mz([0])
    circuit.tick().mz([1])
    circuit.set_meta("detectors", '[{"id":0,"records":[-2]},{"id":1,"records":[-1]}]')
    circuit.set_meta("observables", '[{"id":0,"records":[-1]}]')
    circuit.set_gate_meta(1, 0, "noiseless", True)
    circuit.set_gate_meta(2, 0, "noiseless", True)
    exported = tick_circuit_to_stim(circuit, **_NOISE)
    assert "CX 0 1" in exported
    assert "DEPOLARIZE2" not in exported
    assert f"X_ERROR({_NOISE['p_meas']}) 0" not in exported
    assert f"X_ERROR({_NOISE['p_meas']}) 1" in exported
    native = DetectorErrorModel.from_circuit(circuit, **_NOISE).to_string()
    reference = generate_dem_from_tick_circuit_via_stim(circuit, **_NOISE, decompose_errors=False)
    assert _mechanisms(native)
    _assert_mechanisms(native, reference)


def _mixed_circuit() -> TickCircuit:
    circuit = TickCircuit()
    circuit.tick().pz([0, 1])
    circuit.tick().h([0]).x([1])
    circuit.tick().h([0]).x([1])
    circuit.tick().mz([0, 1])
    circuit.set_meta("detectors", '[{"id":0,"records":[-2],"coords":[0,0,0]},{"id":1,"records":[-1],"coords":[0,0,0]}]')
    circuit.set_meta("observables", '[{"id":0,"records":[-1]}]')
    circuit.set_meta("num_measurements", "2")
    return circuit


def test_noiseless_only_marks_one_batch_in_a_tick() -> None:
    from pecos.qec.surface.circuit_builder import generate_dem_from_tick_circuit_via_stim

    circuit = _mixed_circuit()
    config = {"p1": 0.03, "p2": 0.0, "p_meas": 0.0, "p_prep": 0.0}
    expected = generate_dem_from_tick_circuit_via_stim(circuit, **config, p1_gate_rates={"H": 0.0})
    for tick in (1, 2):
        circuit.set_gate_meta(tick, 0, "noiseless", True)
    actual = DetectorErrorModel.from_circuit(circuit, **config).to_string()
    assert _mechanisms(actual)
    _assert_mechanisms(actual, expected)
    _assert_mechanisms(generate_dem_from_tick_circuit_via_stim(circuit, **config), expected)


def test_noiseless_direct_dag_dem() -> None:
    circuit = _circuit()
    _mark_tick(circuit, 3, True)
    dag = circuit.to_dag_circuit()
    expected = DetectorErrorModel.from_circuit(_circuit(), **(_NOISE | {"p2": 0.0}))
    actual = DetectorErrorModel.from_circuit(dag, **_NOISE)
    _assert_mechanisms(actual.to_string(), expected.to_string())
    _mark_tick(circuit, 3, "true")
    with pytest.raises(ValueError, match=r"noiseless.*CX.*DAG node.*must be a boolean"):
        DetectorErrorModel.from_circuit(circuit.to_dag_circuit(), **_NOISE)


@pytest.mark.parametrize("path", ["stim", "via_stim", "pauli_frame"])
def test_python_noise_paths_validate_and_suppress_batches(path: str) -> None:
    from pecos.qec.surface.circuit_builder import (
        generate_dem_from_tick_circuit_via_pauli_frame,
        generate_dem_from_tick_circuit_via_stim,
        tick_circuit_to_stim,
    )

    build = {
        "stim": tick_circuit_to_stim,
        "via_stim": generate_dem_from_tick_circuit_via_stim,
        "pauli_frame": generate_dem_from_tick_circuit_via_pauli_frame,
    }[path]
    circuit = _mixed_circuit()
    # Active preparation and measurement batches remain; all H batches are suppressed.
    config = {"p1": 0.03, "p2": 0.0, "p_meas": 0.07, "p_prep": 0.08}
    for tick in (1, 2):
        circuit.set_gate_meta(tick, 0, "noiseless", True)
    if path == "pauli_frame":
        with pytest.warns(DeprecationWarning, match="is deprecated"):
            actual = build(circuit, **config)
        with pytest.warns(DeprecationWarning, match="is deprecated"):
            expected = build(_mixed_circuit(), **(config | {"p1": 0.0}))
        # Preserve the legacy simulator's existing gate-batch propagation.
        assert _mechanisms(actual)
        _assert_mechanisms(actual, expected)
    else:
        reference = build(_mixed_circuit(), **config, p1_gate_rates={"H": 0.0})
        actual = build(circuit, **config)
        assert actual == reference
    circuit.set_gate_meta(1, 1, "noiseless", "true")
    with pytest.raises(ValueError, match=r"noiseless.*X.*tick 1.*gate 1.*must be a boolean"):
        build(circuit, **config)


@pytest.mark.parametrize(
    ("method", "args"),
    [
        ("get_locations", ()),
        ("get_location", (0,)),
    ],
)
def test_influence_location_bindings_validate_noiseless(method: str, args: tuple) -> None:
    from pecos_rslib.qec import DagFaultAnalyzer

    circuit = _circuit()
    _mark_tick(circuit, 1, "true")
    influence = DagFaultAnalyzer(circuit.to_dag_circuit()).build_influence_map()
    with pytest.raises(ValueError, match=r"noiseless.*H.*must be a boolean"):
        getattr(influence, method)(*args)


def test_noiseless_tick_error_names_batch() -> None:
    circuit = _mixed_circuit()
    circuit.set_gate_meta(2, 1, "noiseless", 1)
    with pytest.raises(ValueError, match=r"noiseless.*X.*tick 2 gate 1.*must be a boolean"):
        DetectorErrorModel.from_circuit(circuit, **_NOISE)


@pytest.mark.parametrize(("rate", "ticks"), [("p_prep", [0]), ("p1", [1, 2]), ("p2", [3]), ("p_meas", [4])])
def test_stim_export_suppresses_each_noise_kind(rate: str, ticks: list[int]) -> None:
    from pecos.qec.surface.circuit_builder import tick_circuit_to_stim

    circuit = _circuit()
    expected = tick_circuit_to_stim(circuit, **(_NOISE | {rate: 0.0}))
    baseline = tick_circuit_to_stim(circuit, **_NOISE)
    assert baseline != expected
    for tick in ticks:
        _mark_tick(circuit, tick, False)
    assert tick_circuit_to_stim(circuit, **_NOISE) == baseline
    for tick in ticks:
        _mark_tick(circuit, tick, True)
    assert tick_circuit_to_stim(circuit, **_NOISE) == expected


@pytest.mark.parametrize(("rate", "ticks"), [("p_prep", [0]), ("p1", [1, 2]), ("p2", [3]), ("p_meas", [4])])
def test_pauli_frame_suppresses_each_noise_kind(rate: str, ticks: list[int]) -> None:
    from pecos.qec.surface.circuit_builder import generate_dem_from_tick_circuit_via_pauli_frame

    circuit = _circuit()
    circuit.set_meta("num_measurements", "2")
    circuit.set_meta("detectors", '[{"id":0,"records":[-2],"coords":[0,0,0]},{"id":1,"records":[-1],"coords":[1,0,0]}]')
    with pytest.warns(DeprecationWarning, match="is deprecated"):
        expected = generate_dem_from_tick_circuit_via_pauli_frame(circuit, **(_NOISE | {rate: 0.0}))
    with pytest.warns(DeprecationWarning, match="is deprecated"):
        baseline = generate_dem_from_tick_circuit_via_pauli_frame(circuit, **_NOISE)
    assert _mechanisms(baseline) != _mechanisms(expected)
    for tick in ticks:
        _mark_tick(circuit, tick, True)
    with pytest.warns(DeprecationWarning, match="is deprecated"):
        actual = generate_dem_from_tick_circuit_via_pauli_frame(circuit, **_NOISE)
    _assert_mechanisms(actual, expected)


def test_native_wrapper_preserves_marking_and_tick_provenance() -> None:
    from pecos.qec.surface.circuit_builder import generate_dem_from_tick_circuit

    circuit = _circuit()
    _mark_tick(circuit, 3, True)
    native = DetectorErrorModel.from_circuit(circuit, **_NOISE).to_string()
    actual = generate_dem_from_tick_circuit(circuit, **_NOISE, decompose_errors=False)
    _assert_mechanisms(actual, native)
    _mark_tick(circuit, 1, "true")
    with pytest.raises(ValueError, match=r"noiseless.*gate H at tick 1 gate 0 must be a boolean"):
        generate_dem_from_tick_circuit(circuit, **_NOISE)


@pytest.mark.parametrize(("method", "args"), [("classify_fault", (0, 1)), ("get_detector_indices", (0, 1))])
def test_propagation_queries_ignore_noise_metadata(method: str, args: tuple) -> None:
    from pecos_rslib.qec import DagFaultAnalyzer

    circuit = _circuit()
    baseline = DagFaultAnalyzer(circuit.to_dag_circuit()).build_influence_map()
    _mark_tick(circuit, 1, "true")
    marked = DagFaultAnalyzer(circuit.to_dag_circuit()).build_influence_map()
    assert getattr(marked, method)(*args) == getattr(baseline, method)(*args)
