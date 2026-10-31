# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Prepared ideal ticks preserve execution while avoiding per-shot batch checks."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
from functools import partial
from types import SimpleNamespace

import pecos
import pytest
from pecos.analysis import CircuitStatistics
from pecos.circuits import LogicalCircuit, QuantumCircuit
from pecos.circuits.quantum_circuit import TickView
from pecos.engines import hybrid_engine_old
from pecos.noise.error_depolar import DepolarizingErrorModel
from pecos.simulators import CoinToss, PauliProp, PreparedCircuit, SparseStab, SparseStabPy, StateVec, prepare_circuit

STATES = (SparseStabPy, StateVec, SparseStab, CoinToss, PauliProp)


def _state(cls, size=4, shot=0):
    params = {}
    if cls in (StateVec, SparseStab, CoinToss):
        params["seed"] = 42 + shot
    if cls is PauliProp:
        params["track_sign"] = True
    return cls(num_qubits=size, **params)


def _snapshot(state):
    if isinstance(state, SparseStabPy):
        return state.stabs.row_string(), state.destabs.row_string()
    if isinstance(state, SparseStab):
        return state.stab_tableau(), state.destab_tableau()
    if isinstance(state, PauliProp):
        return deepcopy((state.faults, state.sign, state.img))
    if isinstance(state, StateVec):
        return list(state.vector)
    return None


def _circuit():
    circuit = QuantumCircuit(num_qubits=4, cvar_spec={"m": 2})
    circuit.append("init |0>", {0, 1, 2, 3})
    circuit.append("H", {0})
    circuit.append("CX", {(0, 1)})
    circuit.append("measure Z", {0}, var=("m", 0))
    circuit.append("X", {1}, cond={"a": ("m", 0), "op": "==", "b": 1})
    circuit.append("measure Z", {1}, var=("m", 1))
    circuit.append("cop", set(), expr={"t": ("m", 0), "a": ("m", 1), "op": "^", "b": ("m", 0)})
    return circuit


def _shots(cls, circuit, *, prepared, size=4, shots=20, inspector=None, **kwargs):
    engine = pecos.HybridEngine(seed=7)
    state = _state(cls, size)
    program = prepare_circuit(state, circuit) if prepared else circuit
    original_items = deepcopy(tuple(pt.items for pt in program.ticks)) if prepared else None
    results = []
    for i in range(shots):
        if i:
            state = _state(cls, size, i)
        output, _ = engine.run(state, program, shot_id=i, circ_inspector=inspector, **kwargs)
        results.append(({key: str(value) for key, value in output.items()}, _snapshot(state)))
    assert engine.circuit is circuit
    if prepared:
        assert tuple(pt.items for pt in program.ticks) == original_items
    return results


@pytest.mark.parametrize("cls", STATES)
@pytest.mark.parametrize("noise", [False, True])
def test_equivalence(cls, noise):
    circuit = _circuit()

    def run(*, prepared):
        kwargs = {}
        if noise:
            kwargs = {
                "error_gen": DepolarizingErrorModel(),
                "error_params": dict.fromkeys(("p1", "p2", "p_meas", "p_prep"), 0.05),
            }
        return _shots(cls, circuit, prepared=prepared, **kwargs)

    assert run(prepared=False) == run(prepared=True)


def test_replaced_locations_preserve_rng_order():
    locations = {0, 1, 8, 9, 10, 16}
    circuit = QuantumCircuit(num_qubits=17, cvar_spec={"m": 6})
    circuit.append("H", locations)
    circuit.append("measure Z", locations, var_output={q: ("m", i) for i, q in enumerate(sorted(locations))})
    errors = {1: {"replaced": {0, 1}}}
    raw = _shots(SparseStabPy, circuit, prepared=False, size=17, error_circuits=errors)
    assert raw == _shots(SparseStabPy, circuit, prepared=True, size=17, error_circuits=errors)


def _logical():
    qecc = pecos.qeccs.Surface4444(distance=3)
    circuit = LogicalCircuit(suppress_warning=True)
    circuit.append(qecc.gate("ideal init |0>"))
    extraction = qecc.gate("I", num_syn_extract=1)
    circuit.append(extraction)
    return qecc, circuit, extraction


def test_logical_circuit():
    qecc, circuit, _ = _logical()
    time = next(time for _, time, params in circuit.iter_ticks() if not params.get("error_free"))
    assert time == (1, 0, 0)
    errors = {time: {"before": QuantumCircuit([{"X": {next(iter(qecc.data_qudit_set))}}])}}
    raw_stats, prepared_stats = CircuitStatistics(), CircuitStatistics()
    raw = _shots(
        SparseStabPy,
        circuit,
        prepared=False,
        size=qecc.num_qudits,
        shots=5,
        inspector=raw_stats,
        error_circuits=errors,
    )
    prepared = _shots(
        SparseStabPy,
        circuit,
        prepared=True,
        size=qecc.num_qudits,
        shots=5,
        inspector=prepared_stats,
        error_circuits=errors,
    )
    assert raw == prepared
    assert raw_stats.data == prepared_stats.data
    assert raw_stats.data["runs"]


def test_logical_circuit_rejects_malformed_gate():
    qecc, circuit, extraction = _logical()
    extraction.circuits[0].append("X", {10**6})
    state = _state(SparseStabPy, qecc.num_qudits)
    before = _snapshot(state)
    with pytest.raises(ValueError, match="X: qubit 1000000 is out of bounds"):
        prepare_circuit(state, circuit)
    assert _snapshot(state) == before


def _duck(symbol, locations, **params):
    tick = SimpleNamespace(items=lambda: iter([(symbol, locations, params)]))
    return SimpleNamespace(iter_ticks=lambda: iter([(tick, 0, {})]), metadata={})


BAD_BATCHES = [
    ("X", {99}, r"X: qubit 99 is out of bounds \(capacity=4\)"),
    ("Measure +Z", {(0, 1)}, r"Measure \+Z: expected 1 qubit\(s\) per location, got \(0, 1\)"),
    ("CX", {(1, 1)}, "CX requires distinct qubits; repeated qubit 1"),
]


@pytest.mark.parametrize("cls", [SparseStabPy, SparseStab])
@pytest.mark.parametrize(("symbol", "locations", "message"), BAD_BATCHES)
@pytest.mark.parametrize(
    "params",
    [{}, {"skip": True}, {"cond": {"a": 0, "op": "==", "b": 1}}, {"cond2": {"a": 0, "op": "==", "b": 1}}],
)
def test_rejects_all_items_without_touching_state(cls, symbol, locations, message, params):
    if symbol == "CX":
        circuit = _duck(symbol, locations, **params)
    else:
        circuit = QuantumCircuit(num_qubits=4)
        circuit.append(symbol, locations, **params)
    for prepared in (False, True):
        engine = pecos.HybridEngine(seed=7)
        state = _state(cls)
        before = _snapshot(state)
        if params and not prepared:
            engine.run(state, circuit, shot_id=0)
        elif prepared:
            with pytest.raises(ValueError, match=message):
                prepare_circuit(state, circuit)
        else:
            with pytest.raises(ValueError, match=message):
                engine.run(state, circuit, shot_id=0)
        assert _snapshot(state) == before


@pytest.mark.parametrize("cls", [SparseStabPy, SparseStab])
def test_replaced_bad_qubit_is_still_rejected(cls):
    circuit = QuantumCircuit([{"X": {99}}], num_qubits=4)
    state = _state(cls)
    before = _snapshot(state)
    pecos.HybridEngine(seed=7).run(state, circuit, shot_id=0, error_circuits={0: {"replaced": {99}}})
    assert _snapshot(state) == before
    state = _state(cls)
    with pytest.raises(ValueError, match=BAD_BATCHES[0][2]):
        prepare_circuit(state, circuit)
    assert _snapshot(state) == before


@pytest.mark.parametrize("locations", [[0], [], (0,), iter([0])])
@pytest.mark.parametrize("symbol", ["X", "cop", "eop"])
def test_non_set_locations(locations, symbol):
    state = _state(SparseStabPy)
    before = _snapshot(state)
    with pytest.raises(TypeError, match=f"Gate {symbol}: prepared circuits need set locations, got"):
        prepare_circuit(state, _duck(symbol, locations, skip=True))
    assert _snapshot(state) == before


def test_requires_iter_ticks():
    with pytest.raises(TypeError, match=r"prepare_circuit needs a circuit with iter_ticks\(\).*got object"):
        prepare_circuit(object(), object())


class _RecordingInspector:
    supports_operation_events = True
    run_active = False

    def __init__(self):
        self.calls = []

    def start_run(self):
        self.calls.append("start")

    def finish_run(self):
        self.calls.append("finish")

    def abort_run(self):
        self.calls.append("abort")

    def analyze_operation(self, symbol, locations, metadata, *, executed):
        self.calls.append((symbol, locations, metadata, executed))


@pytest.mark.parametrize(("cls", "size"), [(StateVec, 4), (SparseStabPy, 5)])
def test_state_mismatch_precedes_inspector(cls, size):
    circuit = QuantumCircuit([{"X": {0}}], num_qubits=4)
    prepared = prepare_circuit(_state(SparseStabPy), circuit)
    state = _state(cls, size)
    before = _snapshot(state)
    inspector = _RecordingInspector()
    message = (
        f"prepared circuit was validated for SparseStabPy with gate_capacity=4; "
        f"got {cls.__name__} with gate_capacity={size}; re-run prepare_circuit for this state"
    )
    with pytest.raises(ValueError, match=message):
        pecos.HybridEngine(seed=7).run(state, prepared, shot_id=0, circ_inspector=inspector)
    assert _snapshot(state) == before
    assert inspector.calls == []


@pytest.mark.parametrize("method", ["run_circuit", "run_gate"])
@pytest.mark.parametrize("on_instance", [False, True])
def test_overriding_engine_rejects_prepared(method, on_instance):
    calls = []

    def run_circuit(self, *args, **kwargs):
        calls.append("run_circuit")
        return pecos.HybridEngine.run_circuit(self, *args, **kwargs)

    def run_gate(*args, **kwargs):
        calls.append("run_gate")
        return pecos.HybridEngine.run_gate(*args, **kwargs)

    if on_instance:
        engine = pecos.HybridEngine(seed=7)
        override = partial(run_circuit, engine) if method == "run_circuit" else run_gate
        setattr(engine, method, override)
    else:
        override = run_circuit if method == "run_circuit" else staticmethod(run_gate)
        engine = type("RecordingEngine", (pecos.HybridEngine,), {method: override})(seed=7)
    circuit = _circuit()
    engine.run(_state(SparseStabPy), circuit, shot_id=0)
    assert calls
    state = _state(SparseStabPy)
    before = _snapshot(state)
    inspector = _RecordingInspector()
    name = type(engine).__name__
    with pytest.raises(TypeError, match=f"{name} overrides {method}, which prepared circuits bypass"):
        engine.run(state, prepare_circuit(state, circuit), shot_id=0, circ_inspector=inspector)
    assert _snapshot(state) == before
    assert inspector.calls == []


@pytest.mark.parametrize("cls", [SparseStabPy, SparseStab])
@pytest.mark.parametrize("with_error", [False, True])
def test_no_per_shot_decode_or_validation(cls, with_error, monkeypatch):
    circuit = _circuit()
    engine = pecos.HybridEngine(seed=7)
    prepared = prepare_circuit(_state(cls), circuit)
    decoded, validated = [], []
    original_items = TickView.items

    def items(tick, *args, **kwargs):
        decoded.append(tick.circuit)
        return original_items(tick, *args, **kwargs)

    monkeypatch.setattr(TickView, "items", items)
    if cls is SparseStabPy:
        original_validator = cls.validate_gate_batch

        def validator(state, symbol, locations, **params):
            validated.append(symbol)
            return original_validator(state, symbol, locations, **params)

        monkeypatch.setattr(cls, "validate_gate_batch", validator)
    else:
        original_validator = hybrid_engine_old.validate_batch

        def validator(symbol, locations, capacity):
            validated.append(symbol)
            return original_validator(symbol, locations, capacity)

        monkeypatch.setattr(hybrid_engine_old, "validate_batch", validator)
    errors = {0: {"before": QuantumCircuit([{"X": {0}}])}} if with_error else None
    for i in range(5):
        engine.run(_state(cls, shot=i), prepared, shot_id=i, error_circuits=errors)
    assert circuit not in decoded
    assert validated == (["X"] * 5 if with_error else [])


def test_frozen_container_retains_shared_leaves():
    locations, params, tick_params = {0}, {"angles": [0.25]}, {"error_free": True}
    decoded = [("RZ", locations, params)]
    calls = []

    def items():
        calls.append("items")
        return iter(decoded)

    tick = SimpleNamespace(items=items)
    circuit = SimpleNamespace(iter_ticks=lambda: iter([(tick, 0, tick_params)]), metadata={})
    state = _state(StateVec)
    before = _snapshot(state)
    prepared = prepare_circuit(state, circuit)
    assert isinstance(prepared, PreparedCircuit)
    assert calls == ["items"]
    assert prepared in {prepared}
    assert prepared.ticks[0] in {prepared.ticks[0]}
    assert _snapshot(state) == before
    assert prepared.source is circuit
    assert prepared.ticks[0].tick is tick
    assert prepared.ticks[0].params is tick_params
    assert prepared.ticks[0].items[0] is decoded[0]
    assert prepared.ticks[0].items[0][1] is locations
    assert prepared.ticks[0].items[0][2] is params
    with pytest.raises(FrozenInstanceError):
        prepared.source = None
    with pytest.raises(FrozenInstanceError):
        prepared.ticks[0].time = 1
    engine = pecos.HybridEngine(seed=7)
    for i in range(3):
        engine.run(_state(StateVec, shot=i), prepared, shot_id=i)
    assert params == {"angles": [0.25]}


def test_empty_batches_skip_validation_and_rng():
    circuit = _duck("MZForced", set(), forced_outcome=99)
    engine = pecos.HybridEngine(seed=7)
    state = _state(CoinToss)
    prepared = prepare_circuit(state, circuit)
    assert prepared.ticks[0].items[0][1] == set()
    engine.run(state, prepared, shot_id=0)
    assert state.run_gate("Measure", {0}) == _state(CoinToss).run_gate("Measure", {0})
