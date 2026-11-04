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

"""Prepared Standard ticks preserve simulator dispatch and threshold shot streams."""

from collections import Counter
from copy import deepcopy

import pecos as pc
import pytest
from pecos.analysis import threshold_tools
from pecos.circuits import LogicalCircuit, QuantumCircuit
from pecos.circuits.quantum_circuit import TickView
from pecos.engines.circuit_runners import Standard, TimingRunner
from pecos.simulators import PauliProp, PreparedCircuit, SparseStab, SparseStabPy, StateVec, prepare_circuit
from pecos.simulators.batch_validation import PreparedTick


def _state(cls, size=4, shot=0):
    params = {}
    if cls in (StateVec, SparseStab):
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
    circuit = QuantumCircuit(num_qubits=4)
    circuit.append("init |0>", {0, 1, 2, 3})
    circuit.append("H", {0, 2})
    circuit.append("CX", {(0, 1), (2, 3)})
    circuit.append("measure Z", {0, 1, 2, 3})
    return circuit


def _logical(qecc, *, initialize=False):
    circuit = LogicalCircuit(suppress_warning=True)
    if initialize:
        circuit.append(qecc.gate("ideal init |0>"))
    circuit.append(qecc.gate("I", num_syn_extract=1))
    return circuit


def _shots(cls, circuit, size, noise, *, prepared):
    # Seed 3 generates both before-measurement and after-gate errors in 20 shots.
    runner = Standard(seed=3)
    program = prepare_circuit(_state(cls, size), circuit) if prepared else circuit
    kwargs = {}
    if noise == "generated":
        kwargs = {"error_gen": pc.noise.DepolarModel(model_level="circuit"), "error_params": {"p": 0.05}}
    elif noise == "replaced":
        tick, time, _ = next(
            entry
            for entry in circuit.iter_ticks()
            if not entry[2].get("error_free") and any(symbol == "measure Z" for symbol, _, _ in entry[0].items())
        )
        locations = next(locations for symbol, locations, _ in tick.items() if symbol == "measure Z")
        location = next(iter(locations))
        kwargs = {
            "error_circuits": {
                time: {
                    "before": QuantumCircuit([{"X": {0}}]),
                    "after": QuantumCircuit([{"Z": {1}}]),
                    "replaced": {location},
                },
            },
        }
    results, kinds = [], set()
    for shot in range(20):
        state = _state(cls, size, shot)
        output, errors = runner.run(state, program, **kwargs)
        if noise == "replaced":
            assert location not in output.get(time, {})
        structural_errors = {
            time: {kind: frozenset(value) if kind == "replaced" else str(value) for kind, value in entries.items()}
            for time, entries in errors.items()
        }
        for entries in errors.values():
            kinds.update(kind for kind, value in entries.items() if value)
        results.append((deepcopy(output), _snapshot(state), structural_errors))
    if noise == "generated":
        assert {"before", "after"} <= kinds
    return results, list(pc.random.random(4))


@pytest.mark.parametrize("cls", [SparseStabPy, StateVec, SparseStab, PauliProp])
@pytest.mark.parametrize("noise", ["none", "generated", "replaced"])
def test_standard_equivalence(cls, noise):
    circuit = _circuit()
    assert _shots(cls, circuit, 4, noise, prepared=False) == _shots(cls, circuit, 4, noise, prepared=True)


@pytest.mark.parametrize("cls", [SparseStabPy, SparseStab, PauliProp])
@pytest.mark.parametrize("noise", ["none", "generated", "replaced"])
def test_logical_equivalence(cls, noise):
    qecc = pc.qeccs.Surface4444(distance=3)
    circuit = _logical(qecc, initialize=True)
    assert _shots(cls, circuit, qecc.num_qudits, noise, prepared=False) == _shots(
        cls,
        circuit,
        qecc.num_qudits,
        noise,
        prepared=True,
    )


def test_prepared_tick_fault_metadata(monkeypatch):
    circuit = QuantumCircuit([{"X": {0}, "Y": {1}, "Z": {2}, "I": {3}}])
    prepared = prepare_circuit(_state(PauliProp), circuit)
    circuit.metadata["circuit_type"] = "faults"
    assert prepared.ticks[0].metadata is circuit.metadata
    assert list(prepared.ticks[0].items(None)) == list(prepared.ticks[0].batches)
    calls = []
    original = PauliProp.add_faults

    def add_faults(state, tick, **kwargs):
        calls.append(tick)
        return original(state, tick, **kwargs)

    monkeypatch.setattr(PauliProp, "add_faults", add_faults)
    raw_state, prepared_state = _state(PauliProp), _state(PauliProp)
    before = _snapshot(raw_state)
    Standard.run(raw_state, circuit)
    Standard.run(prepared_state, prepared)
    assert len(calls) == 2
    assert isinstance(calls[0], TickView)
    assert calls[1] is prepared.ticks[0]
    assert _snapshot(raw_state) == _snapshot(prepared_state) != before


@pytest.mark.parametrize(("cls", "size"), [(StateVec, 4), (SparseStabPy, 5)])
def test_standard_state_check(cls, size):
    prepared = prepare_circuit(_state(SparseStabPy), QuantumCircuit([{"X": {0}}]))
    state = _state(cls, size)
    before = _snapshot(state)
    error_gen = _RecordingErrors()
    message = (
        f"prepared circuit was validated for SparseStabPy with gate_capacity=4; "
        f"got {cls.__name__} with gate_capacity={size}; re-run prepare_circuit for this state"
    )
    with pytest.raises(ValueError, match=message) as exc:
        Standard.run(state, prepared, error_gen=error_gen)
    assert str(exc.value) == message
    assert _snapshot(state) == before
    assert error_gen.circuit is None


def _spy_decodes(monkeypatch):
    viewed, decoded, ideals = [], [], []
    original_items = TickView.items
    original_iter = vars(QuantumCircuit)["_iter_tick"]
    original_run = SparseStabPy.run_circuit

    def items(tick, *args, **kwargs):
        viewed.append(tick.circuit)
        return original_items(tick, *args, **kwargs)

    def iter_tick(circuit, tick_idx):
        decoded.append((circuit, tick_idx))
        return original_iter(circuit, tick_idx)

    def run_circuit(state, circuit, **kwargs):
        if "removed_locations" in kwargs:
            ideals.append(circuit)
        return original_run(state, circuit, **kwargs)

    monkeypatch.setattr(TickView, "items", items)
    monkeypatch.setattr(QuantumCircuit, "_iter_tick", iter_tick)
    monkeypatch.setattr(SparseStabPy, "run_circuit", run_circuit)
    return viewed, decoded, ideals


@pytest.mark.parametrize("noise", [False, True])
def test_no_per_shot_decode(noise, monkeypatch):
    qecc = pc.qeccs.Surface4444(distance=3)
    circuit = _logical(qecc)
    prepared = prepare_circuit(SparseStabPy(qecc.num_qudits), circuit)
    fixed_ids = {id(c) for c in qecc.gate("I", num_syn_extract=1).circuits}
    viewed, decoded, ideals = _spy_decodes(monkeypatch)
    runner = Standard(seed=1)
    kwargs = {}
    if noise:
        kwargs = {"error_gen": pc.noise.DepolarModel(model_level="code_capacity"), "error_params": {"p": 0.1}}
    for _ in range(5):
        runner.run(SparseStabPy(qecc.num_qudits), prepared, **kwargs)
    assert not [c for c in viewed if id(c) in fixed_ids]
    assert not [(c, tick) for c, tick in decoded if id(c) in fixed_ids]
    assert len(ideals) == 5 * len(prepared.ticks)
    assert all(isinstance(tick, PreparedTick) for tick in ideals)


def _rate(helper, *, runs=5, seed=1, qecc=None, **kwargs):
    if qecc is None:
        qecc = pc.qeccs.Surface4444(distance=3)
    return helper(
        runs,
        qecc,
        3,
        pc.noise.DepolarModel(model_level="code_capacity"),
        {"p": 0.1},
        pc.decoders.MWPM2D(qecc),
        seed=seed,
        verbose=False,
        **kwargs,
    )[0]


@pytest.mark.parametrize(
    ("helper", "runs", "kwargs", "expected"),
    [
        (threshold_tools.codecapacity_logical_rate, 40, {"basis": "zero"}, (0.05, 0.175)),
        (threshold_tools.codecapacity_logical_rate, 40, {"basis": "plus"}, (0.075, 0.1)),
        (threshold_tools.codecapacity_logical_rate2, 40, {"state_sim": SparseStabPy}, (0.1, 0.275)),
        (threshold_tools.codecapacity_logical_rate3, 10, {}, (0.06802721088435375, 0.1388888888888889)),
    ],
    ids=["zero", "plus", "rate2", "rate3"],
)
@pytest.mark.parametrize("seed", [1, 2])
def test_threshold_oracle(helper, runs, kwargs, expected, seed):
    assert _rate(helper, runs=runs, seed=seed, **kwargs) == expected[seed - 1]


def test_helper_decode_count(monkeypatch):
    qecc = pc.qeccs.Surface4444(distance=3)
    gates = [qecc.gate("I", num_syn_extract=1), qecc.gate("ideal init |0>"), qecc.gate("ideal init |+>")]
    fixed = {id(c): c for gate in gates for c in gate.circuits}
    expected = Counter({(id(c), tick): 1 for c in fixed.values() for _, tick, _ in c.iter_ticks()})
    viewed, decoded, ideals = _spy_decodes(monkeypatch)
    _rate(threshold_tools.codecapacity_logical_rate2, qecc=qecc)
    assert {id(c) for c in viewed if id(c) in fixed} == set(fixed)
    assert Counter((id(c), tick) for c, tick in decoded if id(c) in fixed) == expected
    assert ideals
    assert all(isinstance(tick, PreparedTick) for tick in _fixed_ticks(ideals, fixed))


def _fixed_ticks(ideals, fixed):
    return [
        tick for tick in ideals if id(tick.tick.circuit if isinstance(tick, PreparedTick) else tick.circuit) in fixed
    ]


class _OverrideRunner(Standard):
    @staticmethod
    def run(state, circuit, **kwargs):
        assert not isinstance(circuit, PreparedCircuit)
        return Standard.run(state, circuit, **kwargs)


class _InheritedRunner(Standard):
    pass


class _SlotsRunner:
    __slots__ = ()

    @staticmethod
    def run(state, circuit, **kwargs):
        assert not isinstance(circuit, PreparedCircuit)
        return Standard.run(state, circuit, **kwargs)


@pytest.mark.parametrize(
    ("helper", "count"),
    [
        (threshold_tools.codecapacity_logical_rate, 2),
        (threshold_tools.codecapacity_logical_rate2, 3),
        (threshold_tools.codecapacity_logical_rate3, 2),
    ],
)
@pytest.mark.parametrize("kind", ["override", "instance", "standard", "timing", "inherited", "slots", "class"])
def test_runner_selection(helper, count, kind, monkeypatch):
    qecc = pc.qeccs.Surface4444(distance=3)
    gates = [qecc.gate("I", num_syn_extract=1), qecc.gate("ideal init |0>"), qecc.gate("ideal init |+>")]
    fixed = {id(c) for gate in gates for c in gate.circuits}
    runners = {"override": _OverrideRunner, "standard": Standard, "timing": TimingRunner, "inherited": _InheritedRunner}
    if kind == "instance":
        runner = Standard(seed=1)
        runner.run = _OverrideRunner.run
    elif kind == "slots":
        runner = _SlotsRunner()
        pc.random.seed(1)
    elif kind == "class":
        runner = Standard
        pc.random.seed(1)
    else:
        runner = runners[kind](seed=1)
    preparations = []
    original = threshold_tools.prepare_circuit

    def prepare(state, circuit):
        result = original(state, circuit)
        preparations.append((state, result))
        return result

    monkeypatch.setattr(threshold_tools, "prepare_circuit", prepare)
    _, _, ideals = _spy_decodes(monkeypatch)
    _rate(helper, runs=1, qecc=qecc, circuit_runner=runner)
    expected = kind in {"standard", "timing", "inherited"}
    assert len(preparations) == (count if expected else 0)
    assert ideals
    assert _fixed_ticks(ideals, fixed)
    assert all(isinstance(tick, PreparedTick if expected else TickView) for tick in _fixed_ticks(ideals, fixed))
    assert all(isinstance(tick, TickView) for tick in ideals if tick not in _fixed_ticks(ideals, fixed))
    if expected:
        assert len({id(state) for state, _ in preparations}) == 1
        state = preparations[0][0]
        assert _snapshot(state) == _snapshot(SparseStabPy(state.num_qubits))


class _RecordingErrors:
    def __init__(self):
        self.circuit = None

    def start(self, circuit, _error_params):
        self.circuit = circuit
        return {}

    def generate_tick_errors(self, _tick, _time, **_params):
        return {}


def test_error_model_receives_source():
    state = _state(SparseStabPy)
    prepared = prepare_circuit(state, _circuit())
    error_gen = _RecordingErrors()
    Standard.run(state, prepared, error_gen=error_gen)
    assert error_gen.circuit is prepared.source


@pytest.mark.parametrize(("runner_class", "exception"), [(Standard, TypeError), (_OverrideRunner, AttributeError)])
def test_rate3_invalid_init_circuit(runner_class, exception):
    with pytest.raises(exception, match="iter_ticks"):
        _rate(
            threshold_tools.codecapacity_logical_rate3,
            runs=1,
            circuit_runner=runner_class(seed=1),
            init_circuit=object(),
            init_logical_ops=QuantumCircuit([{"Z": {0}}]),
        )
