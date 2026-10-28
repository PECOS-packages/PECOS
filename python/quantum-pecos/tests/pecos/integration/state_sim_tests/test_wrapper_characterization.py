# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Wrapper contracts recorded on dev ad1e97642, updated for #1480 batch validation.

The public pecos.HybridEngine is the old bindings-based engine; PHIR uses the
separate engines.hybrid_engine.HybridEngine. Optional dependencies alone may
skip cases: constructor/runtime failures in an installed backend must fail.
"""

from importlib.util import find_spec
from math import pi
from types import SimpleNamespace

import pecos
import pytest
from pecos import simulators
from pecos.circuits import QuantumCircuit
from pecos.engines.hybrid_engine import HybridEngine as PhirEngine
from pecos.noise.class_errors_circuit import ErrorCircuits
from pecos.noise.noise_impl.noise_meas_bitflip import noise_meas_bitflip
from pecos.noise.parent_class_error_gen import Generator
from pecos.reps.pyphir.op_types import QOp
from pecos.simulators import CoinToss, DefaultSimulator, PauliProp, StateVec
from pecos.simulators.quantum_simulator import QuantumSimulator
from pecos_rslib import AngleSource, GateRegistry

# GPU expectations (CuStateVec/MPS/CudaStateVec) are unverified until run on a CUDA host.
# CoinToss run_gate assertions are input-independent no-op pins of current dispatch.
QUANTUM = ("StateVec", "SparseStabPy", "CuStateVec", "MPS", "CudaStateVec")
PAULI = ("PauliProp",)
WRAPPERS = (*QUANTUM, *PAULI, "CoinToss")
CONTAINERS = (list, tuple, set, frozenset, iter)


def _require_backend(name):
    dependencies = {
        "CuStateVec": ("cupy", "cuquantum"),
        "MPS": ("cupy", "cuquantum", "pytket", "pytket.extensions.cutensornet"),
        "CudaStateVec": ("pecos_rslib_cuda",),
    }
    for dependency in dependencies.get(name, ()):
        try:
            spec = find_spec(dependency)
        except ModuleNotFoundError:
            spec = None
        if spec is None:
            pytest.skip(f"{name}: {dependency} is not installed")
    if name == "CudaStateVec":
        from pecos_rslib_cuda import is_cuquantum_available

        if not is_cuquantum_available():
            pytest.skip("CudaStateVec: cuQuantum SDK is not installed (stub build)")


@pytest.fixture
def state(request):
    name = request.param
    if name == "DefaultSimulator":
        return DefaultSimulator()
    _require_backend(name)
    params = {"num_qubits": 4}
    if name in {"StateVec", "MPS", "CudaStateVec", "CoinToss"}:
        params["seed"] = 42
    if name == "CoinToss":
        params["prob"] = 1
    if name in PAULI:
        params["track_sign"] = True
    return getattr(simulators, name)(**params)


def _items(symbol, locations, **params):
    # A minimal circuit preserves noise/PHIR containers that QuantumCircuit normalizes.
    return SimpleNamespace(items=lambda: iter([(symbol, locations, params)]), metadata={})


def _prepare_one(state):
    if isinstance(state, PauliProp):
        state.faults = {"X": {1}, "Y": set(), "Z": set()}
    else:
        state.run_gate("X", {1})


@pytest.mark.parametrize("state", WRAPPERS, indirect=True)
def test_result_int_keys_and_zero_omission(state):
    _prepare_one(state)
    if isinstance(state, CoinToss):
        # Both CoinToss entry points record ones and omit zeros.
        assert state.run_gate("Measure", {0, 1}) == {0: 1, 1: 1}
        assert state.run_circuit(_items("Measure", {0, 1})) == {0: 1, 1: 1}
        state.prob = 0
        assert state.run_gate("Measure", {0, 1}) == {}
        assert state.run_circuit(_items("Measure", {0, 1})) == {}
    elif type(state).__name__ == "CudaStateVec":
        # KNOWN-WRONG (#1480): CudaStateVec retains zeros that other wrappers omit.
        assert state.run_gate("Measure", [0, 1]) == {0: 0, 1: 1}
    else:
        # Outcome zero has no dictionary entry; the original integer is the key.
        assert state.run_gate("Measure", [0, 1]) == {1: 1}


@pytest.mark.parametrize("state", WRAPPERS, indirect=True)
def test_result_singleton_tuple(state):
    _prepare_one(state)
    if isinstance(state, PauliProp):
        # The generated native binding accepts 1-tuples; the wrapper retains them.
        assert state.run_gate("Measure", [(1,)]) == {(1,): 1}
    elif isinstance(state, CoinToss):
        # Both CoinToss entry points reject non-scalar measurement locations.
        with pytest.raises(ValueError, match="Measure: expected a scalar qubit index"):
            state.run_gate("Measure", [(1,)])
        with pytest.raises(ValueError, match="Measure: expected a scalar qubit index"):
            state.run_circuit(_items("Measure", {(1,)}))
    else:
        # These measurement helpers require a scalar qubit, even in a batch.
        with pytest.raises(ValueError, match="Measure"):
            state.run_gate("Measure", [(1,)])


@pytest.mark.parametrize("state", WRAPPERS, indirect=True)
@pytest.mark.parametrize("container", CONTAINERS, ids=["list", "tuple", "set", "frozenset", "generator"])
def test_location_containers(state, container):
    _prepare_one(state)
    if isinstance(state, CoinToss):
        # Both entry points measure all iterable containers.
        assert state.run_gate("Measure", container([1])) == {1: 1}
        assert state.run_circuit(_items("Measure", container([1]))) == {1: 1}
    else:
        # All five iterable containers dispatch measurements in the other wrappers.
        assert state.run_gate("Measure", container([1])) == {1: 1}


@pytest.mark.parametrize("state", WRAPPERS, indirect=True)
def test_duplicate_x_list(state):
    if isinstance(state, PauliProp):
        state.faults = {"X": set(), "Y": set(), "Z": {1}}
    # A noise-style [symbol, locations] batch retains duplicates, applying X twice.
    assert state.run_gate(*["X", [1, 1]]) == {}
    if isinstance(state, PauliProp):
        assert state.faults == {"X": set(), "Y": set(), "Z": {1}}
        assert state.sign == 0
    elif isinstance(state, CoinToss):
        assert state.run_gate("Measure", {1}) == {1: 1}
    elif type(state).__name__ == "CudaStateVec":
        assert state.run_gate("Measure", {1}) == {1: 0}
    else:
        assert state.run_gate("Measure", {1}) == {}


@pytest.mark.parametrize("state", WRAPPERS, indirect=True)
@pytest.mark.parametrize("pairs", [{(0, 1), (1, 2)}, [(0, 1), (1, 2)], [(1, 2), (0, 1)]])
def test_overlapping_pairs(state, pairs):
    if isinstance(state, PauliProp):
        state.faults = {"X": {0}, "Y": set(), "Z": set()}
    else:
        state.run_gate("X", {0})
    assert state.run_gate("CX", pairs) == {}
    if isinstance(state, CoinToss):
        # CoinToss ignores CX but now runs the following measurements.
        assert state.run_gate("Measure", [0, 1, 2]) == {0: 1, 1: 1, 2: 1}
        return
    # Each literal was recorded on dev; select by the actual container iteration order.
    reverse = list(pairs) == [(1, 2), (0, 1)]
    expected = {0: 1, 1: 1} if reverse else {0: 1, 1: 1, 2: 1}
    if type(state).__name__ == "CudaStateVec" and reverse:
        expected = {0: 1, 1: 1, 2: 0}
    assert state.run_gate("Measure", [0, 1, 2]) == expected


@pytest.mark.parametrize("state", WRAPPERS, indirect=True)
@pytest.mark.parametrize("symbol", ["X", "unknown-wrapper-symbol", "Tick"])
def test_empty_locations(state, symbol):
    if isinstance(state, PauliProp):
        state.faults = {"X": {3}, "Y": set(), "Z": set()}
    circuit = QuantumCircuit()
    circuit.append(symbol, set())
    # QuantumCircuit yields tick-level symbols too; empty sets never look up a binding.
    assert list(circuit.items()) == [(symbol, set(), {})]
    assert state.run_gate(symbol, set()) == {}
    assert state.run_circuit(circuit) == {}


@pytest.mark.parametrize("state", WRAPPERS, indirect=True)
@pytest.mark.parametrize("locations", [[1], [1, 99]])
def test_simulate_gate_false(state, locations):
    if isinstance(state, PauliProp):
        state.faults = {"X": set(), "Y": set(), "Z": {1}}
    # Fixed wrappers validate before skipping; growable PauliProp has no upper bound.
    if locations == [1, 99] and not isinstance(state, (PauliProp, CoinToss)):
        with pytest.raises(ValueError, match="X"):
            state.run_gate("X", locations, simulate_gate=False)
    else:
        assert state.run_gate("X", locations, simulate_gate=False) == {}
    if isinstance(state, PauliProp):
        assert state.faults == {"X": set(), "Y": set(), "Z": {1}}
        assert state.sign == 0
    elif isinstance(state, CoinToss):
        assert state.run_gate("Measure", {1}) == {1: 1}
    elif type(state).__name__ == "CudaStateVec":
        assert state.run_gate("Measure", {1}) == {1: 0}
    else:
        assert state.run_gate("Measure", {1}) == {}


@pytest.mark.parametrize("state", WRAPPERS, indirect=True)
def test_bare_ints(state):
    # KNOWN-WRONG (#1480): Bare 0 is silently skipped while bare 1 raises in non-CoinToss wrappers.
    assert state.run_gate("X", 0) == {}
    if isinstance(state, CoinToss):
        assert state.run_gate("X", 1) == {}
    else:
        # A nonzero bare integer reaches the loop and is not iterable.
        with pytest.raises(TypeError, match="not iterable"):
            state.run_gate("X", 1)


@pytest.mark.parametrize("state", WRAPPERS, indirect=True)
def test_identity_single_qubit_batch(state):
    # I is accepted on a batch of scalar qubits.
    assert state.run_gate("I", {1, 2, 3}) == {}


@pytest.mark.parametrize("state", [*QUANTUM, "CoinToss"], indirect=True)
def test_identity_pair_today(state):
    # CoinToss remains a no-op; quantum wrappers now require the II spelling.
    if isinstance(state, CoinToss):
        assert state.run_gate("I", {(0, 1)}) == {}
    else:
        with pytest.raises(ValueError, match="II"):
            state.run_gate("I", {(0, 1)})


@pytest.mark.parametrize("state", QUANTUM, indirect=True)
def test_identity_pair_future(state):
    # Q1: I has arity one; CoinToss non-measurement gates remain no-ops.
    with pytest.raises(ValueError, match="II"):
        state.run_gate("I", {(0, 1)})


@pytest.mark.parametrize("state", PAULI, indirect=True)
def test_pauli_identity_pair_already_rejected(state):
    # Whole-batch validation now supplies the arity error and II hint.
    with pytest.raises(
        ValueError,
        match=r"^I: expected 1 qubit\(s\) per location, got \(0, 1\); use II for the two-qubit identity$",
    ):
        state.run_gate("I", {(0, 1)})


@pytest.mark.parametrize("state", WRAPPERS, indirect=True)
def test_public_hybrid_engine(state):
    _prepare_one(state)
    output = {"m": pecos.BitInt(2, 0)}
    circuit = QuantumCircuit()
    circuit.append("Measure", {1}, var=("m", 1))
    # KNOWN-WRONG (#1480): CoinToss measures through HybridEngine but drops its own run_gate measurements.
    assert pecos.HybridEngine.__module__ == "pecos.engines.hybrid_engine_old"
    pecos.HybridEngine().run_circuit(state, output, {}, circuit, None)
    assert int(output["m"]) == 2
    pecos.HybridEngine.run_gate(state, output, "Measure", {0}, var=("m", 0))
    assert int(output["m"]) == (3 if isinstance(state, CoinToss) else 2)


@pytest.mark.parametrize("state", ["StateVec", "SparseStabPy"], indirect=True)
def test_error_set_circuit(state):
    after = QuantumCircuit()
    Generator.ErrorSet([pecos.Pauli.X]).error_func(after, QuantumCircuit(), set(), 1, {})
    # Legacy ErrorSet emits {symbol: {gate_location}}, not a flattened/native batch.
    assert list(after.items()) == [("X", {1}, {})]
    assert state.run_circuit(after) == {}
    assert state.run_gate("Measure", {0, 1}) == {1: 1}


@pytest.mark.parametrize("state", PAULI, indirect=True)
def test_growable_pauli(state):
    state = type(state)(num_qubits=2)
    state.faults = {"X": set(), "Y": set(), "Z": {3}}
    # num_qubits is display width, not capacity; a seeded target Z propagates to control 0.
    assert state.run_gate("CX", {(0, 3)}) == {}
    assert state.faults == {"X": set(), "Y": set(), "Z": {0, 3}}
    assert state.run_circuit(QuantumCircuit([{"CX": {(0, 3)}}])) == {}
    assert state.faults == {"X": set(), "Y": set(), "Z": {3}}


@pytest.mark.parametrize("symbol", ["RX", "RY", "RZ", "RXX", "RYY", "RZZ", "RXXRYYRZZ"])
def test_statevec_default_angles(symbol):
    state = StateVec(2, seed=42)
    state.run_gate("H", {0})
    state.run_gate("X", {1})
    assert state.run_gate(symbol, {0} if symbol in {"RX", "RY", "RZ"} else {(0, 1)}) == {}
    # KNOWN-WRONG (#1480): Missing rotation angles silently become identity operations.
    assert list(state.vector) == pytest.approx([0, 0.7071067811865476, 0, 0.7071067811865476])


@pytest.mark.parametrize(
    ("order", "expected", "keys"),
    [([3, 0, 2, 1], {0: 1, 2: 1, 1: 1}, [0, 2, 1]), ([0, 1, 2, 3], {1: 1, 2: 1, 3: 1}, [1, 2, 3])],
)
def test_statevec_seeded_measurement_order(order, expected, keys):
    state = StateVec(4, seed=42)
    state.run_gate("H", [0, 1, 2, 3])
    # The fixed RNG stream is assigned in list order, not sorted or set order.
    result = state.run_gate("Measure", order)
    assert result == expected
    assert list(result) == keys


@pytest.mark.parametrize("state", ["StateVec", "SparseStabPy", *PAULI], indirect=True)
def test_u1q_alias(state):
    if isinstance(state, PauliProp):
        state.faults = {"X": set(), "Y": set(), "Z": {0}}
    # U1q(pi, 0) is an X rotation; Pauli propagation retains Z and reverses its sign.
    assert state.run_gate("U1q", {0}, angles=(pi, 0.0)) == {}
    if isinstance(state, PauliProp):
        assert state.faults == {"X": set(), "Y": set(), "Z": {0}}
        assert state.sign == 1
    else:
        assert state.run_gate("Measure", {0}) == {0: 1}


@pytest.mark.parametrize("state", ["SparseStabPy", *PAULI], indirect=True)
@pytest.mark.parametrize(
    ("symbol", "angle"),
    [("ZZ", pi / 2), ("ZZMax", pi / 2), ("ZZPhase", pi / 2), ("ZZPhase", -pi / 2), ("MSYY", pi / 2), ("MSZZ", pi / 2)],
)
def test_two_qubit_aliases(state, symbol, angle):
    if isinstance(state, PauliProp):
        state.faults = {"X": {0}, "Y": set(), "Z": set()}
    else:
        state.run_gate("H", {0})
    # These spellings bind quarter-turn ZZ/YY rotations, including ZZPhase's angle.
    assert state.run_gate(symbol, {(0, 1)}, angles=(angle,)) == {}
    if isinstance(state, PauliProp):
        expected = {"X": set(), "Y": {0}, "Z": {1}}
        if symbol == "MSYY":
            expected = {"X": set(), "Y": {1}, "Z": {0}}
        assert state.faults == expected
        assert state.sign == (1 if symbol == "MSYY" or angle < 0 else 0)
    elif symbol == "MSYY":
        # Stabilizer strings pin the entangled YY effect without a random measurement.
        assert state.stabs.row_string() == ["-iZWII", " iWXII", "  IIZI", "  IIIZ"]
    elif angle < 0:
        assert state.stabs.row_string() == ["-iWZII", "  IZII", "  IIZI", "  IIIZ"]
        assert state.run_gate("Measure +Y", {0}) == {0: 1}
    else:
        assert state.stabs.row_string() == [" iWZII", "  IZII", "  IIZI", "  IIIZ"]
        assert state.run_gate("Measure +Y", {0}) == {}


@pytest.mark.parametrize(
    "symbol",
    [
        "Measure",
        "measure Z",
        "Measure +Z",
        "Measure -Z",
        "Measure +X",
        "Measure -X",
        "Measure +Y",
        "Measure -Y",
        "MX",
        "MY",
        "MZ",
    ],
)
def test_cointoss_measure_spellings(symbol):
    state = CoinToss(2, prob=1, seed=42)
    # All measurement bindings and prefix spellings draw coins in both entry points.
    assert state.run_circuit(_items(symbol, {1})) == {1: 1}
    assert state.run_gate(symbol, {1}) == {1: 1}
    assert state.bindings[symbol](state, 1) == 1


@pytest.mark.parametrize("backend", QUANTUM)
def test_quantum_simulator_run(backend):
    _require_backend(backend)
    sim = QuantumSimulator(backend="stabilizer" if backend == "SparseStabPy" else backend, seed=42)
    sim.init(2)
    # QuantumSimulator restores missing zeros and maps qubits to classical return locations.
    assert sim.run([QOp("X", [1]), QOp("Measure", [0, 1], returns=[["m", 0], ["m", 1]])]) == [
        {("m", 0): 0, ("m", 1): 1},
    ]


@pytest.mark.parametrize("backend", QUANTUM)
@pytest.mark.parametrize("interpreter", ["python", "rust"])
def test_phir_zero_rzz(backend, interpreter, monkeypatch):
    _require_backend(backend)
    dispatched = []
    original_run = QuantumSimulator.run

    def record_run(self, qops):
        dispatched.extend((op.sim_name, op.args) for op in qops if op.name in {"RZZ", "CX"})
        return original_run(self, qops)

    monkeypatch.setattr(QuantumSimulator, "run", record_run)
    program = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
            {"data": "cvar_define", "data_type": "u32", "variable": "m", "size": 2},
            {"qop": "H", "args": [["q", 0], ["q", 1]]},
            {"qop": "RZZ", "angles": [[0.0], "rad"], "args": [[["q", 0], ["q", 1]]]},
            {"qop": "H", "args": [["q", 0], ["q", 1]]},
            {"qop": "X", "args": [["q", 0], ["q", 1]]},
            {"qop": "CX", "args": [[["q", 0], ["q", 1]]]},
            {"qop": "Measure", "args": [["q", 0], ["q", 1]], "returns": [["m", 0], ["m", 1]]},
        ],
    }
    engine = PhirEngine(qsim="stabilizer" if backend == "SparseStabPy" else backend, cinterp=interpreter)
    # RZZ(0) resolves to II with its inner pair list intact under both PHIR interpreters.
    assert engine.run(program=program, shots=2, seed=42)["m"] == ["01", "01"]
    assert dispatched == [("II", [[0, 1]]), ("CX", [[0, 1]]), ("II", [[0, 1]]), ("CX", [[0, 1]])]


class _Location(int):
    """An in-range qubit index whose object identity cannot come from the small-int cache."""


@pytest.mark.parametrize("state", [*QUANTUM, *PAULI, "DefaultSimulator"], indirect=True)
@pytest.mark.parametrize(
    ("symbol", "location", "one", "zero"),
    [
        ("Measure", _Location(1), {1: 1}, {1: 0}),
        ("Measure", (1,), {(1,): 1}, {(1,): 0}),
        ("CX", (0, 1), {(0, 1): 1}, {(0, 1): 0}),
    ],
)
def test_original_result_keys_from_overridden_binding(state, symbol, location, one, zero):
    state.bindings = dict(state.bindings)
    state.bindings[symbol] = lambda _state, _location, **_params: 1
    # Overrides retain the state's location policy as well as the spelling's arity.
    if location == (1,) and type(state).__name__ in QUANTUM:
        with pytest.raises(ValueError, match="Measure"):
            state.run_gate(symbol, [location])
        return
    # Accepted scalar, singleton and pair keys stay original.
    result = state.run_gate(symbol, [location])
    assert result == one
    assert next(iter(result)) is location
    state.bindings[symbol] = lambda _state, _location, **_params: 0
    # KNOWN-WRONG (#1480): CudaStateVec keeps zero results from overridden bindings too.
    assert state.run_gate(symbol, [location]) == (zero if type(state).__name__ == "CudaStateVec" else {})


@pytest.mark.parametrize("state", [*QUANTUM, *PAULI, "DefaultSimulator"], indirect=True)
def test_list_result_key_is_unhashable(state):
    state.bindings = dict(state.bindings)
    calls = []

    def result_one(_state, location, **_params):
        calls.append(location)
        return 1

    state.bindings["CX"] = result_one
    # KNOWN-WRONG (#1480): A binding runs before result insertion rejects an unhashable key.
    with pytest.raises(TypeError, match="unhashable type: 'list'"):
        state.run_gate("CX", [[0, 1]])
    assert calls == ([(0, 1)] if isinstance(state, StateVec) else [[0, 1]])


@pytest.mark.parametrize("container", CONTAINERS, ids=["list", "tuple", "set", "frozenset", "generator"])
def test_default_simulator_containers(container):
    state = DefaultSimulator()
    calls = []
    state.bindings["X"] = lambda _state, location, **_params: calls.append(location)
    # The base class dispatches each of the five iterable container types.
    assert state.run_gate("X", container([1])) == {}
    assert calls == [1]
    assert state.run_gate("X", [1, 1]) == {}
    assert calls == [1, 1, 1]
    assert state.run_circuit(_items("X", container([1]))) == {}
    assert calls == [1, 1, 1, 1]


def test_default_simulator_dispatch_edges():
    state = DefaultSimulator()
    calls = []
    state.bindings["X"] = lambda _state, location, **_params: calls.append(location)
    state.bindings["I"] = lambda _state, _location, **_params: None
    # Empty, disabled and falsey bare-zero calls never look up or invoke a binding.
    for symbol in ("X", "unknown-wrapper-symbol", "Tick"):
        assert state.run_gate(symbol, set()) == {}
        assert state.run_circuit(_items(symbol, set())) == {}
    assert state.run_gate("X", [1], simulate_gate=False) == {}
    assert state.run_gate("X", [1, 99], simulate_gate=False) == {}
    assert state.run_gate("X", 0) == {}
    assert calls == []
    with pytest.raises(TypeError, match="not iterable"):
        state.run_gate("X", 1)
    assert state.run_gate("I", {1, 2, 3}) == {}
    with pytest.raises(ValueError, match="II"):
        state.run_gate("I", {(0, 1)})
    # The public engine also dispatches base-class user bindings directly and in order.
    pecos.HybridEngine.run_gate(state, {}, "X", [2, 1, 2])
    assert calls == [2, 1, 2]
    state.bindings["CX"] = lambda _state, location, **_params: calls.append(location)
    # The base loop accepts overlapping pairs and dispatches in container iteration order.
    pairs = {(0, 1), (1, 2)}
    assert state.run_gate("CX", pairs) == {}
    assert calls[:3] == [2, 1, 2]
    assert calls[3:] == list(pairs)


def test_default_simulator_identity_pair_future():
    state = DefaultSimulator()
    state.bindings["I"] = lambda _state, _location, **_params: None
    # Known-spelling overrides will inherit the single-qubit I policy in #1480.
    with pytest.raises(ValueError, match="II"):
        state.run_gate("I", {(0, 1)})


@pytest.mark.parametrize("state", WRAPPERS, indirect=True)
@pytest.mark.parametrize("container", CONTAINERS, ids=["list", "tuple", "set", "frozenset", "generator"])
def test_circuit_location_containers(state, container):
    _prepare_one(state)
    circuit = _items("Measure", container([1]))
    if isinstance(state, StateVec) and container in {list, tuple, iter}:
        # KNOWN-WRONG (#1480): StateVec.run_circuit raises TypeError for list/tuple/generator batches.
        with pytest.raises(TypeError, match="unsupported operand type"):
            state.run_circuit(circuit)
    else:
        # The other circuit paths accept each iterable without rebuilding it.
        assert state.run_circuit(circuit) == {1: 1}


@pytest.mark.parametrize("state", PAULI, indirect=True)
def test_pauli_measurement_list_key(state):
    # KNOWN-WRONG (#1480): PauliProp Measure [[1]] raises only when the measured outcome is one.
    assert state.run_gate("Measure", [[1]]) == {}
    state.faults = {"X": {1}, "Y": set(), "Z": set()}
    with pytest.raises(TypeError, match="unhashable type: 'list'"):
        state.run_gate("Measure", [[1]])


def test_cointoss_circuit_bare_int_and_disabled_measurement():
    state = CoinToss(2, prob=1, seed=42)
    # Bare zero measures in both routes; disabled measurements validate and skip.
    assert state.run_circuit(_items("Measure -Z", 0)) == {0: 1}
    assert state.run_gate("Measure -Z", 0) == {0: 1}
    assert state.run_circuit(_items("Measure -Z", {1}, simulate_gate=False)) == {}


@pytest.mark.parametrize("state", [*WRAPPERS, "DefaultSimulator"], indirect=True)
def test_unknown_nonempty_symbol(state):
    if isinstance(state, CoinToss) or type(state).__name__ == "CudaStateVec":
        # KNOWN-WRONG (#1480): CudaStateVec silently ignores unknown symbols (CoinToss gates are no-ops).
        assert state.run_gate("unknown-wrapper-symbol", {0}) == {}
    else:
        # DefaultSimulator and StateVec use a generic Exception for unsupported symbols.
        with pytest.raises(Exception, match="Gate unknown-wrapper-symbol is not supported in this simulator"):
            state.run_gate("unknown-wrapper-symbol", {0})


@pytest.mark.parametrize("state", PAULI, indirect=True)
def test_pauli_identity_frame_circuit_short_circuit(state):
    circuit = QuantumCircuit()
    circuit.append("unknown-wrapper-symbol", {0})
    # KNOWN-WRONG (#1480): PauliProp.run_circuit returns None and skips validation on an identity frame.
    assert state.run_circuit(circuit) is None
    state.faults = {"X": {3}, "Y": set(), "Z": set()}
    with pytest.raises(Exception, match="Gate unknown-wrapper-symbol is not supported"):
        state.run_circuit(circuit)


def test_error_set_string_rejection():
    # KNOWN-WRONG (#1480): ErrorSet rejects its advertised string input after the NumPy migration.
    with pytest.raises(TypeError, match="Arrays cannot contain string objects"):
        Generator.ErrorSet(["X"])


@pytest.mark.parametrize("state", [*WRAPPERS, "DefaultSimulator"], indirect=True)
@pytest.mark.parametrize("values", [[], [0, 1]])
def test_disabled_sized_batch(state, values, sized_sequence):
    assert state.run_gate("X", sized_sequence(values), simulate_gate=False) == {}


@pytest.mark.parametrize("state", ["SparseStabPy", "CuStateVec", "MPS", "CudaStateVec"], indirect=True)
@pytest.mark.parametrize("container", ["custom", "range"])
def test_sized_sequence_pair(state, container, sized_sequence):
    location = sized_sequence([0, 1]) if container == "custom" else range(2)
    state.run_gate("X", [0])
    assert state.run_gate("CX", [location]) == {}
    assert state.run_gate("Measure", [0, 1]) == {0: 1, 1: 1}


@pytest.mark.parametrize("identity", [True, False])
def test_pauli_list_locations_with_removal(identity):
    state = PauliProp(num_qubits=2)
    if not identity:
        state.faults = {"X": {0}, "Y": set(), "Z": set()}
    # Both frames use the existing set subtraction contract; inner lists are unhashable.
    with pytest.raises(TypeError, match="unhashable type: 'list'"):
        state.run_circuit(_items("CX", [[0, 1]]), removed_locations=set())


@pytest.mark.parametrize("symbol", ["RX", "RY", "RZ", "RXY1Q", "R1XY", "U1q"])
@pytest.mark.parametrize("shape", ["tuple", "list", "scalar"])
def test_sparse_rotation_location_protocol(symbol, shape, index_scalar):
    state, fresh = simulators.SparseStabPy(2), simulators.SparseStabPy(2)
    for sim in (state, fresh):
        sim.run_gate("H", [0, 1])
    location = {"tuple": (1,), "list": [1], "scalar": index_scalar(1)}[shape]
    params = {"angle": pi / 2} if symbol in {"RX", "RY", "RZ"} else {"angles": [pi / 2, 0]}
    state.run_gate(symbol, [0, location], **params)
    fresh.run_gate(symbol, [0, 1], **params)
    assert state.stabs.row_string() == fresh.stabs.row_string()
    assert state.destabs.row_string() == fresh.destabs.row_string()


# CoinToss ignores non-measurement symbols before inspecting locations.
@pytest.mark.parametrize("state", [*QUANTUM, *PAULI, "DefaultSimulator"], indirect=True)
def test_zero_dimensional_empty_batch(state, index_scalar):
    assert state.run_gate("unknown-private-gate", index_scalar(0)) == {}


@pytest.mark.parametrize("state", ["SparseStabPy", "CudaStateVec"], indirect=True)
@pytest.mark.parametrize("container", [set, frozenset])
def test_unordered_pair_rejected(state, container):
    with pytest.raises(ValueError, match="CX"):
        state.run_gate("CX", [container({0, 1})])


@pytest.mark.parametrize("state", QUANTUM, indirect=True)
@pytest.mark.parametrize("location", [(0,), [0]])
def test_identity_singleton_follows_state_policy(state, location):
    # "I" is a single-qubit gate: a singleton location follows the same
    # per-state policy as X; these wrappers take scalar locations only.
    with pytest.raises(ValueError, match="I: expected a scalar qubit index"):
        state.run_gate("I", [location])


@pytest.mark.parametrize("state", PAULI, indirect=True)
@pytest.mark.parametrize("location", [(0,), [0]])
def test_identity_singleton_accepted_where_singletons_are(state, location):
    assert state.run_gate("I", [location]) == {}


def test_hybrid_engine_bare_int_still_raises():
    # On dev the public HybridEngine iterated the batch and raised TypeError.
    with pytest.raises(TypeError):
        pecos.HybridEngine.run_gate(simulators.SparseStabPy(2), {}, "X", 0)


def test_pauli_alias_identity():
    assert simulators.PauliFaultProp is PauliProp


def test_registry_binding_shadows_registered_gate():
    state = simulators.SparseStabPy(2)
    registry = GateRegistry()
    registry.define("CUSTOM", 1).step("X", [0]).register_into(registry)
    state.bindings = dict(state.bindings)
    state.bindings["CUSTOM"] = state.bindings["H"]
    assert state.run_circuit(_items("CUSTOM", {0}), gate_registry=registry) == {}
    assert state.stabs.row_string() == ["  XI", "  IZ"]


def test_registry_location_order_angles_and_metadata():
    state = simulators.SparseStabPy(2)
    assert type(state).run_circuit is DefaultSimulator.run_circuit
    registry = GateRegistry()
    registry.define("CUSTOM", 1).angle_arity(1).step_with_metadata(
        "RX",
        [0],
        [AngleSource.input(0)],
        {"step": "first"},
    ).step_with_metadata("RZ", [0], [AngleSource.fixed(0.5)], {"step": "second"}).register_into(registry)
    calls = []
    state.bindings = dict(state.bindings)
    state.bindings["RX"] = lambda _s, q, **p: calls.append(("RX", q, p))
    state.bindings["RZ"] = lambda _s, q, **p: calls.append(("RZ", q, p))
    assert (
        state.run_circuit(
            _items("CUSTOM", [1, 0], angles=(0.25,), step="outer", tag="kept"),
            gate_registry=registry,
        )
        == {}
    )
    assert calls == [
        ("RX", 1, {"angles": (0.25,), "angle": 0.25, "step": "first", "tag": "kept"}),
        ("RZ", 1, {"angles": (0.5,), "angle": 0.5, "step": "second", "tag": "kept"}),
        ("RX", 0, {"angles": (0.25,), "angle": 0.25, "step": "first", "tag": "kept"}),
        ("RZ", 0, {"angles": (0.5,), "angle": 0.5, "step": "second", "tag": "kept"}),
    ]


def test_registry_measurement_steps_return_outputs():
    state = simulators.SparseStabPy(2)
    registry = GateRegistry()
    registry.define("CUSTOM", 1).step("X", [0]).step("Measure", [0]).register_into(registry)
    # Registry canonicalizes Measure to MZ; SparseStabPy requires an explicit MZ binding.
    state.bindings = dict(state.bindings)
    state.bindings["MZ"] = state.bindings["Measure"]
    assert state.run_circuit(_items("CUSTOM", [1, 0]), gate_registry=registry) == {1: 1, 0: 1}


def test_registry_late_failure_preserves_state():
    state = simulators.SparseStabPy(2)
    registry = GateRegistry()
    registry.define("CUSTOM", 1).step("X", [0]).step("T", [0]).register_into(registry)
    # Deliberate #1492 flip: reject the unbound step before applying X.
    with pytest.raises(ValueError, match=r"^Gate CUSTOM step T is not supported in this simulator\.$"):
        state.run_circuit(_items("CUSTOM", {0}), gate_registry=registry)
    assert state.run_gate("Measure", {0}) == {}


def test_statevec_registry_keyword_rejected():
    with pytest.raises(TypeError, match="unexpected keyword argument 'gate_registry'"):
        StateVec(2).run_circuit(_items("X", {0}), gate_registry=GateRegistry())


@pytest.mark.parametrize("state", WRAPPERS, indirect=True)
def test_legacy_tick_removed_locations(state):
    if isinstance(state, PauliProp):
        state.faults = {"X": set(), "Y": set(), "Z": {0, 1}}
    circuit = QuantumCircuit([{"H": {0, 1}}])
    [(tick, time, metadata)] = list(circuit.iter_ticks())
    assert time == 0
    assert metadata == {}
    if isinstance(state, CoinToss):
        with pytest.raises(TypeError, match="unexpected keyword argument 'removed_locations'"):
            state.run_circuit(tick, removed_locations={1})
    else:
        # Exactly the TickView call shape used by circuit_runners/standard.py.
        assert state.run_circuit(tick, removed_locations={1}) == {}
        if isinstance(state, PauliProp):
            assert state.faults == {"X": {0}, "Y": set(), "Z": {1}}
            assert state.sign == 0
        else:
            assert state.run_gate("H", {0}) == {}
            state.run_gate("X", {0, 1})
            assert state.run_gate("Measure", {0, 1}) == {0: 1, 1: 1}


def test_pauli_fault_circuit_from_noise_producer():
    state = PauliProp(num_qubits=2, track_sign=True)
    after = QuantumCircuit([{"X": {0}}, {"Z": {1}}])
    errors = ErrorCircuits()
    errors.add_circuits(0, after_faults=after)
    assert after.metadata == {"circuit_type": "faults"}
    # The noise producer selects add_faults -> native add_paulis, even on an identity frame.
    assert state.run_circuit(errors[0]["after"]) is None
    assert state.faults == {"X": {0}, "Y": set(), "Z": {1}}
    assert state.sign == 0
    assert state.run_circuit(errors[0]["after"]) is None
    assert state.faults == {"X": set(), "Y": set(), "Z": set()}


@pytest.mark.parametrize("backend", QUANTUM)
def test_noise_measurement_bitflips_through_quantum_simulator(backend):
    _require_backend(backend)
    sim = QuantumSimulator(backend="stabilizer" if backend == "SparseStabPy" else backend, seed=42)
    sim.init(2)
    noisy = noise_meas_bitflip(QOp("Measure", [0, 1], returns=[["m", 0], ["m", 1]]), p=1)
    assert len(noisy) == 1
    assert noisy[0].metadata == {"bitflips": [0, 1]}
    assert sim.run([QOp("X", [1]), *noisy]) == [{("m", 0): 1, ("m", 1): 0}]


@pytest.mark.parametrize("state", ["SparseStabPy", *PAULI], indirect=True)
def test_force_output_location_free(state):
    if isinstance(state, PauliProp):
        with pytest.raises(Exception, match="Gate force output is not supported in this simulator"):
            state.run_gate("force output", [300], forced_output=1)
        return
    assert state.run_gate("force output", [300], forced_output=1) == {300: 1}
    assert state.run_gate("force output", [300], forced_output=0) == {}
    assert state.run_gate("force output", [300]) == {300: -1}


@pytest.mark.parametrize("state", WRAPPERS, indirect=True)
def test_public_hybrid_engine_call_shapes(state):
    from pecos.exceptions import NotSupportedGateError

    output = {"m": pecos.BitInt(2, 0)}
    with pytest.raises(NotSupportedGateError, match='The gate "unknown-wrapper-symbol" is not available'):
        pecos.HybridEngine.run_gate(state, output, "unknown-wrapper-symbol", {0})
    assert (
        pecos.HybridEngine.run_gate(
            state,
            output,
            "unknown-wrapper-symbol",
            {0},
            simulate_gate=False,
        )
        is None
    )
    if isinstance(state, PauliProp):
        state.faults = {"X": set(), "Y": set(), "Z": {1}}
    pecos.HybridEngine.run_gate(state, output, "X", {1}, simulate_gate=False)
    if isinstance(state, PauliProp):
        assert state.faults == {"X": set(), "Y": set(), "Z": {1}}
        assert state.sign == 0
    elif isinstance(state, CoinToss):
        # Deliberate #1493 flip: run_gate now returns the coin result.
        assert state.run_gate("Measure", {1}) == {1: 1}
    else:
        assert state.run_gate("Measure", {1}) == ({1: 0} if type(state).__name__ == "CudaStateVec" else {})
    _prepare_one(state)
    pecos.HybridEngine.run_gate(state, output, "Measure", [1, 0], var_output={1: ("m", 0), 0: ("m", 1)})
    assert int(output["m"]) == (3 if isinstance(state, CoinToss) else 1)


@pytest.mark.parametrize("state", [*WRAPPERS, "DefaultSimulator"], indirect=True)
def test_disabled_unknown_symbol(state):
    assert state.run_gate("unknown-wrapper-symbol", [0, 99], simulate_gate=False) == {}


def test_error_set_on_pauli_fault_path():
    state = PauliProp(num_qubits=2)
    after = QuantumCircuit()
    Generator.ErrorSet([pecos.Pauli.X]).error_func(after, QuantumCircuit(), set(), 1, {})
    assert list(after.items()) == [("X", {1}, {})]
    errors = ErrorCircuits()
    errors.add_circuits(0, after_faults=after)
    assert state.run_circuit(errors[0]["after"]) is None
    assert state.faults == {"X": {1}, "Y": set(), "Z": set()}
    assert state.run_gate("Measure", {0, 1}) == {1: 1}


@pytest.mark.parametrize("state", ["StateVec", "SparseStabPy", *PAULI], indirect=True)
def test_error_set_two_qubit_tensor_product(state):
    after = QuantumCircuit()
    Generator.ErrorSetTwoQuditTensorProduct([(pecos.Pauli.X, pecos.Pauli.Z)]).error_func(
        after,
        QuantumCircuit(),
        set(),
        (0, 1),
        {},
    )
    assert list(after.items()) == [("X", {0}, {}), ("Z", {1}, {})]
    if isinstance(state, PauliProp):
        errors = ErrorCircuits()
        errors.add_circuits(0, after_faults=after)
        assert state.run_circuit(errors[0]["after"]) is None
        assert state.faults == {"X": {0}, "Y": set(), "Z": {1}}
    else:
        state.run_gate("H", {1})
        assert state.run_circuit(after) == {}
        state.run_gate("H", {1})
        assert state.run_gate("Measure", {0, 1}) == {0: 1, 1: 1}


@pytest.mark.parametrize("state", [*QUANTUM, *PAULI], indirect=True)
@pytest.mark.parametrize("symbol", ["Measure +X", "Measure +Y", "measure X", "measure Y"])
def test_measurement_basis_aliases(state, symbol):
    if isinstance(state, PauliProp):
        state.faults = {"X": set(), "Y": set(), "Z": {0}}
    else:
        state.run_gate("H", {0})
        state.run_gate("Z" if symbol.endswith("X") else "SZdg", {0})
    name = type(state).__name__
    if name in {"CuStateVec", "MPS"} or (name == "StateVec" and symbol.startswith("measure")):
        with pytest.raises(Exception, match="is not supported in this simulator"):
            state.run_gate(symbol, {0})
    elif name == "CudaStateVec" and symbol.startswith("Measure"):
        # KNOWN-WRONG (#1480): CudaStateVec silently ignores unsupported measurement aliases.
        assert state.run_gate(symbol, {0}) == {}
    else:
        assert state.run_gate(symbol, {0}) == {0: 1}


@pytest.mark.parametrize("state", [*QUANTUM, *PAULI], indirect=True)
@pytest.mark.parametrize("symbol", ["Init", "init |1>", "CNOT", "SqrtXX", "R1XY", "R2XXYYZZ"])
def test_additional_gate_aliases(state, symbol):
    name = type(state).__name__
    if isinstance(state, PauliProp):
        state.faults = {"X": set(), "Y": set(), "Z": {0}}
    elif symbol in {"Init", "CNOT"}:
        state.run_gate("X", {0})
    locations = {(0, 1)} if symbol in {"CNOT", "SqrtXX", "R2XXYYZZ"} else {0}
    params = {"angles": (pi, 0.0)} if symbol == "R1XY" else {}
    if symbol == "R2XXYYZZ":
        params = {"angles": (pi, 0.0, 0.0)}
    if (name == "SparseStabPy" and symbol == "R2XXYYZZ") or (
        name in {"CuStateVec", "MPS", "CudaStateVec"} and symbol == "SqrtXX"
    ):
        if name == "CudaStateVec":
            # KNOWN-WRONG (#1480): CudaStateVec ignores unbound gate aliases instead of rejecting them.
            assert state.run_gate(symbol, locations, **params) == {}
        else:
            with pytest.raises(Exception, match="is not supported in this simulator"):
                state.run_gate(symbol, locations, **params)
        return
    if name == "CudaStateVec" and symbol == "R2XXYYZZ":
        # KNOWN-WRONG (#1480): CudaStateVec silently ignores the unbound R2XXYYZZ alias.
        assert state.run_gate(symbol, locations, **params) == {}
        return
    assert state.run_gate(symbol, locations, **params) == {}
    if symbol == "SqrtXX":
        assert state.run_gate(symbol, locations) == {}
    if isinstance(state, PauliProp):
        assert state.faults == (
            {"X": set(), "Y": set(), "Z": set()}
            if symbol in {"Init", "init |1>"}
            else {"X": set(), "Y": set(), "Z": {0}}
        )
        assert state.sign == (1 if symbol in {"SqrtXX", "R1XY", "R2XXYYZZ"} else 0)
    else:
        expected = {
            "Init": {},
            "init |1>": {0: 1},
            "CNOT": {0: 1, 1: 1},
            "SqrtXX": {0: 1, 1: 1},
            "R1XY": {0: 1},
            "R2XXYYZZ": {0: 1, 1: 1},
        }[symbol]
        if name == "CudaStateVec":
            expected = {"Init": {0: 0, 1: 0}, "init |1>": {0: 1, 1: 0}, "CNOT": {0: 1, 1: 1}, "R1XY": {0: 1, 1: 0}}[
                symbol
            ]
        assert state.run_gate("Measure", [0, 1]) == expected


@pytest.mark.parametrize("state", ["StateVec", "SparseStabPy"], indirect=True)
def test_negative_z_measurement_rejected(state):
    with pytest.raises(Exception, match=r"^Gate Measure -Z is not supported in this simulator\.$"):
        state.run_gate("Measure -Z", {0})


@pytest.mark.parametrize("symbol", ["U1q", "RXY1Q"])
def test_statevec_xy_rotation_missing_angles(symbol):
    state = StateVec(2)
    with pytest.raises(KeyError, match=r"^'angles'$"):
        state.run_gate(symbol, {0})
    assert list(state.vector) == [1, 0, 0, 0]


def test_pecos_integer_scalar():
    # PECOS exposes integer array elements as Python ints, not a separate scalar class.
    location = pecos.array([300])[0]
    assert type(location) is int
    assert location == 300
    state = PauliProp(num_qubits=2)
    state.faults = {"X": {300}, "Y": set(), "Z": set()}
    result = state.run_gate("Measure", [location])
    assert result == {300: 1}
    assert next(iter(result)) is location


def test_binding_mutation_survives_result_key_failure():
    state = simulators.SparseStabPy(2)
    state.bindings = dict(state.bindings)

    def mutate_then_return_one(sim, _location, **_params):
        sim.run_gate("X", {0})
        return 1

    state.bindings["CX"] = mutate_then_return_one
    # KNOWN-WRONG (#1480): Result insertion raises only after the overridden binding changes the state.
    with pytest.raises(TypeError, match=r"^unhashable type: 'list'$"):
        state.run_gate("CX", [[0, 1]])
    assert state.run_gate("Measure", {0}) == {0: 1}


def test_registry_measurement_canonical_symbol_unbound():
    state = simulators.SparseStabPy(2)
    registry = GateRegistry()
    registry.define("CUSTOM", 1).step("X", [0]).step("Measure", [0]).register_into(registry)
    # Deliberate #1492 flip: canonical MZ is checked before applying X.
    with pytest.raises(ValueError, match=r"^Gate CUSTOM step MZ is not supported in this simulator\.$"):
        state.run_circuit(_items("CUSTOM", {0}), gate_registry=registry)
    assert state.run_gate("Measure", {0}) == {}
