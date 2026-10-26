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

"""Whole-batch validation preserves simulator state, RNG position and dispatch."""

import ast
import importlib.util
import random
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pecos
import pytest
from pecos import simulators
from pecos.simulators import DefaultSimulator, PauliProp, StateVec
from pecos.simulators.batch_validation import GATE_ARITIES, replayable_locations, validate_batch
from pecos.simulators.gate_syms import alt_symbols
from pecos_rslib import _gate_bindings_symbols

QUANTUM = ("StateVec", "SparseStabPy", "CuStateVec", "MPS", "CudaStateVec")
PAULI = ("PauliProp", "PauliFaultProp")


def _items(symbol, locations, **params):
    return SimpleNamespace(items=lambda: iter([(symbol, locations, params)]), metadata={})


def _require_backend(name):
    dependencies = {
        "CuStateVec": ("cupy", "cuquantum"),
        "MPS": ("cupy", "cuquantum", "pytket", "pytket.extensions.cutensornet"),
        "CudaStateVec": ("pecos_rslib_cuda",),
    }
    for dependency in dependencies.get(name, ()):
        if importlib.util.find_spec(dependency) is None:
            pytest.skip(f"{name}: {dependency} is not installed")
    if name == "CudaStateVec":
        from pecos_rslib_cuda import is_cuquantum_available

        if not is_cuquantum_available():
            pytest.skip("CudaStateVec: cuQuantum SDK is not installed (stub build)")


class _OrderedLocations(list):
    """Preserve order while supporting StateVec.run_circuit's set subtraction."""

    def __sub__(self, removed):
        return [location for location in self if location not in removed]


def _dispatch(state, route, symbol, locations, output, **params):
    if route == "hybrid":
        pecos.HybridEngine.run_gate(state, output, symbol, locations, **params)
    elif route == "circuit":
        state.run_circuit(_items(symbol, _OrderedLocations(locations), **params))
    else:
        state.run_gate(symbol, locations, **params)


def _make(name):
    _require_backend(name)
    params = {"num_qubits": 4}
    if name in {"StateVec", "MPS", "CudaStateVec"}:
        params["seed"] = 42
    if name in PAULI:
        params["track_sign"] = True
    return getattr(simulators, name)(**params)


def _snapshot(state):
    if isinstance(state, PauliProp):
        return state.faults, state.sign, state.img
    if type(state).__name__ == "SparseStabPy":
        return state.stabs.row_string(), state.destabs.row_string()
    if type(state).__name__ == "CudaStateVec":
        # The CUDA binding exposes sampling, but no amplitude/state accessor.
        return state.sample(128)
    return list(state.vector)


def _prepare(state, symbol):
    if isinstance(state, PauliProp):
        state.faults = {"X": {0}, "Y": set(), "Z": set()}
    else:
        state.run_gate("H" if symbol == "Measure" else "X", [0])


def _rng_probe(state):
    # Noncommuting measurements expose a consumed backend RNG draw even when a
    # failed gate's final state could happen to equal the initial state.
    results = []
    if not isinstance(state, PauliProp):
        for _ in range(32):
            state.run_gate("H", [0])
            results.append(state.run_gate("Measure", [0]))
    # SparseStabPy/CuStateVec use this shared RNG; PauliProp consumes no RNG.
    return results, list(pecos.random.random(16))


@pytest.mark.parametrize("name", [*QUANTUM, *PAULI])
@pytest.mark.parametrize("route", ["gate", "hybrid", "circuit"])
@pytest.mark.parametrize(("symbol", "bad"), [("Measure", -1), ("CX", (2, 2)), ("CX", (2,)), ("CX", (2, 99))])
def test_bad_tail_preserves_state_and_rng(name, route, symbol, bad):
    if name in PAULI and bad == (2, 99):
        bad = (2, -1)  # Growable frames deliberately accept large positive indices.
    pecos.random.seed(721)
    state = _make(name)
    _prepare(state, symbol)
    first = 0 if symbol == "Measure" else (0, 1)
    locations = [first, bad]
    output = {"m": pecos.BitInt(2, 3)}
    with pytest.raises(ValueError, match=symbol):
        _dispatch(state, route, symbol, locations, output, var=("m", 0))
    snapshot = _snapshot(state)
    observed_rng = _rng_probe(state)
    assert int(output["m"]) == 3

    pecos.random.seed(721)
    fresh = _make(name)
    _prepare(fresh, symbol)
    if isinstance(fresh, PauliProp) or name == "SparseStabPy":
        assert snapshot == _snapshot(fresh)
    else:
        assert snapshot == pytest.approx(_snapshot(fresh))
    assert observed_rng == _rng_probe(fresh)


@pytest.mark.parametrize(("symbol", "locations"), [("X", [0, -1]), ("CX", [(0, 1), (2, 2)]), ("I", [(0, 1)])])
@pytest.mark.parametrize("disabled", [False, True])
def test_identity_frame_validates_before_return(symbol, locations, disabled):
    state = PauliProp(num_qubits=1)
    with pytest.raises(ValueError, match=symbol):
        state.run_circuit(_items(symbol, iter(locations), simulate_gate=not disabled))
    assert state.faults == {"X": set(), "Y": set(), "Z": set()}


def test_identity_frame_removed_locations_and_growth():
    state = PauliProp(num_qubits=1)
    assert state.run_circuit(_items("CX", [(0, 99)])) is None
    assert state.run_circuit(_items("X", {0, -1}), removed_locations={-1}) is None
    with pytest.raises(ValueError, match="X"):
        state.run_circuit(_items("X", {0, -1}), removed_locations={0})


@pytest.mark.parametrize(
    ("symbol", "locations", "capacity"),
    [
        ("X", [(0, 1)], 4),
        ("CX", [0], 4),
        ("CX", [(0,)], 4),
        ("CX", [(0, 1, 2)], 4),
        ("X", ["0"], 4),
        ("X", [0.5], 4),
        ("X", [-1], None),
        ("X", [4], 4),
        ("CX", [(0, 4)], 4),
        ("CX", [(0, 0)], None),
        ("I", [(0, 1)], 4),
    ],
)
def test_validation_guards(symbol, locations, capacity):
    with pytest.raises(ValueError, match="II" if symbol == "I" else symbol):
        validate_batch(symbol, locations, capacity)


def test_validation_only_checks_and_preserves_containers():
    pair = [0, 1]
    locations = [pair, pair, [1, 2]]
    assert replayable_locations(locations) is locations
    assert validate_batch("CX", locations, 3) is None
    assert locations == [[0, 1], [0, 1], [1, 2]]
    assert locations[0] is pair
    assert locations[1] is pair
    buffered = replayable_locations(iter(locations))
    assert list(buffered) == locations
    assert next(iter(buffered)) is pair
    validate_batch("X", [0, (1,), [2], 999], None)
    validate_batch("unknown-private-gate", [object()], 0)
    validate_batch("force output", [object(), -5, (1, 1)], 0)


def test_all_bound_spellings_have_arities():
    # Inspect literal binding keys too, without importing optional GPU libraries.
    directory = Path(simulators.__file__).parent
    for path in [*directory.glob("*/bindings.py"), directory / "cointoss/state.py"]:
        if path.parent.name == "pauliprop":
            # This legacy module is unused: current PauliProp binds the native
            # table plus alt_symbols, checked below.
            continue
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values, strict=True):
                    if (
                        isinstance(key, ast.Constant)
                        and isinstance(key.value, str)
                        and isinstance(value, (ast.Name, ast.Attribute, ast.Lambda))
                    ):
                        assert key.value in GATE_ARITIES, (path, key.value)
    for symbol, _measurement, _parameters, arity in _gate_bindings_symbols():
        assert GATE_ARITIES[symbol] == arity
    for alias, target in alt_symbols.items():
        assert GATE_ARITIES[alias] == GATE_ARITIES[target]
    for symbol in ("Measure -X", "Measure -Y", "Measure -Z", "U1q"):
        assert GATE_ARITIES[symbol] == 1
    for symbol in ("MSYY", "MSZZ", "ZZ", "ZZMax", "ZZPhase"):
        assert GATE_ARITIES[symbol] == 2
    assert GATE_ARITIES["force output"] == 0


@pytest.fixture
def cuda_wrapper_class(monkeypatch):
    # Exercise the actual Python dispatch method even on CPU-only CI, with
    # overridden bindings. Native GPU integration remains covered above.
    module = ModuleType("pecos_rslib_cuda")
    module.CuStateVec = object
    monkeypatch.setitem(sys.modules, "pecos_rslib_cuda", module)
    # Avoid caching a stub-backed real CUDA package that could poison later
    # integration tests in the same worker on a machine with GPU dependencies.
    package = ModuleType("pecos.simulators.cuda_statevec")
    package.bindings = SimpleNamespace(gate_dict={})
    monkeypatch.setitem(sys.modules, "pecos.simulators.cuda_statevec", package)
    path = Path(simulators.__file__).parent / "cuda_statevec/state.py"
    spec = importlib.util.spec_from_file_location("_batch_test_cuda_state", path)
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded.CudaStateVec


@pytest.mark.parametrize("route", ["default", "statevec", "cuda", "hybrid"])
@pytest.mark.parametrize("disabled", [False, True])
@pytest.mark.parametrize(("symbol", "bad"), [("X", 4), ("X", -1), ("X", (0, 1)), ("X", "bad"), ("CX", (2, 2))])
def test_overridden_binding_guards_are_atomic(route, disabled, symbol, bad, cuda_wrapper_class):
    cls = {"default": DefaultSimulator, "statevec": StateVec, "cuda": cuda_wrapper_class, "hybrid": DefaultSimulator}[
        route
    ]
    state = cls.__new__(cls)
    state.gate_capacity = 4
    state.rng = random.Random(42)
    fresh_rng = random.Random(42)
    state.events = []

    def mutate(_state, location, **_params):
        state.events.append((location, state.rng.random()))

    state.bindings = {symbol: mutate}
    locations = iter([0 if symbol == "X" else (0, 1), bad])
    with pytest.raises(ValueError, match=symbol):
        _dispatch(state, route, symbol, locations, {}, simulate_gate=not disabled)
    assert state.events == []
    assert state.rng.getstate() == fresh_rng.getstate()


@pytest.mark.parametrize("route", ["default", "statevec", "cuda", "hybrid"])
def test_repeated_overlapping_overrides_dispatch_in_order(route, cuda_wrapper_class):
    cls = {"default": DefaultSimulator, "statevec": StateVec, "cuda": cuda_wrapper_class, "hybrid": DefaultSimulator}[
        route
    ]
    state = cls.__new__(cls)
    state.gate_capacity = 3
    calls = []
    state.bindings = {"CX": lambda _state, location, **_params: calls.append(location)}
    locations = [(1, 2), (0, 1), (1, 2)]
    if route == "hybrid":
        pecos.HybridEngine.run_gate(state, {}, "CX", iter(locations))
    else:
        state.run_gate("CX", iter(locations))
    assert calls == locations


def test_growable_hybrid_never_uses_display_width():
    state = PauliProp(num_qubits=1)
    state.faults = {"X": {0}, "Y": set(), "Z": set()}
    pecos.HybridEngine.run_gate(state, {}, "CX", [(0, 99)])
    assert state.faults == {"X": {0, 99}, "Y": set(), "Z": set()}


@pytest.mark.parametrize("route", ["default", "statevec", "cuda", "hybrid"])
@pytest.mark.parametrize("locations", [0, set()])
def test_empty_short_circuit(route, locations, cuda_wrapper_class):
    cls = {
        "default": DefaultSimulator,
        "statevec": StateVec,
        "cuda": cuda_wrapper_class,
        "hybrid": DefaultSimulator,
    }[route]
    state = cls.__new__(cls)
    state.gate_capacity = 0
    state.bindings = {}
    for symbol in ("X", "unknown-private-gate", "Tick"):
        _dispatch(state, route, symbol, locations, {})


def test_hybrid_cointoss_preserves_output_and_rng():
    state = simulators.CoinToss(4, prob=0.5, seed=42)
    fresh = simulators.CoinToss(4, prob=0.5, seed=42)
    output = {"m": pecos.BitInt(2, 3)}
    with pytest.raises(ValueError, match="Measure"):
        pecos.HybridEngine.run_gate(state, output, "Measure", [0, 4], var=("m", 0))
    assert int(output["m"]) == 3
    assert [state.bindings["Measure"](state, 0) for _ in range(32)] == [
        fresh.bindings["Measure"](fresh, 0) for _ in range(32)
    ]
