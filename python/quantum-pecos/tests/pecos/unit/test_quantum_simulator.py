"""CPU coverage for backend selection, lazy loading, and constructor contracts."""

from __future__ import annotations

import inspect
from unittest.mock import Mock

import pytest
from pecos import simulators
from pecos.simulators import quantum_simulator as dispatch

BACKENDS = [
    (None, "SparseStabPy", False),
    ("stabilizer", "SparseStabPy", False),
    ("state-vector", "StateVec", True),
    ("StateVec", "StateVec", True),
    ("MPS", "MPS", True),
    ("mps", "MPS", True),
    ("CuStateVec", "CuStateVec", False),
    ("CudaStateVec", "CudaStateVec", True),
]


@pytest.mark.parametrize(("backend", "class_name", "takes_seed"), BACKENDS)
def test_backend_constructor_contract(monkeypatch, backend, class_name, takes_seed) -> None:
    """Every exact name selects its constructor and forwards only supported seeds."""
    constructor = Mock()
    if class_name == "CudaStateVec":
        monkeypatch.setitem(vars(simulators), class_name, constructor)
    else:
        monkeypatch.setattr(dispatch, class_name, constructor)
    sim = dispatch.QuantumSimulator(backend, seed=7, extra="preserved")
    sim.init(2)
    kwargs = {"num_qubits": 2, "extra": "preserved"}
    if takes_seed:
        kwargs["seed"] = 7
    constructor.assert_called_once_with(**kwargs)
    assert sim.state is constructor.return_value
    assert sim.qsim_params == {"seed": 7, "extra": "preserved"}


@pytest.mark.parametrize("backend", [None, "stabilizer", "state-vector", "StateVec"])
def test_cpu_backend_class(backend) -> None:
    """CPU entries construct the public simulator class."""
    sim = dispatch.QuantumSimulator(backend)
    sim.init(2)
    expected = dispatch.SparseStabPy if backend in {None, "stabilizer"} else dispatch.StateVec
    assert isinstance(sim.state, expected)


@pytest.mark.parametrize("backend", ["", "e", "vector", "Qulacs", "CudaStabilizer", object(), [], 3])
def test_invalid_backend_does_not_load_cuda(monkeypatch, backend) -> None:
    """Rejected names and objects fail at construction, before any lazy load."""
    load = Mock(side_effect=AssertionError("CUDA must stay lazy"))
    monkeypatch.setattr(dispatch.importlib, "import_module", load)
    with pytest.raises(ValueError, match="accepted backends") as error:
        dispatch.QuantumSimulator(backend)
    for name, _, _ in BACKENDS:
        assert repr(name) in str(error.value)
    load.assert_not_called()


@pytest.mark.parametrize(
    ("backend", "dependency"),
    [
        ("MPS", "pytket-cutensornet"),
        ("mps", "pytket-cutensornet"),
        ("CuStateVec", "CuPy"),
        ("CudaStateVec", "pecos-rslib-cuda"),
    ],
)
def test_unavailable_backend(monkeypatch, backend, dependency) -> None:
    """Missing optional classes name both the backend and its dependency."""
    if backend == "CudaStateVec":
        monkeypatch.setitem(vars(simulators), backend, None)
    else:
        monkeypatch.setattr(dispatch, "MPS" if backend == "mps" else backend, None)
    with pytest.raises(ImportError, match=dependency) as error:
        dispatch.QuantumSimulator(backend).init(2)
    assert repr(backend) in str(error.value)


@pytest.mark.parametrize("backend", ["StateVec", "MPS", "mps", "CudaStateVec"])
def test_seed_reaches_loader_constructor(monkeypatch, backend) -> None:
    """Seed capability is honored even when a loader supplies another constructor."""
    _, takes_seed, dependency = vars(dispatch)["_BACKENDS"][backend]
    constructor = Mock()
    monkeypatch.setitem(vars(dispatch)["_BACKENDS"], backend, (lambda: constructor, takes_seed, dependency))
    dispatch.QuantumSimulator(backend, seed=42).init(2)
    constructor.assert_called_once_with(num_qubits=2, seed=42)


def test_constructor_typeerror_is_not_retried(monkeypatch) -> None:
    """An internal seed-related TypeError must propagate without a second call."""
    error = TypeError("inner() got an unexpected keyword argument 'seed'")
    constructor = Mock(side_effect=error)
    monkeypatch.setitem(vars(dispatch)["_BACKENDS"], "StateVec", (lambda: constructor, True, None))
    with pytest.raises(TypeError) as caught:
        dispatch.QuantumSimulator("StateVec", seed=7).init(2)
    assert caught.value is error
    constructor.assert_called_once_with(num_qubits=2, seed=7)


def test_custatevec_actionable_runtime_error_is_preserved(monkeypatch) -> None:
    """Dependency checks inside CuStateVec retain their original RuntimeError."""
    from pecos.simulators.custatevec import state

    error = RuntimeError("CuStateVec requires CuPy and cuQuantum >= 25.03")
    monkeypatch.setattr(state, "require_custatevec", Mock(side_effect=error))
    with pytest.raises(RuntimeError) as caught:
        dispatch.QuantumSimulator("CuStateVec", seed=7).init(2)
    assert caught.value is error


def test_cuda_loader_is_lazy(monkeypatch) -> None:
    """Construction and CPU initialization do not resolve the CUDA class."""
    constructor = Mock()
    load = Mock(return_value=Mock(CudaStateVec=constructor))
    monkeypatch.setattr(dispatch.importlib, "import_module", load)
    sim = dispatch.QuantumSimulator("CudaStateVec", seed=7)
    dispatch.QuantumSimulator("StateVec").init(2)
    load.assert_not_called()
    sim.init(2)
    load.assert_called_once_with("pecos.simulators")
    constructor.assert_called_once_with(num_qubits=2, seed=7)


def test_cuda_import_failure_yields_named_error(monkeypatch) -> None:
    """Exercise the package's ImportError-to-None path through the lazy loader."""
    monkeypatch.setitem(vars(simulators), "CudaStateVec", None)
    monkeypatch.delitem(vars(simulators), "CudaStateVec")
    monkeypatch.setattr(simulators, "import_module", Mock(side_effect=ImportError("missing extension")))
    with pytest.raises(ImportError, match=r"CudaStateVec.*pecos-rslib-cuda"):
        dispatch.QuantumSimulator("CudaStateVec").init(2)


@pytest.mark.parametrize("backend", list(dispatch._BACKENDS))
def test_seed_flag_matches_real_constructor(backend) -> None:
    """Each table entry's seed flag agrees with the real class signature."""
    loader, takes_seed, _ = dispatch._BACKENDS[backend]
    cls = loader()
    if cls is None:
        pytest.skip(f"{backend!r} is unavailable on this machine")
    params = inspect.signature(cls).parameters.values()
    accepts_seed = any(
        p.name == "seed" or p.kind is inspect.Parameter.VAR_KEYWORD for p in params
    )
    assert accepts_seed == takes_seed
