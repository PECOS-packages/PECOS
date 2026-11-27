"""Basic tests for pecos_rslib_cuda Python bindings."""

import types

import pytest


@pytest.fixture
def pecos_rslib_cuda() -> types.ModuleType:
    """Import the module, skip if not available."""
    try:
        import pecos_rslib_cuda

        return pecos_rslib_cuda
    except ModuleNotFoundError as error:
        if error.name != "pecos_rslib_cuda":
            raise
        pytest.skip("pecos_rslib_cuda not installed")


def test_version(pecos_rslib_cuda) -> None:
    """Test that version is accessible."""
    assert hasattr(pecos_rslib_cuda, "__version__")
    assert isinstance(pecos_rslib_cuda.__version__, str)


def test_is_cuquantum_available(pecos_rslib_cuda) -> None:
    """Test availability check function."""
    result = pecos_rslib_cuda.is_cuquantum_available()
    assert isinstance(result, bool)


@pytest.mark.cuda
def test_custatevec_creation(pecos_rslib_cuda) -> None:
    """Test CuStateVec creation (requires CUDA)."""
    if not pecos_rslib_cuda.is_cuquantum_available():
        pytest.skip("cuQuantum not available")

    sim = pecos_rslib_cuda.CuStateVec(4)
    assert sim.num_qubits == 4


@pytest.mark.cuda
def test_custatevec_bell_state(pecos_rslib_cuda) -> None:
    """Test creating a Bell state (requires CUDA)."""
    if not pecos_rslib_cuda.is_cuquantum_available():
        pytest.skip("cuQuantum not available")

    sim = pecos_rslib_cuda.CuStateVec(2)
    sim.h([0])
    sim.cx([0, 1])

    # Measure multiple times and check correlations
    correlations = 0
    trials = 100
    for _ in range(trials):
        sim.reset()
        sim.h([0])
        sim.cx([0, 1])
        results = sim.mz([0, 1])
        # In a Bell state, qubits should always be correlated
        if results[0] == results[1]:
            correlations += 1

    # Should be 100% correlated
    assert correlations == trials


@pytest.mark.parametrize(
    ("name", "args"),
    [("CuStateVec", (4,)), ("CuTensorNet", ()), ("CuDensityMat", (2,))],
)
def test_constructors_report_missing_sdk(pecos_rslib_cuda, name, args) -> None:
    """Without the cuQuantum SDK, every simulator constructor fails with the not-available error."""
    if pecos_rslib_cuda.is_cuquantum_available():
        pytest.skip("cuQuantum is available, so the missing-SDK error cannot occur")

    with pytest.raises(RuntimeError, match="cuQuantum SDK not available"):
        getattr(pecos_rslib_cuda, name)(*args)


@pytest.mark.cuda
@pytest.mark.parametrize(
    ("gate", "args", "qubits"),
    [
        (name, (), [0, 0])
        for name in [
            "h",
            "x",
            "y",
            "z",
            "s",
            "sdg",
            "sx",
            "sxdg",
            "sy",
            "sydg",
            "sz",
            "szdg",
            "f",
            "fdg",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "t",
            "tdg",
            "mx",
            "mnx",
            "my",
            "mny",
            "mz",
        ]
    ]
    + [
        (name, (), [1, 2, 0, 1])
        for name in [
            "cx",
            "cy",
            "cz",
            "swap",
            "iswap",
            "sxx",
            "sxxdg",
            "syy",
            "syydg",
            "szz",
            "szzdg",
            "g",
        ]
    ]
    + [(name, (0.25,), [1, 2, 0, 1]) for name in ["rx", "ry", "rz", "rxx", "ryy", "rzz", "crx", "cry", "crz"]]
    + [
        ("rxyxy2q", (0.25, 0.5), [1, 2, 0, 1]),
        ("rxy1q", (0.25, 0.5), [0, 0]),
        ("u", (0.25, 0.5, 0.75), [0, 0]),
    ],
)
def test_custatevec_rejects_overlapping_batches(pecos_rslib_cuda, gate, args, qubits):
    if not pecos_rslib_cuda.is_custatevec_usable():
        pytest.skip("cuStateVec GPU backend not available")
    sim = pecos_rslib_cuda.CuStateVec(3)
    sim.x([0])
    with pytest.raises(ValueError, match="requires distinct qubits") as error:
        getattr(sim, gate)(*args, qubits)
    duplicate = 0 if qubits == [0, 0] else 1
    assert str(error.value) == f"{gate.upper()} requires distinct qubits; duplicated qubits: [{duplicate}]"
    assert sim.mz([0, 1, 2]) == [1, 0, 0]
