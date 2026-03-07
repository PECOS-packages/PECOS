"""Basic tests for pecos_rslib_cuda Python bindings."""

import types

import pytest


@pytest.fixture
def pecos_rslib_cuda() -> types.ModuleType:
    """Import the module, skip if not available."""
    try:
        import pecos_rslib_cuda

        return pecos_rslib_cuda
    except ImportError:
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


@pytest.mark.cuda
def test_custabilizer_creation(pecos_rslib_cuda) -> None:
    """Test CuStabilizer creation (requires CUDA)."""
    if not pecos_rslib_cuda.is_cuquantum_available():
        pytest.skip("cuQuantum not available")

    try:
        sim = pecos_rslib_cuda.CuStabilizer(100)
        assert sim.num_qubits == 100
    except RuntimeError as e:
        if "not supported" in str(e).lower() or "API changed" in str(e):
            pytest.skip("CuStabilizer API changed in cuQuantum 25.11+")
        raise


@pytest.mark.cuda
def test_custabilizer_ghz_state(pecos_rslib_cuda) -> None:
    """Test creating a GHZ state with stabilizer (requires CUDA)."""
    if not pecos_rslib_cuda.is_cuquantum_available():
        pytest.skip("cuQuantum not available")

    try:
        n = 10
        sim = pecos_rslib_cuda.CuStabilizer(n)

        # Create GHZ state: H on first qubit, then CX chain
        sim.h([0])
        for i in range(n - 1):
            sim.cx([i, i + 1])

        # All qubits should be correlated in measurement
        results = sim.mz(list(range(n)))

        # All should be the same value
        assert all(r == results[0] for r in results)
    except RuntimeError as e:
        if "not supported" in str(e).lower() or "API changed" in str(e):
            pytest.skip("CuStabilizer API changed in cuQuantum 25.11+")
        raise
