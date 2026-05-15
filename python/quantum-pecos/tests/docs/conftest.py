"""Pytest configuration for documentation tests.

This module provides fixtures and utilities for testing documentation code examples.
"""

import subprocess
import sys

import pytest

# Cache CUDA availability
_CUDA_AVAILABLE: bool | None = None


def _check_cuda_available() -> bool:
    """Check if CUDA is available for running GPU examples.

    Uses the same pattern as the Justfile: `pecos cuda check -q` for toolkit,
    plus cupy availability check for Python CUDA packages.
    """
    # Check for CUDA toolkit using pecos CLI (same as Justfile pattern)
    try:
        result = subprocess.run(
            ["cargo", "run", "-p", "pecos-cli", "--quiet", "--", "cuda", "check", "-q"],
            capture_output=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False

    # Check for cupy Python package (needed for Python CUDA examples)
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import cupy; print(cupy.cuda.is_available())"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if result.returncode != 0 or "True" not in result.stdout:
            return False
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.SubprocessError):
        return False

    return True


def cuda_available() -> bool:
    """Return cached CUDA availability status."""
    global _CUDA_AVAILABLE  # noqa: PLW0603
    if _CUDA_AVAILABLE is None:
        _CUDA_AVAILABLE = _check_cuda_available()
    return _CUDA_AVAILABLE


@pytest.fixture(scope="session")
def cuda_check() -> bool:
    """Fixture that returns CUDA availability."""
    return cuda_available()


@pytest.fixture(autouse=True)
def restore_cwd():
    """Restore the current working directory after each test.

    Some tests (e.g., WASM examples) change the working directory,
    which can interfere with other tests that rely on path resolution.
    """
    from pathlib import Path

    original_cwd = Path.cwd()
    yield
    import os

    os.chdir(original_cwd)


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU")
    config.addinivalue_line("markers", "cuda: marks tests as requiring CUDA")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:  # noqa: ARG001
    """Print CUDA status at collection time."""
    cuda = cuda_available()
    print(f"\nCUDA available: {cuda}")
