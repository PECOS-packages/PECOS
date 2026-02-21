"""Pytest configuration for documentation tests.

This module provides fixtures and utilities for testing documentation code examples.
"""

from __future__ import annotations

import functools
import shutil
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

    from _pytest.config import Config


def _check_cuda_available() -> bool:
    """Check if CUDA is available for running GPU examples.

    Uses the same pattern as the Justfile: `pecos cuda check -q` for toolkit,
    plus cupy availability check for Python CUDA packages.
    """
    # Check for CUDA toolkit using pecos CLI (same as Justfile pattern)
    cargo_path = shutil.which("cargo")
    if cargo_path is None:
        return False

    try:
        result = subprocess.run(
            [
                cargo_path,
                "run",
                "-p",
                "pecos",
                "--features",
                "cli",
                "--",
                "cuda",
                "check",
                "-q",
            ],
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


@functools.lru_cache(maxsize=1)
def cuda_available() -> bool:
    """Return cached CUDA availability status."""
    return _check_cuda_available()


@pytest.fixture(scope="session")
def cuda_check() -> bool:
    """Fixture that returns CUDA availability."""
    return cuda_available()


@pytest.fixture(autouse=True)
def restore_cwd() -> Generator[None, None, None]:
    """Restore the current working directory after each test.

    Some tests (e.g., WASM examples) change the working directory,
    which can interfere with other tests that rely on path resolution.
    """
    import os

    original_cwd = Path.cwd()
    yield
    os.chdir(original_cwd)


def pytest_configure(config: Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "gpu: marks tests as requiring GPU")
    config.addinivalue_line("markers", "cuda: marks tests as requiring CUDA")


def pytest_collection_modifyitems(
    config: Config,  # noqa: ARG001
    items: list[pytest.Item],  # noqa: ARG001
) -> None:
    """Print CUDA status at collection time."""
    cuda = cuda_available()
    if not cuda:
        # Provide more detail about why CUDA is unavailable
        cargo_path = shutil.which("cargo")
        toolkit_ok = False
        cupy_ok = False

        if cargo_path is not None:
            try:
                result = subprocess.run(
                    [
                        cargo_path,
                        "run",
                        "-p",
                        "pecos",
                        "--features",
                        "cli",
                        "--",
                        "cuda",
                        "check",
                        "-q",
                    ],
                    capture_output=True,
                    timeout=30,
                    check=False,
                )
                toolkit_ok = result.returncode == 0
            except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
                toolkit_ok = False

        try:
            result = subprocess.run(
                [sys.executable, "-c", "import cupy; print(cupy.cuda.is_available())"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            cupy_ok = result.returncode == 0 and "True" in result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            cupy_ok = False

        if toolkit_ok and not cupy_ok:
            print(
                "\nCUDA Python tests: skipped (cupy not installed - run 'pecos cuda setup-python')",
            )
        elif not toolkit_ok:
            print("\nCUDA Python tests: skipped (CUDA toolkit not available)")
        else:
            print("\nCUDA Python tests: skipped")
    else:
        print("\nCUDA Python tests: enabled")
