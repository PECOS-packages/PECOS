"""Tests for the version attributes of the native ``pecos_rslib*`` extension modules."""

from importlib.metadata import version

import pecos_rslib
import pecos_rslib_exp
import pecos_rslib_llvm
import pytest

MODULES = [
    (pecos_rslib, "pecos-rslib"),
    (pecos_rslib_llvm, "pecos-rslib-llvm"),
    (pecos_rslib_exp, "pecos-rslib-exp"),
]


@pytest.mark.parametrize(("module", "distribution"), MODULES, ids=lambda param: getattr(param, "__name__", param))
def test_version_matches_distribution_metadata(module: object, distribution: str) -> None:
    """``__version__`` reports the installed wheel's version.

    The Rust crate version is a separate train (``[workspace.package].version`` in the root
    ``Cargo.toml``), so a module built from ``CARGO_PKG_VERSION`` reports a different number
    than the one pip or uv installed. build.rs injects the ``pyproject.toml`` version instead.
    """
    assert module.__version__ == version(distribution)
