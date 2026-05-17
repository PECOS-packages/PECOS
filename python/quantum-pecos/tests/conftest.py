"""Shared pytest setup for the Python test suite."""

# The test tree contains ``tests/pecos``. Pytest can import tests from that
# directory as a namespace package named ``pecos`` when an individual file is run
# in isolation. Import the installed/source package first so later
# ``import pecos`` statements resolve to the public PECOS package.
import pecos


def pytest_configure(config):
    """Register markers at the test-tree root so they are known for ANY
    invocation (e.g. running a single file directly), not only when
    pytest happens to pick a ``pyproject.toml`` whose
    ``[tool.pytest.ini_options].markers`` lists them."""
    config.addinivalue_line(
        "markers",
        "slow: mark tests that provide extra integration coverage but are "
        "excluded from the default fast Python test lane",
    )
