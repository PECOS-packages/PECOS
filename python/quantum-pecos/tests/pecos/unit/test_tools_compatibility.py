"""Deprecated modules preserve the complete public analysis namespace."""

import ast
import importlib
import inspect
import warnings

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "fault_tolerance_checking",
        "logic_circuit_speed",
        "pseudo_threshold_tools",
        "syndromes",
        "threshold_tools",
        "tool_anticommute",
        "tool_collection",
    ],
)
def test_tools_compatibility_module_reexports_public_names(module) -> None:
    analysis = importlib.import_module(f"pecos.analysis.{module}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = importlib.import_module(f"pecos.tools.{module}")

    public_names = set(analysis.__all__)
    imported_names = {
        alias.asname or alias.name.split(".")[0]
        for node in ast.parse(inspect.getsource(analysis)).body
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    }
    assert not public_names & imported_names
    assert set(legacy.__all__) == public_names
    for name in public_names:
        assert getattr(legacy, name) is getattr(analysis, name), name
