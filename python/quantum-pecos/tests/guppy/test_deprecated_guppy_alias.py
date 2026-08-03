# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Regression tests for the deprecated pecos.guppy alias (renamed to pecos.guppy_gen)."""

import importlib
import sys

import pecos
import pecos.guppy_gen
import pytest


def _purge_alias():
    """Remove any cached pecos.guppy state so each test exercises a cold import.

    Checks the module __dict__ directly: hasattr(pecos, "guppy") would trigger
    pecos.__getattr__, importing the alias and emitting the warning outside the
    pytest.warns context.
    """
    for name in list(sys.modules):
        if name == "pecos.guppy" or name.startswith("pecos.guppy."):
            del sys.modules[name]
    vars(pecos).pop("guppy", None)


def test_import_warns_and_matches_new_module():
    _purge_alias()
    with pytest.warns(DeprecationWarning, match="pecos.guppy has been renamed to pecos.guppy_gen"):
        legacy = importlib.import_module("pecos.guppy")
    assert legacy.make_surface_code is pecos.guppy_gen.make_surface_code
    assert legacy.__all__ == pecos.guppy_gen.__all__


def test_submodule_import_resolves_to_real_module():
    # Unpickling objects serialized before the rename does exactly this:
    # import the submodule by its old dotted path, then getattr the name.
    _purge_alias()
    with pytest.warns(DeprecationWarning, match="pecos.guppy has been renamed"):
        legacy_surface = importlib.import_module("pecos.guppy.surface")
    assert legacy_surface is pecos.guppy_gen.surface
    assert legacy_surface.make_surface_code is pecos.guppy_gen.surface.make_surface_code


def test_attribute_access_on_pecos_package():
    _purge_alias()
    with pytest.warns(DeprecationWarning, match="pecos.guppy has been renamed"):
        legacy = pecos.guppy
    assert legacy.get_num_qubits is pecos.guppy_gen.get_num_qubits


def test_submodules_are_package_attributes():
    # `import pecos.guppy.surface; pecos.guppy.surface` needs the submodule
    # bound as an attribute; the sys.modules alias alone does not provide it.
    _purge_alias()
    with pytest.warns(DeprecationWarning, match="pecos.guppy has been renamed"):
        legacy = importlib.import_module("pecos.guppy")
    for submodule in ("color", "surface", "transversal", "variant"):
        assert getattr(legacy, submodule) is getattr(pecos.guppy_gen, submodule)


def test_star_import_contract_keeps_guppy():
    # `from pecos import *` exported "guppy" before the rename.
    assert "guppy" in pecos.__all__
