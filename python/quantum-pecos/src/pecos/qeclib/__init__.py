# Copyright 2024 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Deprecated: pecos.qeclib has moved to pecos.slr.qeclib.

This module provides backward compatibility imports. Please update your imports to use
`pecos.slr.qeclib` instead.
"""

import sys
import warnings

warnings.warn(
    "pecos.qeclib is deprecated and will be removed in a future version. "
    "Please use pecos.slr.qeclib instead.",
    DeprecationWarning,
    stacklevel=2,
)

# Import and re-export submodules from the new location
from pecos.slr.qeclib import generic, qubit, steane, surface

# Make submodules available under the old path as well
# This allows "from pecos.qeclib.qubit import H" to work
sys.modules["pecos.qeclib.generic"] = generic
sys.modules["pecos.qeclib.qubit"] = qubit
sys.modules["pecos.qeclib.steane"] = steane
sys.modules["pecos.qeclib.surface"] = surface

# Also register sub-submodules for nested imports
# e.g., from pecos.qeclib.qubit.qgate_base import QGate
for attr_name in dir(qubit):
    attr = getattr(qubit, attr_name)
    if hasattr(attr, "__name__") and attr.__name__.startswith("pecos.slr.qeclib.qubit."):
        old_name = attr.__name__.replace("pecos.slr.qeclib.", "pecos.")
        sys.modules[old_name] = attr

for attr_name in dir(steane):
    attr = getattr(steane, attr_name)
    if hasattr(attr, "__name__") and attr.__name__.startswith("pecos.slr.qeclib.steane."):
        old_name = attr.__name__.replace("pecos.slr.qeclib.", "pecos.")
        sys.modules[old_name] = attr

for attr_name in dir(surface):
    attr = getattr(surface, attr_name)
    if hasattr(attr, "__name__") and attr.__name__.startswith("pecos.slr.qeclib.surface."):
        old_name = attr.__name__.replace("pecos.slr.qeclib.", "pecos.")
        sys.modules[old_name] = attr

for attr_name in dir(generic):
    attr = getattr(generic, attr_name)
    if hasattr(attr, "__name__") and attr.__name__.startswith("pecos.slr.qeclib.generic."):
        old_name = attr.__name__.replace("pecos.slr.qeclib.", "pecos.")
        sys.modules[old_name] = attr

# Note: color488 is not registered to avoid matplotlib import errors
# Users of color488 should update to pecos.slr.qeclib.color488

__all__ = ["generic", "qubit", "steane", "surface"]
