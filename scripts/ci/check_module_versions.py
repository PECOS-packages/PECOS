#!/usr/bin/env python3
# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Assert that installed PECOS modules report their own distribution's version.

The extension modules build their `__version__` from the wheel's `[project].version` (see
`pecos_build::python::emit_python_version`), while the Rust crate version they are compiled
from is a different number. This checks the two against each other on a real installation,
which is the only place the wiring can be observed end to end.
"""

from __future__ import annotations

import importlib
import sys
from importlib.metadata import version

DISTRIBUTIONS = {
    "pecos": "quantum-pecos",
    "pecos_rslib": "pecos-rslib",
    "pecos_rslib_cuda": "pecos-rslib-cuda",
    "pecos_rslib_exp": "pecos-rslib-exp",
    "pecos_rslib_llvm": "pecos-rslib-llvm",
}


def main(module_names: list[str]) -> int:
    if not module_names:
        print(f"usage: {sys.argv[0]} MODULE [MODULE ...]", file=sys.stderr)
        return 2

    errors = []
    for name in module_names:
        distribution = DISTRIBUTIONS.get(name)
        if distribution is None:
            errors.append(f"{name}: not a known PECOS module; add it to {__file__}")
            continue

        module = importlib.import_module(name)
        reported = getattr(module, "__version__", None)
        installed = version(distribution)
        if reported != installed:
            errors.append(f"{name}.__version__ is {reported!r}, but {distribution} {installed!r} is installed")
        else:
            print(f"{name} {reported}")

    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
