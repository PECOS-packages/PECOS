#!/usr/bin/env python3
# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Smoke-test an installed PECOS Selene plugin wheel.

The plugin test job runs against editable installs, so a wheel that ships
without its native library, or with one that cannot be loaded on the target
platform, would pass CI. This imports the package from the installed wheel,
builds its Selene component with default arguments, and loads the bundled
library.

Usage: smoke_selene_wheel.py <package>   (run outside the source tree)
"""

from __future__ import annotations

import ctypes
import importlib
import inspect
import sys
from pathlib import Path

from selene_core import SeleneComponent


def main(package: str) -> None:
    module = importlib.import_module(package)
    location = Path(module.__file__).resolve()
    if not location.is_relative_to(Path(sys.prefix).resolve()):
        sys.exit(f"{package} imported from {location}, not from the installed wheel")

    components = [
        obj
        for obj in vars(module).values()
        if inspect.isclass(obj) and issubclass(obj, SeleneComponent) and obj.__module__.startswith(package)
    ]
    if len(components) != 1:
        sys.exit(f"expected one Selene component in {package}, found {components}")

    library = components[0]().library_file
    if not library.is_file():
        sys.exit(f"{components[0].__name__}.library_file does not exist: {library}")
    ctypes.CDLL(str(library))
    print(f"{package}: {components[0].__name__} loaded {library.name}")


if __name__ == "__main__":
    main(sys.argv[1])
