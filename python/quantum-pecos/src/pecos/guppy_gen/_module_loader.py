# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Load generated source from a real file for Guppy introspection."""

import importlib.util
import sys
from pathlib import Path


def load_guppy_source(source: str, temp_file: Path, module_name: str) -> dict:
    """Write and import source; callers own module identity and caching."""
    temp_file.write_text(source)
    spec = importlib.util.spec_from_file_location(module_name, temp_file)
    if spec is None or spec.loader is None:
        msg = f"Failed to create module spec for {temp_file}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return vars(module)
