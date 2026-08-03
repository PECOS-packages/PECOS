# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Deprecated: use pecos.guppy_gen instead."""

import sys
import warnings

warnings.warn(
    "pecos.guppy has been renamed to pecos.guppy_gen. "
    "Please update your imports. pecos.guppy will be removed in a future release.",
    DeprecationWarning,
    stacklevel=2,
)

from pecos import guppy_gen  # noqa: E402
from pecos.guppy_gen import *  # noqa: E402, F403
from pecos.guppy_gen import __all__  # noqa: E402

# Alias the submodules so `import pecos.guppy.surface` and unpickling of
# objects serialized before the rename resolve to the real modules.
for _submodule in ("color", "surface", "transversal", "variant"):
    sys.modules[f"{__name__}.{_submodule}"] = getattr(guppy_gen, _submodule)
del _submodule
