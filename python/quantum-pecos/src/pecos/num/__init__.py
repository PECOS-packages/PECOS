# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Polymorphic wrappers for pecos.num functions.

This module provides numpy-compatible functions that automatically dispatch
to the appropriate Rust implementation based on input types.

All functionality is provided by pecos_rslib - this module just re-exports
with clean documentation.
"""

from __future__ import annotations

# Import all functions from pecos_rslib
# pecos_rslib handles all the complexity of calling Rust polymorphic functions,
# list-to-array conversion, broadcasting, etc.
# Re-export other functions
# Import mean/std from pecos_rslib (not .num) to get Python wrappers with axis support
# Re-export submodules
from pecos_rslib import (
    cos,
    e,
    # Polymorphic math functions
    exp,
    isclose,
    # Polymorphic comparison functions
    isnan,
    mean,
    num,
    # Constants
    pi,
    power,
    sin,
    sqrt,
    std,
    tau,
)

# Import remaining functions from pecos_rslib.num (Rust module)
from pecos_rslib.num import (
    Poly1d,
    brentq,
    ceil,
    curve_fit,
    diag,
    floor,
    linspace,
    newton,
    polyfit,
    round,  # noqa: A004 - intentionally shadow builtin for numpy compatibility
)

# Make submodules available
stats = num.stats
math = num.math
compare = num.compare
array = num.array
optimize = num.optimize
polynomial = num.polynomial
# Note: curve_fit is already imported above from pecos_rslib.num
random = num.random

__all__ = [
    "Poly1d",
    "array",
    "brentq",
    # Direct exports
    "ceil",
    "compare",
    "cos",
    "curve_fit",
    "diag",
    "e",
    "exp",
    "floor",
    "isclose",
    # Polymorphic functions
    "isnan",
    "linspace",
    "math",
    "mean",
    "newton",
    "optimize",
    # Constants
    "pi",
    "polyfit",
    "polynomial",
    "power",
    "random",
    "round",
    "sin",
    "sqrt",
    # Submodules
    "stats",
    "std",
    "tau",
]
