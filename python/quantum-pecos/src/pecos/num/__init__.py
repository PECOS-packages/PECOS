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
# Import dtypes module from pecos_rslib
from pecos_rslib import (
    abs,  # noqa: A004 - intentionally shadow builtin
    all,  # noqa: A004 - intentionally shadow builtin
    allclose,
    any,  # noqa: A004 - intentionally shadow builtin
    cos,
    dtypes,
    e,
    # Polymorphic math functions
    exp,
    # Polymorphic comparison functions
    isclose,
    isnan,
    ln,  # Natural logarithm (clearer than log)
    log,  # Logarithm with custom base
    mean,
    num,
    # Constants
    pi,
    power,
    sin,
    sqrt,
    std,
    sum,  # noqa: A004 - intentionally shadow builtin
    tau,
    where,
)

# Import remaining functions from pecos_rslib.num (Rust module)
# Note: arange imported from num_wrapper to get dtype inference wrapper
from pecos_rslib.num import (
    Poly1d,
    arange,
    array,
    brentq,
    ceil,
    curve_fit,
    delete,
    diag,
    floor,
    linspace,
    newton,
    ones,
    polyfit,
    round,  # noqa: A004 - intentionally shadow builtin
    zeros,
)

# Import types module for generic Array typing support
from pecos.num import types

# Make submodules available
stats = num.stats
math = num.math
compare = num.compare
# Note: array function already imported above from pecos_rslib.num
optimize = num.optimize
polynomial = num.polynomial
# Note: curve_fit is already imported above from pecos_rslib.num
random = num.random
linalg = num.linalg

# zeros_complex() has been removed - use zeros(shape, dtype="complex") instead

__all__ = [
    "Poly1d",
    "abs",
    "all",  # Boolean AND reduction
    "allclose",
    "any",  # Boolean OR reduction
    "arange",
    "array",
    "brentq",
    # Direct exports
    "ceil",
    "compare",
    "cos",
    "curve_fit",
    "delete",
    "diag",
    "dtypes",  # Dtype system
    "e",
    "exp",
    "floor",
    # Polymorphic functions
    "isclose",
    "isnan",
    "linalg",
    "linspace",
    "ln",  # Natural logarithm
    "log",  # Logarithm with base
    "math",
    "mean",
    "newton",
    "ones",
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
    "sum",
    "tau",
    "types",  # Type hints for Array
    "where",
    "zeros",
]
