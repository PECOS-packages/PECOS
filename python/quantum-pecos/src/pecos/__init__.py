# Copyright 2018 The PECOS Developers
# Copyright 2014-2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract
# DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Performance Estimator of Codes On Surfaces (PECOS).

==================================================
A framework for developing, studying, and evaluating quantum error-correcting codes.
"""

# Allow for other namespace packages
__path__ = __import__("pkgutil").extend_path(__path__, __name__)

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("quantum-pecos")
except PackageNotFoundError:
    __version__ = "0.0.0"

# PECOS namespaces
import sys
from typing import NoReturn

from _pecos_rslib import (
    Array,  # Array type with generic dtype support (Array[f64], etc.)
    Pauli,  # Quantum Pauli operators (I, X, Y, Z)
    PauliString,  # Multi-qubit Pauli operators
    abs,  # Absolute value  # noqa: A004
    all,  # All elements true  # noqa: A004
    allclose,  # Approximate equality (arrays)
    any,  # Any element true  # noqa: A004
    array,  # Array creation
    array_equal,  # Array equality
    complex64,
    complex128,
    cos,  # Cosine
    cosh,  # Hyperbolic cosine
    dtypes,  # Keep dtypes module for dtype instances (dtypes.i64, etc.)
    exp,  # Exponential
    f32,
    f64,
    graph,
    i8,
    i16,
    i32,
    i64,
    isclose,  # Approximate equality (element-wise)
    isnan,  # Check for NaN
    ln,  # Natural logarithm
    log,  # Logarithm with base
    max,  # Maximum value  # noqa: A004
    mean,  # Mean/average
    min,  # Minimum value  # noqa: A004
    num,
    power,  # Power function
    sin,  # Sine
    sinh,  # Hyperbolic sine
    sqrt,  # Square root
    std,  # Standard deviation
    sum,  # Sum  # noqa: A004
    tan,  # Tangent
    tanh,  # Hyperbolic tangent
    u8,
    u16,
    u32,
    u64,
    where,  # Conditional selection
)

# Note: Mathematical constants (pi, e, tau, frac_pi_2, sqrt_2, ln_2, etc.) are NOT imported
# They are only available via dtype namespaces: pc.f64.pi, pc.f64.frac_pi_2, etc.
# This makes precision explicit and supports future f32, complex constants
# Polynomial and optimization functions (commonly used, so at top level)
from _pecos_rslib.num import (
    Poly1d,  # Polynomial evaluation
    arange,  # Range arrays
    brentq,  # Brent's root finding
    ceil,  # Ceiling function
    curve_fit,  # Non-linear curve fitting
    delete,  # Delete elements
    diag,  # Diagonal extraction
    floor,  # Floor function
    linspace,  # Linearly spaced arrays
    newton,  # Newton-Raphson root finding
    ones,  # Arrays of ones
    polyfit,  # Polynomial fitting
    round,  # Rounding  # noqa: A004
    zeros,  # Arrays of zeros
)

# Type hints for arrays and scalars
from pecos import typing

# Graph algorithms
# ============================================================================
# NumPy-style Numerical Computing API (Hybrid Flat + Structured)
# ============================================================================
#
# PECOS follows NumPy's organization:
#   - Common functions at top level: pecos.array(), pecos.sin(), pecos.mean()
#   - Specialized functions in submodules: pecos.linalg.norm(), pecos.random.randint()
#
# This provides the best user experience:
#   import pecos as pc
#   arr = pc.array([1, 2, 3])        # Common operations - flat and convenient
#   norm = pc.linalg.norm(arr)       # Specialized operations - organized
#   one = pc.i64(1)                  # Data types - flat for convenience
# Import the Rust num module directly from _pecos_rslib
# ============================================================================
# Top-level: Common numerical functions (like NumPy's flat namespace)
# ============================================================================
# Array creation and manipulation
# Mathematical functions (element-wise operations)
# Statistical functions
# Comparison and logical functions
# Data types - import scalar type classes directly (NumPy-like API)
# This allows: pc.i64(42) and def foo(x: pc.i64) just like np.int64(42) and def foo(x: np.int64)
# Mathematical constants
# Type aliases for numeric types (from pecos.typing, not pecos_rslib)
from pecos.typing import (
    # Also export runtime type tuples for isinstance checks
    COMPLEX_TYPES,
    FLOAT_TYPES,
    INEXACT_TYPES,
    INTEGER_TYPES,
    NUMERIC_TYPES,
    SIGNED_INTEGER_TYPES,
    UNSIGNED_INTEGER_TYPES,
    Complex,
    Float,
    Inexact,
    Integer,
    Numeric,
    SignedInteger,
    UnsignedInteger,
)

# ============================================================================
# Structured submodules: Specialized functionality (like NumPy's submodules)
# ============================================================================

# Linear algebra: pecos.linalg.norm(), pecos.linalg.svd()
linalg = num.linalg

# Random number generation: pecos.random.randint(), pecos.random.normal()
random = num.random

# Optimization: pecos.optimize.brentq(), pecos.optimize.newton()
optimize = num.optimize

# Polynomial operations: pecos.polynomial.polyfit(), pecos.polynomial.Poly1d
polynomial = num.polynomial

# Statistics: pecos.stats.* (if we add more advanced stats functions)
stats = num.stats

# Mathematical functions: pecos.math.* (less common functions)
math = num.math

# Comparison functions: pecos.compare.* (advanced comparisons)
compare = num.compare

# Note: pecos.num namespace has been removed
# Everything is now directly under pecos for a cleaner API:
#   - pecos.array() instead of pecos.num.array()
#   - pecos.linalg.norm() instead of pecos.num.linalg.norm()
#
# This follows the principle: "flat is better than nested" for the main namespace

# These imports come after sys.modules setup - this is intentional
from pecos import (  # noqa: E402
    circuit_converters,
    circuits,
    decoders,
    engines,
    error_models,
    frontends,
    misc,
    protocols,
    qeccs,
    simulators,
    tools,
)
from pecos.circuits.quantum_circuit import QuantumCircuit  # noqa: E402
from pecos.engines import circuit_runners  # noqa: E402
from pecos.engines.cvm.binarray import BinArray  # noqa: E402
from pecos.engines.hybrid_engine_old import HybridEngine  # noqa: E402

# Import Guppy functionality (with graceful fallback)
try:
    from pecos.frontends import (
        get_guppy_backends,
        sim,
    )

    GUPPY_INTEGRATION_AVAILABLE = True
except ImportError:
    GUPPY_INTEGRATION_AVAILABLE = False

    def sim(*args: object, **kwargs: object) -> NoReturn:
        """Stub for sim when Guppy integration is not available."""
        del args, kwargs  # Unused
        msg = "Guppy integration not available. Install with: pip install quantum-pecos[guppy]"
        raise ImportError(
            msg,
        )

    def get_guppy_backends() -> dict:
        """Stub for get_guppy_backends."""
        return {"guppy_available": False, "rust_backend": False}


__all__ = [
    "GUPPY_INTEGRATION_AVAILABLE",
    "BinArray",
    "HybridEngine",
    "QuantumCircuit",
    "__version__",
    "circuit_converters",
    "circuit_runners",
    "circuits",
    "complex64",
    "complex128",
    "decoders",
    # Keep dtypes module for dtype instances
    "dtypes",
    "engines",
    "error_models",
    "f32",
    "f64",
    "frontends",
    "get_guppy_backends",
    # Scalar type classes (NumPy-like API)
    "i8",
    "i16",
    "i32",
    "i64",
    "misc",
    "num",  # Numerical computing module from _pecos_rslib
    "protocols",
    "qeccs",
    # Guppy integration
    "sim",
    "simulators",
    "tools",
    "typing",  # Type hints for arrays and scalars
    "u8",
    "u16",
    "u32",
    "u64",
]
