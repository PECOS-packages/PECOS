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
from typing import TYPE_CHECKING

import pecos_rslib
from pecos_rslib import (
    Array,  # Array type with generic dtype support (Array[f64], etc.)
    BitInt,  # Fixed-width binary integer type
    Pauli,  # Quantum Pauli operators (I, X, Y, Z)
    PauliString,  # Multi-qubit Pauli operators
    WasmForeignObject,  # WASM foreign object for classical coprocessor
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
from pecos_rslib.num import (
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
# Numerical Computing API (Hybrid Flat + Structured)
# ============================================================================
#
# PECOS follows this organization:
#   - Common functions at top level: pecos.array(), pecos.sin(), pecos.mean()
#   - Specialized functions in submodules: pecos.linalg.norm(), pecos.random.randint()
#
# This provides the best user experience:
#   import pecos as pc
#   arr = pc.array([1, 2, 3])        # Common operations - flat and convenient
#   norm = pc.linalg.norm(arr)       # Specialized operations - organized
#   one = pc.i64(1)                  # Data types - flat for convenience
# Import the Rust num module directly from pecos_rslib
# ============================================================================
# Top-level: Common numerical functions
# ============================================================================
# Array creation and manipulation
# Mathematical functions (element-wise operations)
# Statistical functions
# Comparison and logical functions
# Data types - import scalar type classes directly
# This allows: pc.i64(42) and def foo(x: pc.i64)
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

# ===================================================
# Structured submodules: Specialized functionality
# ===================================================

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
from pecos import (
    analysis,  # QEC analysis tools (threshold, fault tolerance, stabilizers)
    benchmarks,  # Performance benchmarking
    circuit_converters,
    circuits,
    decoders,
    engines,
    error_models,
    exceptions,  # Exception classes
    graph,
    misc,
    programs,
    protocols,
    qeccs,
    simulators,
    testing,  # Testing utilities (like numpy.testing)
    tools,
)

# Deprecated APIs
from pecos._deprecated import BinArray

# Engine builder classes and factory functions
from pecos._engine_builders import (
    PhirJsonEngineBuilder,
    QasmEngineBuilder,
    QisEngineBuilder,
    phir_json_engine,
    qasm_engine,
    qis_engine,
)

# Simulation entry point
from pecos._sim import get_guppy_backends, sim
from pecos.circuits.quantum_circuit import QuantumCircuit
from pecos.engines import circuit_runners
from pecos.engines.hybrid_engine_old import HybridEngine

# Import program wrappers from programs submodule for convenience
# These can also be accessed via pecos.programs.Qasm, etc.
from pecos.programs import Guppy, Hugr, PhirJson, ProgramWrapper, Qasm, Qis, Wasm, Wat

# Re-export noise and quantum engine builders from pecos_rslib
# These don't need wrappers since they don't take program types
depolarizing_noise = pecos_rslib.depolarizing_noise
biased_depolarizing_noise = pecos_rslib.biased_depolarizing_noise
general_noise = pecos_rslib.general_noise
state_vector = pecos_rslib.state_vector
sparse_stabilizer = pecos_rslib.sparse_stabilizer

# Re-export noise model builder classes for direct instantiation
GeneralNoiseModelBuilder = pecos_rslib.GeneralNoiseModelBuilder


__all__ = [
    "COMPLEX_TYPES",
    "FLOAT_TYPES",
    "INEXACT_TYPES",
    "INTEGER_TYPES",
    "NUMERIC_TYPES",
    "SIGNED_INTEGER_TYPES",
    "UNSIGNED_INTEGER_TYPES",
    # Core types
    "Array",
    # Deprecated
    "BinArray",  # Deprecated - use BitInt instead
    "BitInt",
    # Type categories
    "Complex",
    "Float",
    "GeneralNoiseModelBuilder",
    # Program wrapper classes for sim() - also available via pecos.programs
    "Guppy",
    "Hugr",
    # Legacy
    "HybridEngine",
    "Inexact",
    "Integer",
    "Numeric",
    "Pauli",
    "PauliString",
    "PhirJson",
    # Engine builder classes
    "PhirJsonEngineBuilder",
    "Poly1d",
    "ProgramWrapper",
    "Qasm",
    "QasmEngineBuilder",
    "Qis",
    "QisEngineBuilder",
    "QuantumCircuit",
    "SignedInteger",
    "UnsignedInteger",
    "Wasm",
    "WasmForeignObject",
    "Wat",
    # Version
    "__version__",
    # Mathematical functions
    "abs",
    "all",
    "allclose",
    # Subpackages
    "analysis",  # QEC analysis (threshold, fault tolerance, stabilizers)
    "any",
    # Polynomial and optimization
    "arange",
    "array",
    "array_equal",
    "benchmarks",  # Performance benchmarking
    # Noise model builders
    "biased_depolarizing_noise",
    "brentq",
    "ceil",
    # Subpackages - Utilities
    "circuit_converters",
    "circuit_runners",
    # Subpackages - Core
    "circuits",
    # Numeric submodules (like numpy.linalg, numpy.random)
    "compare",
    "complex64",
    "complex128",
    "cos",
    "cosh",
    "curve_fit",
    "decoders",
    "delete",
    "depolarizing_noise",
    "diag",
    # Data types
    "dtypes",
    "engines",
    "error_models",
    "exceptions",  # Exception classes
    "exp",
    "f32",
    "f64",
    "floor",
    "general_noise",
    "get_guppy_backends",
    "graph",
    "i8",
    "i16",
    "i32",
    "i64",
    "isclose",
    "isnan",
    "linalg",
    "linspace",
    "ln",
    "log",
    "math",
    "max",
    "mean",
    "min",
    "misc",  # Kept for backwards compatibility
    "newton",
    "num",
    "ones",
    "optimize",
    # Engine builder functions
    "phir_json_engine",
    "polyfit",
    "polynomial",
    "power",
    "programs",
    "protocols",
    "qasm_engine",
    "qeccs",
    "qis_engine",
    "random",
    "round",
    # Simulation entry point
    "sim",
    "simulators",
    "sin",
    "sinh",
    # Quantum simulators
    "sparse_stabilizer",
    "sqrt",
    "state_vector",
    "stats",
    "std",
    "sum",
    "tan",
    "tanh",
    "testing",  # Testing utilities (like numpy.testing)
    "tools",  # Kept for backwards compatibility
    "typing",
    "u8",
    "u16",
    "u32",
    "u64",
    "where",
    "zeros",
]
