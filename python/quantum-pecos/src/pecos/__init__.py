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
    AngleSource,  # Angle source specification for gate decomposition
    Array,  # Array type with generic dtype support (Array[f64], etc.)
    BitInt,  # Fixed-width binary integer type
    BitUInt,  # Unsigned fixed-width binary integer type
    ByteMessage,  # Binary protocol for quantum commands and measurement results
    ByteMessageBuilder,  # Builder for ByteMessage
    GateRegistry,  # Gate registration system for custom gate decomposition
    GateSignatureMismatchError,  # Raised when custom gate arity mismatches
    Nanoseconds,  # Time duration in nanoseconds
    Pauli,  # Quantum Pauli operators (I, X, Y, Z)
    PauliString,  # Multi-qubit Pauli operators
    ShotMap,  # Simulation result: register name -> measurement outcomes
    ShotVec,  # Simulation result: vector of shots
    TimeUnits,  # Abstract time duration in arbitrary units
    WasmForeignObject,  # WASM foreign object for classical coprocessor
    X,  # Single-qubit Pauli X constructor: X(qubit) -> PauliString
    Y,  # Single-qubit Pauli Y constructor: Y(qubit) -> PauliString
    Z,  # Single-qubit Pauli Z constructor: Z(qubit) -> PauliString
    abs,  # Absolute value
    acos,  # Inverse cosine
    acosh,  # Inverse hyperbolic cosine
    all,  # All elements true
    allclose,  # Approximate equality (arrays)
    angle64,  # Fixed-point angle type with exact constants (pi, frac_pi_2, etc.)
    any,  # Any element true
    array,  # Array creation
    array_equal,  # Array equality
    asin,  # Inverse sine
    asinh,  # Inverse hyperbolic sine
    atan,  # Inverse tangent
    atan2,  # Two-argument inverse tangent
    atanh,  # Inverse hyperbolic tangent
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
    inf,  # Infinity
    isclose,  # Approximate equality (element-wise)
    isnan,  # Check for NaN
    kron,  # Kronecker product
    ln,  # Natural logarithm
    log,  # Logarithm with base
    max,  # Maximum value
    mean,  # Mean/average
    min,  # Minimum value
    nan,  # Not a number
    num,
    power,  # Power function
    sin,  # Sine
    sinh,  # Hyperbolic sine
    sqrt,  # Square root
    std,  # Standard deviation
    sum,  # Sum
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
    round,  # Rounding
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
# Make RngPcg accessible via pecos.random.RngPcg
random.RngPcg = pecos_rslib.RngPcg

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
    exceptions,  # Exception classes
    graph,
    guppy,  # Direct Guppy code generation for QEC - bypasses SLR
    noise,
    programs,
    protocols,
    qec,  # Pure QEC geometry (surface, color codes) - no SLR dependencies
    qeccs,
    quantum,  # Quantum types (DagCircuit, Gate, Pauli, etc.)
    simulators,
    testing,  # Testing utilities (like numpy.testing)
)

# pecos.tools is deprecated (renamed to pecos.analysis).
# Not eagerly imported to avoid triggering the deprecation warning on every `import pecos`.
# Lazy import via __getattr__ so `pc.tools.X` still works but emits a warning.


def __getattr__(name: str):
    if name == "tools":
        # Lazy import -- tools/__init__.py emits the deprecation warning
        import importlib

        return importlib.import_module("pecos.tools")
    if name == "misc":
        msg = (
            "pecos.misc has been removed. Its contents have been moved to:\n"
            "  - pecos.analysis (threshold_curve, stabilizer_funcs)\n"
            "  - pecos.quantum (commute, gate_groups)\n"
            "  - pecos.engines (std_output)"
        )
        raise AttributeError(msg)
    msg = f"module 'pecos' has no attribute {name!r}"
    raise AttributeError(msg)


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
    selene_engine,
)

# Simulation entry point
from pecos._sim import get_guppy_backends, sim
from pecos.circuits.quantum_circuit import QuantumCircuit
from pecos.engines import circuit_runners
from pecos.engines.hybrid_engine_old import HybridEngine

# Import WasmError from pecos.exceptions (Python-defined, inherits from pecos_rslib.WasmError)
# so that errors re-raised through the Python layer display as pecos.WasmError
from pecos.exceptions import WasmError

# Import program wrappers from programs submodule for convenience
# These can also be accessed via pecos.programs.Qasm, etc.
from pecos.programs import Guppy, Hugr, PhirJson, ProgramWrapper, Qasm, Qis, Wasm, Wat

# Re-export noise and quantum engine builders from pecos_rslib
# These don't need wrappers since they don't take program types
BiasedDepolarizingNoiseModelBuilder = pecos_rslib.BiasedDepolarizingNoiseModelBuilder
DepolarizingNoiseModelBuilder = pecos_rslib.DepolarizingNoiseModelBuilder
depolarizing_noise = pecos_rslib.depolarizing_noise
biased_depolarizing_noise = pecos_rslib.biased_depolarizing_noise
general_noise = pecos_rslib.general_noise
state_vector = pecos_rslib.state_vector
stabilizer = pecos_rslib.stabilizer
sparse_stab = pecos_rslib.sparse_stab
stab_vec = pecos_rslib.stab_vec
density_matrix = pecos_rslib.density_matrix
hugr_engine = pecos_rslib.hugr_engine

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
    "AngleSource",
    "Array",
    "BiasedDepolarizingNoiseModelBuilder",
    "BinArray",
    "BitInt",
    "BitUInt",
    "Complex",
    "DepolarizingNoiseModelBuilder",
    "Float",
    "GateRegistry",
    "GateSignatureMismatchError",
    "GeneralNoiseModelBuilder",
    "Guppy",
    "Hugr",
    "HybridEngine",
    "Inexact",
    "Integer",
    "Nanoseconds",
    "Numeric",
    "Pauli",
    "PauliString",
    "PhirJson",
    "PhirJsonEngineBuilder",
    "Poly1d",
    "ProgramWrapper",
    "Qasm",
    "QasmEngineBuilder",
    "Qis",
    "QisEngineBuilder",
    "QuantumCircuit",
    "ShotMap",
    "ShotVec",
    "SignedInteger",
    "TimeUnits",
    "UnsignedInteger",
    "Wasm",
    "WasmError",
    "WasmForeignObject",
    "Wat",
    "X",
    "Y",
    "Z",
    "__version__",
    "abs",
    "acos",
    "acosh",
    "all",
    "allclose",
    "analysis",
    "angle64",
    "any",
    "arange",
    "array",
    "array_equal",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "benchmarks",
    "biased_depolarizing_noise",
    "brentq",
    "ceil",
    "circuit_converters",
    "circuit_runners",
    "circuits",
    "compare",
    "complex64",
    "complex128",
    "cos",
    "cosh",
    "curve_fit",
    "decoders",
    "delete",
    "density_matrix",
    "depolarizing_noise",
    "diag",
    "dtypes",
    "engines",
    "exceptions",
    "exp",
    "f32",
    "f64",
    "floor",
    "general_noise",
    "get_guppy_backends",
    "graph",
    "guppy",
    "hugr_engine",
    "i8",
    "i16",
    "i32",
    "i64",
    "inf",
    "isclose",
    "isnan",
    "kron",
    "linalg",
    "linspace",
    "ln",
    "log",
    "math",
    "max",
    "mean",
    "min",
    "nan",
    "newton",
    "noise",
    "num",
    "ones",
    "optimize",
    "phir_json_engine",
    "polyfit",
    "polynomial",
    "power",
    "programs",
    "protocols",
    "qasm_engine",
    "qec",
    "qeccs",
    "qis_engine",
    "quantum",
    "random",
    "round",
    "selene_engine",
    "sim",
    "simulators",
    "sin",
    "sinh",
    "sparse_stab",
    "sqrt",
    "stab_vec",
    "stabilizer",
    "state_vector",
    "stats",
    "std",
    "sum",
    "tan",
    "tanh",
    "testing",
    "tools",
    "typing",
    "u8",
    "u16",
    "u32",
    "u64",
    "where",
    "zeros",
]
