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
import warnings
from typing import NoReturn

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
# Import the Rust num module directly from pecos_rslib
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
from pecos import (
    circuit_converters,
    circuits,
    decoders,
    engines,
    error_models,
    graph,
    misc,
    programs,
    protocols,
    qeccs,
    simulators,
    tools,
)
from pecos.circuits.quantum_circuit import QuantumCircuit
from pecos.engines import circuit_runners
from pecos.engines.hybrid_engine_old import HybridEngine


def BinArray(*args, **kwargs):  # noqa: N802
    """Deprecated: Use BitInt instead.

    BinArray is a deprecated alias for BitInt. It will be removed in a future version.
    Please update your code to use BitInt directly.
    """
    warnings.warn(
        "BinArray is deprecated and will be removed in a future version. "
        "Please use BitInt instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return BitInt(*args, **kwargs)


# Import program wrappers from programs submodule for convenience
# These can also be accessed via pecos.programs.Qasm, etc.
from pecos.programs import Guppy, Hugr, PhirJson, ProgramWrapper, Qasm, Qis, Wasm, Wat


def sim(program):
    """Create a simulation builder for a quantum program.

    This is the primary entry point for running quantum simulations in PECOS.

    Args:
        program: A wrapped quantum program (Guppy, Qasm, Qis, Hugr, PhirJson, Wasm, or Wat),
                 a raw Rust program type from pecos_rslib,
                 or a Guppy-decorated function (which will be auto-wrapped).

    Returns:
        A SimBuilder that can be configured and run.

    Example:
        >>> from pecos import sim, Qasm
        >>> results = sim(Qasm("OPENQASM 2.0; qreg q[2]; ...")).run(1000)

        >>> # Guppy functions are auto-wrapped
        >>> @guppy
        ... def my_circuit():
        ...     q = qubit()
        ...     return measure(q)
        ...
        >>> results = sim(my_circuit).run(100)
    """
    # Auto-wrap Guppy-decorated functions (they have a 'compile' method)
    if hasattr(program, "compile") and not hasattr(program, "_to_program"):
        program = Guppy(program)

    # If it's a Python wrapper, extract the underlying Rust type
    if hasattr(program, "_to_program"):
        return pecos_rslib.sim(program._to_program())
    # It's already a Rust type (from pecos_rslib), pass directly
    return pecos_rslib.sim(program)


# =============================================================================
# Engine Builder Wrappers
# =============================================================================
# These wrap the pecos_rslib engine builders to accept Python program wrappers


class QasmEngineBuilder:
    """Python wrapper for QASM engine builder.

    This wrapper accepts Python Qasm objects from pecos.programs.

    Example:
        >>> from pecos import qasm_engine, Qasm
        >>> results = (
        ...     qasm_engine()
        ...     .program(Qasm("OPENQASM 2.0; qreg q[2]; ..."))
        ...     .to_sim()
        ...     .run(1000)
        ... )
    """

    def __init__(self):
        self._builder = pecos_rslib.qasm_engine()

    def program(self, program):
        """Set the program for this engine.

        Args:
            program: A Qasm object (from pecos.programs or pecos_rslib.programs)
        """
        # If it's a Python wrapper, extract the underlying Rust type
        if hasattr(program, "_to_program"):
            self._builder = self._builder.program(program._to_program())
        else:
            # It's already a Rust type
            self._builder = self._builder.program(program)
        return self

    def wasm(self, wasm_path: str):
        """Set the WebAssembly module for foreign function calls."""
        self._builder = self._builder.wasm(wasm_path)
        return self

    def to_sim(self):
        """Convert to simulation builder."""
        return self._builder.to_sim()


class PhirJsonEngineBuilder:
    """Python wrapper for PHIR JSON engine builder.

    This wrapper accepts Python PhirJson objects from pecos.programs.

    Example:
        >>> from pecos import phir_json_engine, PhirJson
        >>> results = (
        ...     phir_json_engine()
        ...     .program(PhirJson('{"format": "PHIR/JSON", ...}'))
        ...     .to_sim()
        ...     .run(1000)
        ... )
    """

    def __init__(self):
        self._builder = pecos_rslib.phir_json_engine()

    def program(self, program):
        """Set the program for this engine.

        Args:
            program: A PhirJson object (from pecos.programs or pecos_rslib.programs)
        """
        # If it's a Python wrapper, extract the underlying Rust type
        if hasattr(program, "_to_program"):
            self._builder = self._builder.program(program._to_program())
        else:
            # It's already a Rust type
            self._builder = self._builder.program(program)
        return self

    def wasm(self, wasm_path: str):
        """Set the WebAssembly module for foreign function calls."""
        self._builder = self._builder.wasm(wasm_path)
        return self

    def to_sim(self):
        """Convert to simulation builder."""
        return self._builder.to_sim()


class QisEngineBuilder:
    """Python wrapper for QIS engine builder.

    This wrapper accepts Python Qis or Hugr objects from pecos.programs.

    Example:
        >>> from pecos import qis_engine, Qis
        >>> results = qis_engine().program(Qis(llvm_ir_code)).to_sim().run(1000)
    """

    def __init__(self):
        self._builder = pecos_rslib.qis_engine()

    def program(self, program):
        """Set the program for this engine.

        Args:
            program: A Qis or Hugr object (from pecos.programs or pecos_rslib.programs)
        """
        # If it's a Python wrapper, extract the underlying Rust type
        if hasattr(program, "_to_program"):
            self._builder = self._builder.program(program._to_program())
        else:
            # It's already a Rust type
            self._builder = self._builder.program(program)
        return self

    def selene_runtime(self):
        """Use Selene simple runtime."""
        self._builder = self._builder.selene_runtime()
        return self

    def interface(self, builder):
        """Set the interface builder."""
        self._builder = self._builder.interface(builder)
        return self

    def to_sim(self):
        """Convert to simulation builder."""
        return self._builder.to_sim()


def qasm_engine():
    """Create a QASM engine builder.

    Returns:
        QasmEngineBuilder: A builder for QASM simulations.

    Example:
        >>> from pecos import qasm_engine, Qasm
        >>> results = (
        ...     qasm_engine()
        ...     .program(Qasm("OPENQASM 2.0; qreg q[2]; ..."))
        ...     .to_sim()
        ...     .run(1000)
        ... )
    """
    return QasmEngineBuilder()


def phir_json_engine():
    """Create a PHIR JSON engine builder.

    Returns:
        PhirJsonEngineBuilder: A builder for PHIR JSON simulations.

    Example:
        >>> from pecos import phir_json_engine, PhirJson
        >>> results = (
        ...     phir_json_engine()
        ...     .program(PhirJson('{"format": "PHIR/JSON", ...}'))
        ...     .to_sim()
        ...     .run(1000)
        ... )
    """
    return PhirJsonEngineBuilder()


def qis_engine():
    """Create a QIS engine builder.

    Returns:
        QisEngineBuilder: A builder for QIS/HUGR simulations.

    Example:
        >>> from pecos import qis_engine, Qis
        >>> results = qis_engine().program(Qis(llvm_ir_code)).to_sim().run(1000)
    """
    return QisEngineBuilder()


# Re-export noise and quantum engine builders from pecos_rslib
# These don't need wrappers since they don't take program types
depolarizing_noise = pecos_rslib.depolarizing_noise
biased_depolarizing_noise = pecos_rslib.biased_depolarizing_noise
general_noise = pecos_rslib.general_noise
state_vector = pecos_rslib.state_vector
sparse_stabilizer = pecos_rslib.sparse_stabilizer

# Re-export noise model builder classes for direct instantiation
GeneralNoiseModelBuilder = pecos_rslib.GeneralNoiseModelBuilder


# Check for Guppy availability (guppylang is an optional dependency)
def get_guppy_backends() -> dict:
    """Get available Guppy backends.

    Returns a dict with:
        - guppy_available: True if guppylang is installed
        - rust_backend: Always True (HUGR support is built into pecos-rslib)
    """
    result = {"guppy_available": False, "rust_backend": True}
    try:
        import guppylang

        result["guppy_available"] = True
    except ImportError:
        pass
    return result


__all__ = [
    "BinArray",  # Deprecated - use BitInt instead
    "BitInt",
    # Noise model builder classes
    "GeneralNoiseModelBuilder",
    # Program wrapper classes for sim() - also available via pecos.programs
    "Guppy",
    "Hugr",
    "HybridEngine",
    "PhirJson",
    "Qasm",
    "Qis",
    "QuantumCircuit",
    "Wasm",
    "WasmForeignObject",
    "Wat",
    "__version__",
    # Engine builders - accept Python program wrappers
    "biased_depolarizing_noise",
    "circuit_converters",
    "circuit_runners",
    "circuits",
    "complex64",
    "complex128",
    "decoders",
    "depolarizing_noise",
    # Keep dtypes module for dtype instances
    "dtypes",
    "engines",
    "error_models",
    "f32",
    "f64",
    "general_noise",
    "get_guppy_backends",
    # Scalar type classes (NumPy-like API)
    "i8",
    "i16",
    "i32",
    "i64",
    "misc",
    "num",  # Numerical computing module from pecos_rslib
    # Engine builder functions
    "phir_json_engine",
    "programs",  # Quantum program types (Qasm, Qis, etc.)
    "protocols",
    "qasm_engine",
    "qeccs",
    "qis_engine",
    # Guppy integration
    "sim",
    "simulators",
    "sparse_stabilizer",
    "state_vector",
    "tools",
    "typing",  # Type hints for arrays and scalars
    "u8",
    "u16",
    "u32",
    "u64",
]
