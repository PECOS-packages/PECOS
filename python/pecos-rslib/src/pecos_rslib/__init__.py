# Copyright 2024 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""PECOS Rust library Python bindings.

This package provides Python bindings for high-performance Rust implementations of quantum simulators and computational
components within the PECOS framework, enabling efficient quantum circuit simulation and error correction computations.
"""

import ctypes
import logging
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, NoReturn

# Import all modules at the top to avoid E402 errors
from pecos_rslib._pecos_rslib import (
    adjust_tableau_string,  # Tableau formatting utility (Rust-optimized)
    array,  # Array creation function (no NumPy dependency)
    binding,  # llvmlite-compatible binding module for bitcode
    ByteMessage,
    ByteMessageBuilder,
    CppSparseSim,  # Pure Rust CppSparseSim (direct from Rust)
    dtypes,  # Rust-backed dtype system  # noqa: F401 - re-exported in __all__
    ir,  # llvmlite-compatible LLVM IR module
    num,  # Numerical computing functions (scipy.optimize replacements)
    Pauli,  # Quantum Pauli operators (I, X, Y, Z)
    PauliString,  # Multi-qubit Pauli operators
    QuestDensityMatrix,
    QuestStateVec,
    RsStateVec,  # Pure Rust state vector simulator (direct from Rust)
    RsWasmForeignObject,
    ShotMap,
    ShotVec,
    SparseSim,  # Pure Rust sparse stabilizer simulator (direct from Rust)
    SparseStabEngineRs,
    StateVecEngineRs,
    # Graph classes
    Graph,
    graph,  # Graph submodule
)
from pecos_rslib.rscoin_toss import CoinToss
from pecos_rslib.rspauli_prop import PauliPropRs
from pecos_rslib.typing import (
    Array,
    Complex,
    Float,
    Inexact,
    Integer,
    Numeric,
    SignedInteger,
    UnsignedInteger,
)
from pecos_rslib.num_wrapper import (  # Enhanced num module with axis support
    # Functions
    mean,
    sum,  # noqa: A004 - intentionally shadow builtin
    max,  # noqa: A004 - intentionally shadow builtin
    min,  # noqa: A004 - intentionally shadow builtin
    power,
    sqrt,
    exp,
    ln,  # Natural logarithm
    log,  # Logarithm with base
    abs,  # noqa: A004 - intentionally shadow builtin
    isnan,
    cos,
    sin,
    tan,
    sinh,
    cosh,
    tanh,
    asin,
    acos,
    atan,
    asinh,
    acosh,
    atanh,
    atan2,
    floor,
    ceil,
    round,  # noqa: A004 - intentionally shadow builtin
    isclose,
    allclose,
    array_equal,
    all,  # noqa: A004 - intentionally shadow builtin - Boolean AND reduction
    any,  # noqa: A004 - intentionally shadow builtin - Boolean OR reduction
    where,
    std,
    brentq,
    newton,
    polyfit,
    curve_fit,
    Poly1d,
    diag,
    linspace,
    arange,  # Python wrapper with dtype inference
    zeros,
    ones,
    delete,
    # Note: Mathematical constants (pi, tau, e, frac_pi_2, etc.) are NOT imported here
    # They are only available via dtype namespaces: pc.f64.pi, pc.f64.frac_pi_2, etc.
    # This makes precision explicit and allows for future f32, complex constants
    # Keep inf and nan for convenience (they're not precision-specific)
    inf,
    nan,
    # Submodules
    random,
    stats,  # Wrapped stats module with PECOS Array support
)

# Create backwards-compatible aliases - Python wrappers now point directly to Rust implementations
# This eliminates the need for Python wrapper files while maintaining API compatibility
CppSparseSimRs = CppSparseSim  # Alias for backwards compatibility
SparseSimRs = SparseSim  # Alias for backwards compatibility
StateVecRs = RsStateVec  # Alias for backwards compatibility

# Graph module is registered as a submodule in Rust (see graph_bindings.rs)
# Register in sys.modules for proper import support
sys.modules["pecos_rslib.graph"] = graph

# Import scalar type classes directly from dtypes submodule for NumPy-like API
# This allows: pc.i64(42) and def foo(x: pc.i64) just like np.int64(42) and def foo(x: np.int64)
# Access scalar classes via the .type attribute of dtype instances
i8 = dtypes.i8.type
i16 = dtypes.i16.type
i32 = dtypes.i32.type
i64 = dtypes.i64.type
u8 = dtypes.u8.type
u16 = dtypes.u16.type
u32 = dtypes.u32.type
u64 = dtypes.u64.type
f32 = dtypes.f32.type
f64 = dtypes.f64.type
complex64 = dtypes.complex64.type
complex128 = dtypes.complex128.type

# Mathematical constants are now defined as #[classattr] in Rust (dtypes.rs)
# for both ScalarF64 (f64.pi, etc.) and ScalarF32 (f32.pi, etc.)
# No Python-side setup is needed - constants are available directly on the type classes

# Import the num module directly from Rust - it already has all submodules set up correctly

# Add Python-friendly aliases (without underscores)
num.where = num.where_

# Register submodules in sys.modules so Python can find them
sys.modules["pecos_rslib.num"] = num
sys.modules["pecos_rslib.num.stats"] = (
    stats  # stats from num_wrapper (same as num.stats)
)
sys.modules["pecos_rslib.num.math"] = num.math
sys.modules["pecos_rslib.num.compare"] = num.compare
sys.modules["pecos_rslib.num.array"] = num.array
sys.modules["pecos_rslib.num.optimize"] = num.optimize
sys.modules["pecos_rslib.num.polynomial"] = num.polynomial
sys.modules["pecos_rslib.num.curve_fit"] = num.curve_fit
sys.modules["pecos_rslib.num.random"] = num.random
# Note: graph module is created synthetically above (no graph.py file needed)

# stats is exported from num_wrapper (which re-exports num.stats)

# HUGR compilation functions - explicit, no automatic fallback
try:
    from pecos_rslib._pecos_rslib import (
        compile_hugr_to_llvm as _compile_hugr_to_llvm_rust_impl,
    )

    def compile_hugr_to_llvm_rust(hugr_bytes: bytes, output_path=None) -> str:
        """PECOS's Rust HUGR to LLVM compiler.

        Args:
            hugr_bytes: HUGR program as bytes
            output_path: Optional path to write LLVM IR to file

        Returns:
            LLVM IR as string
        """
        # Call the Rust function (which only takes hugr_bytes)
        llvm_ir = _compile_hugr_to_llvm_rust_impl(hugr_bytes)

        # If output_path is provided, write to file
        if output_path is not None:
            from pathlib import Path

            Path(output_path).write_text(llvm_ir)

        return llvm_ir

except ImportError:

    def compile_hugr_to_llvm_rust(hugr_bytes: bytes, output_path=None) -> str:
        """PECOS's Rust HUGR to LLVM compiler."""
        raise ImportError(
            "PECOS's Rust HUGR compiler is not available. "
            "This should not happen - please report this as a bug."
        )


# Default to PECOS's Rust compiler
compile_hugr_to_llvm = compile_hugr_to_llvm_rust


try:
    from pecos_rslib.phir import PhirJsonEngine, PhirJsonSimulation

    _phir_imports_available = True
except ImportError:
    _phir_imports_available = False

    # Provide stubs
    class PhirJsonEngine:
        def __init__(self, *args, **kwargs):
            raise ImportError("PhirJsonEngine not available")

    class PhirJsonSimulation:
        def __init__(self, *args, **kwargs):
            raise ImportError("PhirJsonSimulation not available")


logger = logging.getLogger(__name__)


def _load_selene_runtime():
    """Load the Selene runtime library if available."""
    try:
        selene_paths = [
            # Use the real libselene.so from Selene repo
            "../selene/target/debug/libselene.so",
            "../selene/target/release/libselene.so",
            # Fallback paths
            "target/debug/libselene.so",
            "target/release/libselene.so",
        ]
        for path_str in selene_paths:
            if Path(path_str).exists():
                ctypes.CDLL(path_str, mode=ctypes.RTLD_GLOBAL)
                logger.info(f"Loaded Selene runtime from: {path_str}")
                return True
    except (OSError, ImportError, AttributeError) as err:
        logger.warning(f"Could not load Selene runtime: {err}")
        return False
    else:
        logger.warning("Could not load Selene runtime library")
        return False


# Load the Selene runtime library
_selene_loaded = _load_selene_runtime()

# Guppy conversion utilities - try importing but don't fail
try:
    from pecos_rslib.guppy_conversion import guppy_to_hugr
except ImportError:

    def guppy_to_hugr(*_args, **_kwargs):
        msg = "guppy_to_hugr not available"
        raise ImportError(msg)


# Program types - try importing but don't fail
try:
    from pecos_rslib.programs import (
        HugrProgram,
        QisProgram,
        PhirJsonProgram,
        QasmProgram,
        WasmProgram,
        WatProgram,
    )
except ImportError:
    # Provide stubs if not available
    class QasmProgram:
        @staticmethod
        def from_string(_qasm: str) -> "QasmProgram":
            msg = "QasmProgram not available"
            raise ImportError(msg)

    class QisProgram:
        @staticmethod
        def from_string(_llvm: str) -> "QisProgram":
            msg = "QisProgram not available"
            raise ImportError(msg)

    class HugrProgram:
        @staticmethod
        def from_bytes(_bytes: bytes) -> "HugrProgram":
            msg = "HugrProgram not available"
            raise ImportError(msg)

    class PhirJsonProgram:
        @staticmethod
        def from_json(_json: str) -> "PhirJsonProgram":
            msg = "PhirJsonProgram not available"
            raise ImportError(msg)

    class WasmProgram:
        @staticmethod
        def from_bytes(_bytes: bytes) -> "WasmProgram":
            msg = "WasmProgram not available"
            raise ImportError(msg)

    class WatProgram:
        @staticmethod
        def from_string(_wat: str) -> "WatProgram":
            msg = "WatProgram not available"
            raise ImportError(msg)


# Import the new sim API - use Python wrapper that handles Guppy
# Note: We explicitly override the sim module with the sim function
try:
    # Try to import the wrapper that handles Guppy programs
    from pecos_rslib.sim_wrapper import sim as _sim_func

    sim = _sim_func  # Override any module import with the function
except ImportError:
    # Fall back to sim from sim.py module (which re-exports Rust sim)
    try:
        from pecos_rslib.sim import sim as _sim_func

        sim = _sim_func  # Override any module import with the function
    except ImportError:
        # Last resort - try directly from Rust
        try:
            from pecos_rslib._pecos_rslib import sim as _sim_func

            sim = _sim_func  # Override any module import with the function
        except ImportError:

            def sim(*_args, **_kwargs) -> None:
                raise ImportError(
                    "sim() function not available - ensure pecos-rslib is built with sim support",
                )


# Try to import other sim-related functions but don't fail if unavailable
try:
    from pecos_rslib.sim import (
        BiasedDepolarizingNoiseModelBuilder,
        DepolarizingNoiseModelBuilder,
        GeneralNoiseModelBuilder,
        QisEngineBuilder,
        PhirJsonEngineBuilder,
        QasmEngineBuilder,
        SimBuilder,
        phir_json_engine,
        qasm_engine,
    )

    # Import QIS engine functions directly from Rust
    from pecos_rslib._pecos_rslib import (
        qis_engine,
        qis_helios_interface,
        qis_selene_helios_interface,
        QisInterfaceBuilder,
    )
except ImportError:
    # Provide stubs if not available
    def qasm_engine(*_args, **_kwargs) -> NoReturn:
        raise ImportError("qasm_engine not available")

    def qis_engine(*_args, **_kwargs) -> NoReturn:
        raise ImportError("qis_engine not available")

    def qis_helios_interface(*_args, **_kwargs) -> NoReturn:
        raise ImportError("qis_helios_interface not available")

    def qis_selene_helios_interface(*_args, **_kwargs) -> NoReturn:
        raise ImportError("qis_selene_helios_interface not available")

    class QisInterfaceBuilder:
        def __init__(self) -> None:
            raise ImportError("QisInterfaceBuilder not available")

    def phir_json_engine(*_args, **_kwargs) -> NoReturn:
        raise ImportError("phir_json_engine not available")

    # Builder classes
    class QasmEngineBuilder:
        def __init__(self) -> None:
            raise ImportError("QasmEngineBuilder not available")

    class QisEngineBuilder:
        def __init__(self) -> None:
            raise ImportError("QisEngineBuilder not available")

    class PhirJsonEngineBuilder:
        def __init__(self) -> None:
            raise ImportError("PhirJsonEngineBuilder not available")

    class SimBuilder:
        def __init__(self) -> None:
            raise ImportError("SimBuilder not available")

    class GeneralNoiseModelBuilder:
        def __init__(self) -> None:
            raise ImportError("GeneralNoiseModelBuilder not available")

    class DepolarizingNoiseModelBuilder:
        def __init__(self) -> None:
            raise ImportError("DepolarizingNoiseModelBuilder not available")

    class BiasedDepolarizingNoiseModelBuilder:
        def __init__(self) -> None:
            raise ImportError("BiasedDepolarizingNoiseModelBuilder not available")


# Import quantum engine builders from sim module - try but don't fail
try:
    from pecos_rslib.sim import (
        SparseStabilizerEngineBuilder,
        StateVectorEngineBuilder,
        biased_depolarizing_noise,
        depolarizing_noise,
        general_noise,
        sparse_stab,
        sparse_stabilizer,
        state_vector,
    )
except ImportError:
    # Provide stubs
    class StateVectorEngineBuilder:
        def __init__(self) -> None:
            raise ImportError("StateVectorEngineBuilder not available")

    class SparseStabilizerEngineBuilder:
        def __init__(self) -> None:
            raise ImportError("SparseStabilizerEngineBuilder not available")

    def state_vector(*_args, **_kwargs) -> NoReturn:
        raise ImportError("state_vector not available")

    def sparse_stabilizer(*_args, **_kwargs) -> NoReturn:
        raise ImportError("sparse_stabilizer not available")

    def sparse_stab(*_args, **_kwargs) -> NoReturn:
        raise ImportError("sparse_stab not available")

    def general_noise(*_args, **_kwargs) -> NoReturn:
        raise ImportError("general_noise not available")

    def depolarizing_noise(*_args, **_kwargs) -> NoReturn:
        raise ImportError("depolarizing_noise not available")

    def biased_depolarizing_noise(*_args, **_kwargs) -> NoReturn:
        raise ImportError("biased_depolarizing_noise not available")


# Import GeneralNoiseFactory and convenience functions - try but don't fail
try:
    from pecos_rslib.general_noise_factory import (
        GeneralNoiseFactory,
        IonTrapNoiseFactory,
        create_noise_from_dict,
        create_noise_from_json,
    )
except ImportError:
    # Provide stubs
    class GeneralNoiseFactory:
        def __init__(self) -> None:
            raise ImportError("GeneralNoiseFactory not available")

    def create_noise_from_dict(*_args, **_kwargs) -> NoReturn:
        raise ImportError("create_noise_from_dict not available")

    def create_noise_from_json(*_args, **_kwargs) -> NoReturn:
        raise ImportError("create_noise_from_json not available")

    class IonTrapNoiseFactory:
        def __init__(self) -> None:
            raise ImportError("IonTrapNoiseFactory not available")


# Import namespace modules for better discoverability - try but don't fail
try:
    from pecos_rslib import noise, programs, quantum
except ImportError:
    # Create empty namespace objects
    import types

    noise = types.ModuleType("noise")
    quantum = types.ModuleType("quantum")
    programs = types.ModuleType("programs")

# HUGR-LLVM pipeline is not currently available
RUST_HUGR_AVAILABLE = True  # Available via sim() API
HUGR_LLVM_PIPELINE_AVAILABLE = True  # Available via sim() API


def check_rust_hugr_availability() -> tuple[bool, str]:
    """Check if Rust HUGR backend is available."""
    # The sim() API handles HUGR internally, so we report it as available
    return True, "HUGR support available via sim() API"


def RustHugrCompiler(*_args, **_kwargs) -> NoReturn:
    raise ImportError("HUGR-LLVM pipeline not available")


def RustHugrLlvmEngine(*_args, **_kwargs) -> NoReturn:
    raise ImportError("HUGR-LLVM pipeline not available")


# The compile_hugr_to_llvm_rust function is imported from the Rust module above
# at line 44. We don't redefine it here to avoid overriding the real implementation.


def create_qis_engine_from_hugr_rust(*_args, **_kwargs) -> NoReturn:
    raise ImportError("HUGR-LLVM pipeline not available")


# All conditional imports are now at the top of the file


def get_compilation_backends() -> dict[str, Any]:
    """Get information about available compilation backends.

    Returns:
        dict: Dictionary with backend availability information
    """
    return {
        "default_backend": "phir",  # PHIR is the default backend
        "backends": {
            "phir": {
                "available": True,
                "description": "PHIR pipeline: HUGR → PHIR → LLVM IR",
                "dependencies": ["MLIR tools"],
            },
            "hugr-llvm": {
                "available": HUGR_LLVM_PIPELINE_AVAILABLE,
                "description": "HUGR-LLVM pipeline: HUGR → LLVM IR (via hugr-llvm)",
                "dependencies": ["hugr-llvm"],
            },
        },
    }


try:
    __version__ = version("pecos-rslib")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    # Main simulation API
    "sim",
    # Core simulators
    "SparseSimRs",
    "CppSparseSimRs",
    "CppSparseSim",  # Pure Rust CppSparseSim
    "SparseSim",  # Pure Rust SparseSim
    "StateVecRs",
    "RsStateVec",  # Pure Rust state vector simulator
    "CoinToss",
    "PauliPropRs",
    "ByteMessage",
    "ByteMessageBuilder",
    "StateVecEngineRs",
    "SparseStabEngineRs",
    # Array types
    "Array",  # Numpy-independent array type
    "array",  # Array creation function (no NumPy dependency)
    # Quantum types
    "Pauli",  # Pauli operators (I, X, Y, Z)
    "PauliString",  # Multi-qubit Pauli operators
    # llvmlite-compatible modules
    "ir",
    "binding",
    # Numerical computing (scipy.optimize replacements)
    "num",
    # Graph algorithms (MWPM decoder)
    "graph",  # Graph module
    "Graph",  # Graph class
    # Utility functions
    "adjust_tableau_string",  # Tableau formatting utility
    # Numerical functions from num_wrapper
    "mean",
    "sum",
    "max",
    "min",
    "power",
    "sqrt",
    "exp",
    "ln",  # Natural logarithm
    "log",  # Logarithm with base
    "abs",
    "isnan",
    "cos",
    "sin",
    "tan",
    "sinh",
    "cosh",
    "tanh",
    "asin",
    "acos",
    "atan",
    "asinh",
    "acosh",
    "atanh",
    "atan2",
    "floor",
    "ceil",
    "round",
    "isclose",
    "allclose",
    "array_equal",
    "all",  # Boolean AND reduction
    "any",  # Boolean OR reduction
    "where",
    "std",
    "brentq",
    "newton",
    "polyfit",
    "curve_fit",
    "Poly1d",
    "diag",
    "linspace",
    "arange",  # Python wrapper with dtype inference
    "zeros",
    "ones",
    "delete",
    # Note: Mathematical constants (pi, e, frac_pi_2, etc.) are NOT exported here
    # Access via dtype namespaces: pc.f64.pi, pc.f64.frac_pi_2, etc.
    # Only inf/nan are exported (not precision-specific)
    "inf",
    "nan",
    # Numerical submodules
    "random",
    "stats",
    # QuEST simulators
    "QuestStateVec",
    "QuestDensityMatrix",
    # WebAssembly foreign object
    "RsWasmForeignObject",
    # QIS engine (replaces Selene engine)
    "qis_engine",
    # QASM simulation - DEPRECATED: Use sim() instead
    # "NoiseModel",  # Deprecated
    # "QuantumEngine",  # Deprecated
    # "run_qasm",  # Deprecated - use sim()
    # "get_noise_models",  # Deprecated
    # "get_quantum_engines",  # Deprecated
    # "qasm_sim",  # Deprecated - use sim()
    # Shot result types
    "ShotVec",
    "ShotMap",
    "GeneralNoiseModelBuilder",
    "DepolarizingNoiseModelBuilder",
    "BiasedDepolarizingNoiseModelBuilder",
    # LLVM execution - currently not available
    # "execute_llvm",
    # "reset_llvm_runtime",
    # HUGR/LLVM compilation
    "compile_hugr_to_llvm",
    # Guppy conversion - may not be available
    # "guppy_to_hugr",
    # Program types
    "QasmProgram",
    "QisProgram",
    "HugrProgram",
    "PhirJsonProgram",
    "WasmProgram",
    "WatProgram",
    # Noise factory
    "GeneralNoiseFactory",
    "create_noise_from_dict",
    "create_noise_from_json",
    "IonTrapNoiseFactory",
    # HUGR-LLVM pipeline functionality
    "RustHugrCompiler",
    "RustHugrLlvmEngine",
    "compile_hugr_to_llvm_rust",
    "create_qis_engine_from_hugr_rust",
    "check_rust_hugr_availability",
    "RUST_HUGR_AVAILABLE",
    "HUGR_LLVM_PIPELINE_AVAILABLE",
    # PHIR pipeline functionality
    "PhirJsonEngine",
    "PhirJsonEngineBuilder",
    "PhirJsonProgram",
    "PhirJsonSimulation",
    "compile_hugr_to_llvm",
    "phir_json_engine",
    # Backend information
    "get_compilation_backends",
    # New sim API
    "sim",
    "qasm_engine",
    "qis_engine",
    "qis_helios_interface",
    "qis_selene_helios_interface",
    "QisInterfaceBuilder",
    "phir_json_engine",
    "QasmEngineBuilder",
    "QisEngineBuilder",
    "PhirJsonEngineBuilder",
    "SimBuilder",
    # Quantum engine builders
    "StateVectorEngineBuilder",
    "SparseStabilizerEngineBuilder",
    "state_vector",
    "sparse_stabilizer",
    "sparse_stab",
    # Noise builder free functions
    "general_noise",
    "depolarizing_noise",
    "biased_depolarizing_noise",
    # Namespace modules for discoverability
    "noise",
    "quantum",
    "programs",
    # Type aliases
    "Integer",
    "SignedInteger",
    "UnsignedInteger",
    "Float",
    "Complex",
    "Numeric",
    "Inexact",
    # Scalar type classes (NumPy-like API)
    "i8",
    "i16",
    "i32",
    "i64",
    "u8",
    "u16",
    "u32",
    "u64",
    "f32",
    "f64",
    "complex64",
    "complex128",
    # Keep dtypes module for dtype instances
    "dtypes",
]

# IMPORTANT: Override sim module with sim function
# This must be done after __all__ is defined to ensure the function is used
try:
    from pecos_rslib.sim_wrapper import sim as _sim_function

    sim = _sim_function
except ImportError:
    try:
        from pecos_rslib.sim import sim as _sim_function

        sim = _sim_function
    except ImportError:
        from pecos_rslib._pecos_rslib import sim as _sim_function

        sim = _sim_function
