"""PECOS Quantum Programming Frontends.

This module provides frontends for various quantum programming languages
that compile to QIR for execution on PECOS.
"""

from typing import Any

from pecos.frontends.guppy_api import guppy_to_hugr, sim
from pecos.frontends.guppy_frontend import GuppyFrontend


# Helper function for backend checking
def get_guppy_backends() -> dict[str, Any]:
    """Get available Guppy backends."""
    result = {"guppy_available": False, "rust_backend": False}
    try:
        import guppylang

        result["guppy_available"] = True
        from _pecos_rslib import check_rust_hugr_availability

        rust_available, msg = check_rust_hugr_availability()
        result["rust_backend"] = rust_available
        result["rust_message"] = msg
    except ImportError:
        pass
    return result


__all__ = [
    "GuppyFrontend",
    "get_guppy_backends",
    "guppy_to_hugr",
    "sim",
]
