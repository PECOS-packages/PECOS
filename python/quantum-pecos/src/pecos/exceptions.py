# Copyright 2021 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Exception classes for PECOS.

This module provides a centralized location for all PECOS exception types,
following NumPy's pattern of having a dedicated exceptions module.

Example:
    >>> from pecos.exceptions import PECOSError, GateError
    >>>
    >>> try:
    ...     # Some operation that might fail
    ...     pass
    ... except GateError as e:
    ...     print(f"Gate error: {e}")
    ...
"""

from __future__ import annotations

# Import Rust-defined WasmError so Python WasmError can inherit from it.
# This allows catching either pecos_rslib.WasmError or pecos.exceptions.WasmError.
try:
    from pecos_rslib import WasmError as _RsWasmError
except ImportError:
    _RsWasmError = None


class PECOSError(Exception):
    """Base exception raised by PECOS."""


class PECOSTypeError(TypeError):
    """Type error in PECOS operations."""


class ConfigurationError(PECOSError):
    """Indicates invalid configuration settings."""


class NotSupportedGateError(PECOSError):
    """Indicates a gate not supported by a simulator."""


class GateError(PECOSError):
    """General gate errors."""


class GateOverlapError(GateError):
    """Raised when gates act on qudits that are already being acted on."""


class CircuitError(PECOSError):
    """Error in circuit construction or execution."""


class SimulationError(PECOSError):
    """Error during quantum simulation."""


class DecoderError(PECOSError):
    """Error in decoder operations."""


class QECCError(PECOSError):
    """Error in quantum error correcting code operations."""


# WasmError inherits from both PECOSError and the Rust-defined WasmError (when available).
# This means:
#   - Errors raised by Rust (pecos_rslib.WasmError) are catchable as pecos_rslib.WasmError
#   - Errors raised by Python (pecos.exceptions.WasmError) are catchable as both
#     PECOSError and pecos_rslib.WasmError
if _RsWasmError is not None:

    class WasmError(PECOSError, _RsWasmError):  # type: ignore[misc]
        """Base WASM-related exception type."""

else:

    class WasmError(PECOSError):  # type: ignore[no-redef]
        """Base WASM-related exception type."""


class MissingCCOPError(WasmError):
    """Indicates missing a classical function library."""


class WasmRuntimeError(WasmError):
    """Indicates a runtime WASM error."""


__all__ = [
    "CircuitError",
    "ConfigurationError",
    "DecoderError",
    "GateError",
    "GateOverlapError",
    "MissingCCOPError",
    "NotSupportedGateError",
    "PECOSError",
    "PECOSTypeError",
    "QECCError",
    "SimulationError",
    "WasmError",
    "WasmRuntimeError",
]
