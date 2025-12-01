# Copyright 2021 The PECOS Developers
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Exception classes for PECOS quantum error correction framework.

This module defines custom exception types used throughout PECOS for error
handling, including base exceptions and specialized errors for simulators,
WASM integration, and classical coprocessor operations.
"""

import re


class PECOSError(Exception):
    """Base exception raised by PECOS."""


class ConfigurationError(PECOSError):
    """Indicates invalid configuration settings."""


class NotSupportedGateError(PECOSError):
    """Indicates a gate not supported by a simulator."""


class WasmError(PECOSError):
    """Base WASM-related exception type."""


class MissingCCOPError(WasmError):
    """Indicates missing a classical function library."""


class WasmRuntimeError(WasmError):
    """Indicates a runtime WASM error."""


class HugrTypeError(PECOSError):
    """Error raised when HUGR compilation encounters unsupported types."""

    def __init__(self, original_error: str) -> None:
        """Initialize HugrTypeError with the original error message."""
        self.original_error = original_error
        self.unsupported_type = self._extract_type(original_error)
        super().__init__(self._create_message())

    @staticmethod
    def _extract_type(error: str) -> str | None:
        """Extract the unsupported type from the error message."""
        # Pattern: "Unknown type: int(6)" or "Unknown type: bool"
        match = re.search(r"Unknown type: (\w+)(?:\((\d+)\))?", error)
        if match:
            type_name = match.group(1)
            width = match.group(2)
            if width:
                return f"{type_name}({width})"
            return type_name
        return None

    def _create_message(self) -> str:
        """Create a helpful error message."""
        base_msg = f"HUGR compilation failed: {self.original_error}"

        if self.unsupported_type:
            if self.unsupported_type.startswith("int"):
                return (
                    f"{base_msg}\n\n"
                    "Classical integer types are not yet supported in the HUGR→LLVM compiler.\n"
                    "Workarounds:\n"
                    "1. Use quantum operations that return measurement results (bool)\n"
                    "2. Perform classical computations outside the Guppy function\n"
                    "3. Wait for future updates to support classical types"
                )
            if self.unsupported_type == "bool":
                return (
                    f"{base_msg}\n\n"
                    "Direct boolean returns are not yet fully supported.\n"
                    "Workarounds:\n"
                    "1. Return measurement results from quantum operations\n"
                    "2. Use the function for quantum state preparation only"
                )

        return base_msg
