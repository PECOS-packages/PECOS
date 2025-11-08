# Copyright 2022 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Wasmtime WebAssembly runtime integration for PECOS.

This module provides integration with the Wasmtime WebAssembly runtime, enabling high-performance execution of WASM
modules for classical computations within the PECOS quantum error correction framework.

This is now a thin wrapper around the Rust implementation (RsWasmForeignObject) from pecos-rslib,
which provides better performance and thread safety compared to the previous Python implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pecos_rslib._pecos_rslib import RsWasmForeignObject

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


class WasmtimeObj:
    """Wrapper class for Wasmtime WebAssembly runtime using Rust implementation.

    This class provides a Python-friendly interface to the Rust-based WasmForeignObject,
    maintaining API compatibility with the previous Python implementation.

    The Rust implementation provides:
    - Better performance through native code execution
    - Thread-safe operation with RwLock/Mutex synchronization
    - Configurable timeout (default: 1 second to match old Python version)
    - Configurable memory limits (default: unlimited)
    - Support for both i32 and i64 parameter types
    """

    def __init__(
        self,
        file: str | bytes | Path,
        timeout: float | None = None,
        memory_size: int | None = None,
    ) -> None:
        """Initialize a WasmtimeObj using the Rust implementation.

        Args:
            file: Path to WASM file (.wasm or .wat), file bytes, or Path object to load.
                  WAT files are automatically compiled to WASM by the Rust runtime.
            timeout: Optional timeout in seconds for WASM execution (default: 1.0 second).
            memory_size: Optional maximum memory size in bytes per linear memory (default: None = unlimited).
                        For example, 10 * 1024 * 1024 for 10 MB limit.
        """
        # Create the Rust object with optional timeout and memory limit
        self._rust_obj = RsWasmForeignObject(
            file,
            timeout=timeout,
            memory_size=memory_size,
        )

        # Get WASM bytes for compatibility with serialization
        self.wasm_bytes = self._rust_obj.to_dict()["wasm_bytes"]

    def init(self) -> None:
        """Initialize object before running a series of experiments.

        This creates a new WASM instance and calls the 'init' function.

        Raises:
            RuntimeError: If the 'init' function is not exported by the WASM module.
        """
        self._rust_obj.init()

    def shot_reinit(self) -> None:
        """Call before each shot to reset variables.

        This calls the 'shot_reinit' function in the WASM module if it exists.
        It's a no-op if the function is not present.
        """
        self._rust_obj.shot_reinit()

    def new_instance(self) -> None:
        """Reset object internal state by creating a new WASM instance."""
        self._rust_obj.new_instance()

    def get_funcs(self) -> list[str]:
        """Get list of function names exported by the WASM module.

        Returns:
            List of function names available for execution.
        """
        return self._rust_obj.get_funcs()

    def exec(self, func_name: str, args: Sequence) -> tuple:
        """Execute a function in the WASM module with timeout protection.

        Args:
            func_name: Name of the function to execute.
            args: Sequence of arguments to pass to the function (will be converted to i64).

        Returns:
            Tuple containing the function result(s). Single values are returned as (value,).

        Raises:
            RuntimeError: If function not found or execution fails/times out.

        Notes:
            The Rust implementation automatically handles i32/i64 type conversion based on
            the function signature, with bounds checking for i32 parameters.

            Default timeout is 1 second, but can be configured via the constructor.
        """
        # Convert args to list of i64
        args_list = [int(a) for a in args]

        # Execute via Rust - it returns either a single value or tuple
        result = self._rust_obj.exec(func_name, args_list)

        # Ensure we always return a tuple for API compatibility
        if isinstance(result, (list, tuple)):
            return tuple(result)
        return (result,)

    def teardown(self) -> None:
        """Cleanup resources by stopping the epoch increment thread."""
        self._rust_obj.teardown()

    def __del__(self) -> None:
        """Ensure cleanup happens when object is garbage collected."""
        try:
            if hasattr(self, "_rust_obj"):
                self._rust_obj.teardown()
        except Exception:  # noqa: BLE001, S110
            # Broad exception handling is required in __del__ to prevent errors during
            # interpreter shutdown. We silently ignore all exceptions.
            pass

    def to_dict(self) -> dict:
        """Convert the WasmtimeObj to a dictionary for serialization.

        Returns:
            Dictionary containing the object class and WASM bytes for pickling.
        """
        return {"fobj_class": WasmtimeObj, "wasm_bytes": self.wasm_bytes}

    @staticmethod
    def from_dict(wasmtime_dict: dict) -> WasmtimeObj:
        """Create a WasmtimeObj from a dictionary (for unpickling).

        Args:
            wasmtime_dict: Dictionary containing object class, WASM bytes, and optionally timeout and memory_size.

        Returns:
            New WasmtimeObj instance.
        """
        # Get timeout and memory_size if present (for backward compatibility, defaults are handled by __init__)
        timeout = wasmtime_dict.get("timeout")
        memory_size = wasmtime_dict.get("memory_size")
        return wasmtime_dict["fobj_class"](
            wasmtime_dict["wasm_bytes"],
            timeout=timeout,
            memory_size=memory_size,
        )
