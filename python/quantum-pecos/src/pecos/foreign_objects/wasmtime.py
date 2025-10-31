"""Wasmtime WebAssembly runtime integration for PECOS.

This module provides integration with the Wasmtime WebAssembly runtime, enabling high-performance execution of WASM
modules for classical computations within the PECOS quantum error correction framework. It supports advanced features
like timeout handling, resource limits, and secure sandboxed execution of WebAssembly code.
"""

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

from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING

from wasmtime import Config, Engine, FuncType, Instance, Module, Store, Trap, TrapCode

from pecos.errors import MissingCCOPError, WasmRuntimeError
from pecos.foreign_objects.wasm_execution_timer_thread import (
    WASM_EXECUTION_MAX_TICKS,
    WASM_EXECUTION_TICK_LENGTH_S,
    WasmExecutionTimerThread,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class WasmtimeObj:
    """Wrapper class to create a wasmtime instance and access its functions.

    For more info on using Wasmer, see: https://wasmerio.github.io/wasmer-python/api/wasmer/wasmer.html
    """

    def __init__(self, file: str | bytes | Path) -> None:
        """Initialize a WasmtimeObj.

        Args:
        ----
            file: Path to WASM file, file bytes, or Path object to load.
        """
        if isinstance(file, str | Path):
            with Path.open(Path(file), "rb") as f:
                wasm_bytes = f.read()
        else:
            wasm_bytes = file

        self.wasm_bytes = wasm_bytes

        self.store = None
        self.module = None
        self.instance = None
        self.func_names = None

        self.spin_up_wasm()

    def init(self) -> None:
        """Initialize object before running a series of experiments."""
        self.new_instance()
        self.get_funcs()

        if "init" not in self.get_funcs():
            msg = "Missing `init()` from Wasm module."
            raise Exception(msg)

        self.exec("init", [])

    def shot_reinit(self) -> None:
        """Call before each shot to, e.g., reset variables."""
        if "shot_reinit" in self.get_funcs():
            self.exec("shot_reinit", [])

    def new_instance(self) -> None:
        """Reset object internal state."""
        self.instance = Instance(self.store, self.module, [])

    def spin_up_wasm(self) -> None:
        """Initialize the WASM module with epoch interruption and start timer thread."""
        config = Config()
        config.epoch_interruption = True
        engine = Engine(config)
        self.store = Store(engine)
        self.module = Module(self.store.engine, self.wasm_bytes)
        self.stop_flag = Event()
        self.inc_thread_handle = WasmExecutionTimerThread(
            self.stop_flag,
            self._increment_engine,
        )
        self.inc_thread_handle.start()
        self.new_instance()

    def get_funcs(self) -> list[str]:
        """Get list of function names exported by the WASM module.

        Returns:
            List of function names available for execution.
        """
        if self.func_names is None:
            fs = [
                str(f.name) for f in self.module.exports if isinstance(f.type, FuncType)
            ]

            self.func_names = fs

        return self.func_names

    def _increment_engine(self) -> None:
        self.store.engine.increment_epoch()

    def exec(self, func_name: str, args: Sequence) -> tuple:
        """Execute a function in the WASM module with timeout protection.

        Args:
            func_name: Name of the function to execute.
            args: Sequence of arguments to pass to the function.

        Returns:
            Tuple containing the function result.

        Raises:
            MissingCCOPError: If function not found in WASM module.
            WasmRuntimeError: If WASM execution fails or times out.
        """
        try:
            func = self.instance.exports(self.store)[func_name]
        except KeyError as e:
            message = f"No method found with name {func_name} in WASM"
            raise MissingCCOPError(message) from e

        try:
            self.store.engine.increment_epoch()
            self.store.set_epoch_deadline(WASM_EXECUTION_MAX_TICKS)
            output = func(self.store, *args)
        except Trap as t:
            if t.trap_code is TrapCode.INTERRUPT:
                message = (
                    f"WASM error: WASM failed during run-time. Execution time of "
                    f"function '{func_name}' exceeded maximum "
                    f"{WASM_EXECUTION_MAX_TICKS * WASM_EXECUTION_TICK_LENGTH_S}s"
                )
            else:
                message = (
                    f"WASM error: WASM failed during run-time. Execution of "
                    f"function '{func_name}' resulted in {t.trap_code}\n"
                    f"{t.message}"
                )
            raise WasmRuntimeError(message) from t
        except Exception as e:
            message = (
                f"Error during execution of function '{func_name}' with args {args}"
            )
            raise WasmRuntimeError(message) from e

        return output

    def teardown(self) -> None:
        """Cleanup resources by stopping the timer thread."""
        self.stop_flag.set()
        self.inc_thread_handle.join()

    def __del__(self) -> None:
        """Ensure cleanup happens when object is garbage collected."""
        try:
            if (
                hasattr(self, "stop_flag")
                and hasattr(self, "inc_thread_handle")
                and not self.stop_flag.is_set()
            ):
                self.teardown()
        except Exception as e:  # noqa: BLE001
            # Broad exception handling is required in __del__ to prevent errors during interpreter shutdown.
            # Any exception type could be raised, and logging is not safe at this stage.
            del e  # Explicitly acknowledge we're ignoring the exception

    def to_dict(self) -> dict:
        """Convert the WasmtimeObj to a dictionary for serialization.

        Returns:
            Dictionary containing the object class and WASM bytes.
        """
        return {"fobj_class": WasmtimeObj, "wasm_bytes": self.wasm_bytes}

    @staticmethod
    def from_dict(wasmtime_dict: dict) -> WasmtimeObj:
        """Create a WasmtimeObj from a dictionary.

        Args:
            wasmtime_dict: Dictionary containing object class and WASM bytes.

        Returns:
            New WasmtimeObj instance.
        """
        return wasmtime_dict["fobj_class"](wasmtime_dict["wasm_bytes"])
