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
from typing import TYPE_CHECKING

from wasmtime import FuncType, Instance, Module, Store, Trap

from pecos.foreign_objects.foreign_object_abc import ForeignObject
from pecos.errors import WasmRuntimeError, MissingCCOPError

if TYPE_CHECKING:
    from collections.abc import Sequence


class WasmtimeObj(ForeignObject):
    """Wrapper class to create a wasmtime instance and access its functions.

    For more info on using Wasmer, see: https://wasmerio.github.io/wasmer-python/api/wasmer/wasmer.html
    """

    def __init__(self, file: str | bytes | Path) -> None:
        if isinstance(file, (str, Path)):
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
        self.store = Store()
        self.module = Module(self.store.engine, self.wasm_bytes)
        self.new_instance()

    def get_funcs(self) -> list[str]:
        if self.func_names is None:
            fs = []
            for f in self.module.exports:
                if isinstance(f.type, FuncType):
                    fs.append(str(f.name))

            self.func_names = fs

        return self.func_names

    def exec(self, func_name: str, args: Sequence) -> tuple:
        try:
            func = self.instance.exports(self.store)[func_name]
        except KeyError as e:
            raise MissingCCOPError(f"No method found with name {func_name} in WASM") from e
        
        try:
            return func(self.store, *args)
        except Trap as t:
            message = (f"Error during execution of function '{func_name}' with args: {args}\n"
                    f"Trap code: {t.trap_code}\n{t.message}")
            raise WasmRuntimeError(message) from t 
        except Exception as e:
            raise WasmRuntimeError(f"Error during execution of function {func_name} with args {args}") from e

    def to_dict(self) -> dict:
        return {"fobj_class": WasmtimeObj, "wasm_bytes": self.wasm_bytes}

    @staticmethod
    def from_dict(wasmtime_dict: dict) -> WasmtimeObj:
        return wasmtime_dict["fobj_class"](wasmtime_dict["wasm_bytes"])
