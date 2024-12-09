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

from wasmer import FunctionType, Instance, Module, Store, engine
from wasmer_compiler_cranelift import Compiler as Cranelift

from pecos.errors import MissingCCOPError, WasmRuntimeError
from pecos.foreign_objects.foreign_object_abc import ForeignObject

if TYPE_CHECKING:
    from collections.abc import Sequence


class WasmerObj(ForeignObject):
    """Wrapper class to create a Wasmer instance and access its functions.

    For more info on using Wasmer, see: https://wasmerio.github.io/wasmer-python/api/wasmer/wasmer.html
    """

    def __init__(
        self,
        file: str | bytes | Path,
        compiler: object | None = None,
    ) -> None:
        self.compiler = compiler

        if isinstance(file, (str, Path)):
            with Path.open(Path(file), "rb") as f:
                wasm_bytes = f.read()
        else:
            wasm_bytes = file

        self.wasm_bytes = wasm_bytes

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
        self.instance = Instance(self.module)

    def spin_up_wasm(self) -> None:
        compiler = self.compiler
        if compiler is None:
            compiler = Cranelift

        store = Store(engine.JIT(compiler))

        self.module = Module(store, self.wasm_bytes)
        self.new_instance()

    def get_funcs(self) -> list[str]:
        if self.func_names is None:
            fs = []
            for f in self.module.exports:
                if isinstance(f.type, FunctionType):
                    fs.append(str(f.name))

            self.func_names = fs

        return self.func_names

    def exec(self, func_name: str, args: Sequence) -> tuple:
        try:
            func = getattr(self.instance.exports, func_name)
        except AttributeError as e:
            message = f"Func {func_name} not found in WASM"
            raise MissingCCOPError(message) from e

        params = func.type.params
        if len(args) != len(params):
            msg = f"Wasmer function `{func_name}` takes {len(params)} args and {len(args)} were given!"
            raise WasmRuntimeError(msg)

        try:
            return func(*args)
        except Exception as ex:
            raise WasmRuntimeError(ex.args[0]) from ex

    def to_dict(self) -> dict:
        return {"fobj_class": WasmerObj, "wasm_bytes": self.wasm_bytes}

    @staticmethod
    def from_dict(wasmer_dict: dict) -> WasmerObj:
        return wasmer_dict["fobj_class"](wasmer_dict["wasm_bytes"])
