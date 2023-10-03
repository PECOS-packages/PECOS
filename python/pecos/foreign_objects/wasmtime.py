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

from typing import Any, Optional, Sequence, Union

from wasmtime import Store, Module, Instance, Func, FuncType

from .foreign_object_abc import ForeignObject


class WasmtimeObj(ForeignObject):
    """Wrapper class to create a wasmtime instance and access its functions.

    For more info on using Wasmer, see: https://wasmerio.github.io/wasmer-python/api/wasmer/wasmer.html
    """

    def __init__(self,
                 file: Union[str, bytes]) -> None:

        if isinstance(file, str):
            with open(file, 'rb') as f:
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
            raise Exception("Missing `init()` from Wasm module.")

        self.exec("init", [])

    def shot_reinit(self) -> None:
        """Call before each shot to, e.g., reset variables."""
        if "shot_reinit" in self.get_funcs():
            self.exec("shot_reinit", [])

    def new_instance(self) -> None:
        """Reset object internal state."""
        self.instance = Instance(self.store, self.module, [])

    def spin_up_wasm(self):

        self.store = Store()
        self.module = Module(self.store.engine, self.wasm_bytes)
        self.new_instance()

    def get_funcs(self):

        if self.func_names is None:
            fs = []
            for f in self.module.exports:
                if isinstance(f.type, FuncType):
                    fs.append(str(f.name))

            self.func_names = fs

        return self.func_names

    def exec(self,
             func_name: str,
             args: Sequence) -> Any:

        func = self.instance.exports(self.store)[func_name]
        return func(self.store, *args)

    def to_dict(self):
        return {"fobj_class": WasmtimeObj, "wasm_bytes": self.wasm_bytes}

    @staticmethod
    def from_dict(wasmer_dict: dict):
        return wasmer_dict["fobj_class"](wasmer_dict["wasm_bytes"])
