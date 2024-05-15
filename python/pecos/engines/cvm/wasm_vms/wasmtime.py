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

from __future__ import annotations

import contextlib

from pecos.engines.cvm.sim_func import sim_funcs

with contextlib.suppress(ImportError):
    from pecos.foreign_objects.wasmtime import WasmtimeObj


def read_wasmtime(path: str | bytes):
    """Helper method to create a wasmtime instance."""

    class WASM:
        """Helper class to provide the same interface as other Wasm objects."""

        def __init__(self, _path: str | bytes):
            self.wasmtime = WasmtimeObj(_path)
            self.wasmtime.init()

        def get_funcs(self):
            return self.wasmtime.get_funcs()

        def exec(self, func_name, args, debug=False):
            if debug and func_name.startswith("sim_"):
                method = sim_funcs[func_name]
                return method(*args)

            else:
                args = [int(b) for _, b in args]
                return self.wasmtime.exec(func_name, args)

        def teardown(self):
            self.wasmtime.teardown()

    return WASM(path)
