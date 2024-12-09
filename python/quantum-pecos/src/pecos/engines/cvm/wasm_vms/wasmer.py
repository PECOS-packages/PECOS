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

import contextlib
import sys
from pathlib import Path

from pecos.engines.cvm.sim_func import sim_funcs

with contextlib.suppress(ImportError):
    from wasmer import Instance, Module, Store, engine

with contextlib.suppress(ImportError):
    from wasmer_compiler_cranelift import Compiler as CompilerCranelift

with contextlib.suppress(ImportError):
    from wasmer_compiler_llvm import Compiler as CompilerLLVM


def read_wasmer(path, compiler="wasm_cl"):
    """Helper method to create a wasmer instance."""

    class WasmerInstance:
        """Wrapper class to create a wasmer instance and access its functions."""

        if "wasmer" not in sys.modules:
            msg = 'wasmer is being called but not installed! Install "wasmer"'
            raise ImportError(msg)

        # '"wasmer_compiler_cranelift"!')

        def __init__(self, file: str | bytes, compiler="wasm_cl") -> None:
            if isinstance(file, str):
                with Path.open(file, "rb") as f:
                    wasm_b = f.read()
            else:
                wasm_b = file

            store = (
                Store(engine.JIT(CompilerLLVM))
                if compiler == "wasm_llvm"
                else Store(engine.JIT(CompilerCranelift))
            )

            module = Module(store, wasm_b)
            instance = Instance(module)

            self.wasm = instance
            self.module = module

        def get_funcs(self):
            fs = []
            for f in self.module.exports:
                if str(f.type).startswith("FunctionType"):
                    fs.append(str(f.name))

            return fs

        def exec(self, func_name, args, debug=False):
            if debug and func_name.startswith("sim_"):
                method = sim_funcs[func_name]
                return method(*args)

            else:
                method = getattr(self.wasm.exports, func_name)
                args = [int(b) for _, b in args]
                return method(*args)

        def teardown(self):
            pass  # Only needed for wasmtime

    return WasmerInstance(path, compiler)
