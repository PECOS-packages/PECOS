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

import contextlib
from io import BytesIO

with contextlib.suppress(ImportError):
    import pywasm


def read_pywasm(wasm):
    """Read in either a file path or byte object meant for use with pywasm to define the ccop."""
    if isinstance(wasm, str):
        p = pywasm.load(wasm)
    else:
        reader = BytesIO(wasm)
        module = pywasm.binary.Module.from_reader(reader)
        p = pywasm.Runtime(module)

    class PywasmReader:
        def __init__(self, p) -> None:
            self.p = p
            self.func_exports = self.get_funcs()

        def get_funcs(self):
            fs = []
            for f in self.p.machine.module.export_list:
                if str(f.value).startswith("FunctionAddress"):
                    fs.append(str(f.name))

            return fs

        def exec(self, func, args, debug=False):
            args = [int(b) for _, b in args]
            return self.p.exec(func, args)

        def teardown(self):
            pass  # Only needed for wasmtime

    return PywasmReader(p)
