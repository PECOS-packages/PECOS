"""Generic quantum error correction operations.

This package provides generic operations that can be used across different QEC codes.
"""

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

from importlib import import_module

from pecos.slr.gen_codes.gen_qasm import QASMGenerator
from pecos.slr.gen_codes.generator import Generator
from pecos.slr.gen_codes.language import Language


# QIRGenerator requires pecos-rslib-llvm; defer that native import until use.
def __getattr__(name: str) -> object:
    if name == "QIRGenerator":
        try:
            qir_generator = import_module("pecos.slr.gen_codes.gen_qir").QIRGenerator
        except ImportError:
            qir_generator = None
        globals()[name] = qir_generator
        return qir_generator
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def __dir__() -> list[str]:
    return sorted({*globals(), "QIRGenerator"})


__all__ = ["Generator", "Language", "QASMGenerator", "QIRGenerator"]
