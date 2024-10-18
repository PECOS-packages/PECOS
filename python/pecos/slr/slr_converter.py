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

from typing import TYPE_CHECKING

from pecos.slr.gen_codes import Language, QASMGenerator, QIRGenerator

if TYPE_CHECKING:
    from pecos.slr.gen_codes import Generator


class SlrConverter:

    def __init__(self, block):
        self._block = block

    def generate(self, target: Language, skip_headers: bool = False) -> str:  # noqa: FBT001
        generator: Generator = None
        if target == Language.QASM:
            generator = QASMGenerator(skip_headers=skip_headers)
        elif target == Language.QIR:
            generator = QIRGenerator()
        else:
            msg = f"Code gen target '{target}' is not supported."
            raise NotImplementedError(msg)

        generator.generate_block(self._block)
        return generator.get_output()

    def qasm(self, skip_headers: False = False):
        return self.generate(Language.QASM, skip_headers)

    def qir(self):
        return self.generate(Language.QIR)
