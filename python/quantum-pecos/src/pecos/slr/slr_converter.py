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

from pecos.slr.gen_codes.gen_qasm import QASMGenerator
from pecos.slr.gen_codes.language import Language

try:
    from pecos.slr.gen_codes.gen_qir import QIRGenerator
except ImportError:
    QIRGenerator = None


class SlrConverter:

    def __init__(self, block):
        self._block = block

    def generate(
        self,
        target: Language,
        *,
        skip_headers: bool = False,
        add_versions: bool = False,
    ) -> str:
        if target == Language.QASM:
            generator = QASMGenerator(skip_headers=skip_headers)
        elif target in [Language.QIR, Language.QIRBC]:
            self._check_qir_imported()
            generator = QIRGenerator()
        else:
            msg = f"Code gen target '{target}' is not supported."
            raise NotImplementedError(msg)

        generator.generate_block(self._block)
        if target == Language.QIRBC:

            return generator.get_bc()
        return generator.get_output()

    @staticmethod
    def _check_qir_imported():
        if QIRGenerator is None:
            msg = (
                "Trying to compile QIR without the appropriate optional dependencies install. "
                "Use optional dependency group `qir` or `all`"
            )
            raise Exception(
                msg,
            )

    def qasm(self, *, skip_headers: bool = False, add_versions: bool = False):
        return self.generate(
            Language.QASM,
            skip_headers=skip_headers,
            add_versions=add_versions,
        )

    def qir(self):
        self._check_qir_imported()
        return self.generate(Language.QIR)

    def qir_bc(self):
        self._check_qir_imported()
        return self.generate(Language.QIRBC)
