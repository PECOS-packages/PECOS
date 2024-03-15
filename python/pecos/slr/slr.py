# Copyright 2023 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.


from pecos import __version__
from pecos.slr.block import Block


class Main(Block):
    """
    A simple representation for Hybrid Quantum and Classical (HyQC) programs.

    This serves as the entry point for the program.
    """

    def __init__(self, *args, vargs=None, ops=None):
        super().__init__(*args, ops=ops, vargs=vargs)
        self.num_qubits = None

    def get_var(self, sym: str):
        """Returns a variable object whose name matches the string provided."""
        return self.vars.get(sym)

    def qasm(self, include="hqslib1.inc", header=None):
        def stamp_version(qasm):
            qasm.append(f"// Generated using: PECOS version {__version__}")

        qasm = []
        if header is not None:
            qasm.append(header)
            stamp_version(qasm)
        else:
            qasm.append("OPENQASM 2.0;")
            if include is not None:
                qasm.append(f'include "{include}";')

            stamp_version(qasm)

            for v in self.vars.vars:
                qasm.append(v.qasm())

        for op in self.ops:
            qm = op.qasm()
            if qm is not None:
                qasm.append(qm)

        qasm = "\n".join(qasm)

        return qasm


class CFunc: ...
