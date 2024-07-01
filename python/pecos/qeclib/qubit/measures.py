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

from pecos.qeclib.qubit.metaclasses import NoParamsQGate


class MeasureGate(NoParamsQGate):
    """A measurement of a qubit in the Z basis."""

    def __init__(self):
        super().__init__("Measure Z", qasm_sym="measure")
        self.cout = None
        self.qasm_sym = "measure"

    def __gt__(self, cout):
        g = self.copy()

        if isinstance(cout, tuple):
            g.cout = cout
        else:
            g.cout = (cout,)

        return g

    def qasm(self):
        sym = self.qasm_sym

        str_list = []
        for q, c in zip(self.qargs, self.cout, strict=True):
            str_list.append(f"{sym} {str(q)} -> {c};")

        return " ".join(str_list)


Measure = MeasureGate()
