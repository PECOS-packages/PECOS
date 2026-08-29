"""Single-qubit non-Clifford gate implementations.

This module provides non-Clifford gate implementations including T gates
and other gates that extend beyond the Clifford group, enabling
universal quantum computation when combined with Clifford gates.
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

from pecos.slr.qeclib.qubit.qgate_base import QGate


class T(QGate):
    """Conventional T phase gate, diag(1, exp(iπ/4)).

    Its exact relationship to the symmetric rotation family is
    T = exp(iπ/8) RZ(π/4).
    """


class Tdg(QGate):
    """Conventional T-dagger phase gate, diag(1, exp(-iπ/4)).

    Its exact relationship to the symmetric rotation family is
    Tdg = exp(-iπ/8) RZ(-π/4).
    """
