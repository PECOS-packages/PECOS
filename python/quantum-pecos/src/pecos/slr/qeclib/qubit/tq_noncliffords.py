"""Two-qubit non-Clifford gate implementations.

This module provides two-qubit non-Clifford gate implementations that
extend beyond the Clifford group, enabling universal quantum computation
when combined with Clifford gates in quantum algorithms.
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

from pecos.slr.qeclib.qubit.qgate_base import TQGate


class CH(TQGate):
    """Controlled-Hadamard gate.

    This gate applies a Hadamard gate to the target qubit if the control qubit is in state |1⟩.
    """
