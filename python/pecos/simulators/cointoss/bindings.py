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

from pecos.simulators.cointoss.gates import ignore_gate, measure

# Supporting gates from table:
#   https://github.com/CQCL/phir/blob/main/spec.md#table-ii---quantum-operations

gate_dict = {
    "Init": ignore_gate,
    "Measure": measure,
    "I": ignore_gate,
    "X": ignore_gate,
    "Y": ignore_gate,
    "Z": ignore_gate,
    "RX": ignore_gate,
    "RY": ignore_gate,
    "RZ": ignore_gate,
    "R1XY": ignore_gate,
    "SX": ignore_gate,
    "SXdg": ignore_gate,
    "SY": ignore_gate,
    "SYdg": ignore_gate,
    "SZ": ignore_gate,
    "SZdg": ignore_gate,
    "H": ignore_gate,
    "F": ignore_gate,
    "Fdg": ignore_gate,
    "T": ignore_gate,
    "Tdg": ignore_gate,
    "CX": ignore_gate,
    "CY": ignore_gate,
    "CZ": ignore_gate,
    "RXX": ignore_gate,
    "RYY": ignore_gate,
    "RZZ": ignore_gate,
    "R2XXYYZZ": ignore_gate,
    "SXX": ignore_gate,
    "SXXdg": ignore_gate,
    "SYY": ignore_gate,
    "SYYdg": ignore_gate,
    "SZZ": ignore_gate,
    "SZZdg": ignore_gate,
    "SWAP": ignore_gate,
}
