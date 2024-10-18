# Copyright 2019 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

from projectq.ops import QubitOperator

from pecos.circuits import QuantumCircuit


def find_logical_signs(
    state,
    logical_circuit: QuantumCircuit,
    allow_float=False,
) -> int:
    """Find the sign of the logical operator.

    Args:
    ----
        state:
        logical_circuit:

    Returns:
    -------

    """
    if len(logical_circuit) != 1:
        msg = "Logical operators are expected to only have one tick."
        raise Exception(msg)

    logical_xs = set()
    logical_zs = set()

    op_string = []

    for symbol, gate_locations, _ in logical_circuit.items():
        if symbol == "X":
            logical_xs.update(gate_locations)
            for loc in gate_locations:
                op_string.append("X%s" % loc)
        elif symbol == "Z":
            logical_zs.update(gate_locations)
            for loc in gate_locations:
                op_string.append("Z%s" % loc)
        elif symbol == "Y":
            logical_xs.update(gate_locations)
            logical_zs.update(gate_locations)
            for loc in gate_locations:
                op_string.append("Y%s" % loc)
        else:
            raise Exception(
                'Can not currently handle logical operator with operator "%s"!'
                % symbol,
            )

    op_string = " ".join(op_string)
    state.eng.flush()
    result = state.eng.backend.get_expectation_value(
        QubitOperator(op_string),
        state.qureg,
    )

    if not allow_float:
        result = round(result, 5)
        if result == -1:
            return 1
        elif result == 1:
            return 0
        else:
            print("Operator being measured:", op_string)
            print("RESULT FOUND:", result)
            msg = "Unexpected result found!"
            raise Exception(msg)

    return result
