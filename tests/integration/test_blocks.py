# Copyright 2023 The PECOS developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.
"""Tests to ensure the sequence of operations are as expected."""
from pecos.classical_interpreters.phir_classical_interpreter import PHIRClassicalInterpreter


def get_seq(program):
    """Get the sequences of operations produced by using the PHIR interpreter."""
    interp = PHIRClassicalInterpreter()
    interp.init(program)

    ops_seq = []
    for buffered_ops in interp.execute(interp.program.ops):
        subseq = []

        # Create a simple identifier for operations
        for op in buffered_ops:
            op_ident = [op.name]
            if hasattr(op, "metadata") and op.metadata.get("angles") is not None:
                op_ident.append(op.metadata["angles"])
            if hasattr(op, "args") and op.args is not None:
                op_ident.append(op.args)
            if hasattr(op, "returns") and op.returns is not None:
                op_ident.append(op.returns)
            subseq.append(tuple(op_ident))
        ops_seq.append(subseq)
    return ops_seq


def test_seq():
    program = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "metadata": {"source": "pytket-phir v0.2.0", "strict_parallelism": "true"},
        "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
            {"data": "cvar_define", "data_type": "u32", "variable": "m", "size": 2},
            {"qop": "RZ", "angles": [[1.0], "pi"], "args": [["q", 0], ["q", 1]]},
            {
                "block": "sequence",
                "ops": [
                    {"qop": "R1XY", "angles": [[0.5, 0.5], "pi"], "args": [["q", 0]]},
                    {"qop": "Measure", "args": [["q", 0], ["q", 1]], "returns": [["m", 0], ["m", 1]]},
                    {"qop": "R1XY", "angles": [[0.5, 0.5], "pi"], "args": [["q", 0]]},
                ],
            },
            {"qop": "Measure", "args": [["q", 0], ["q", 1]], "returns": [["m", 0], ["m", 1]]},
            {"qop": "X", "args": [["q", 0]]},
        ],
    }

    seq = get_seq(program)

    assert seq == [
        [
            ("RZ", [3.141592653589793], [0, 1]),
            ("R1XY", [1.5707963267948966, 1.5707963267948966], [0]),
            ("Measure", [0, 1], [["m", 0], ["m", 1]]),
        ],
        [
            ("R1XY", [1.5707963267948966, 1.5707963267948966], [0]),
        ],
        [
            ("Measure", [0, 1], [["m", 0], ["m", 1]]),
        ],
        [
            ("X", [0]),
        ],
    ]
