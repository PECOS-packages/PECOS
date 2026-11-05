# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""A Python bool literal in a classical expression is emitted to OpenQASM as the bit value 1 or 0."""

import pytest
from pecos.slr import CReg, Main, QReg, SlrConverter
from pecos.slr.gen_codes.gen_qasm import QASMGenerator
from pecos_rslib import Qasm, sim, state_vector


def _ast_qasm(program: Main) -> str:
    return SlrConverter(program).qasm()


def _legacy_qasm(program: Main) -> str:
    generator = QASMGenerator(_internal=True)
    generator.generate_block(program)
    return generator.get_output()


@pytest.mark.parametrize("generate", [_ast_qasm, _legacy_qasm], ids=["ast", "legacy"])
@pytest.mark.parametrize("bit", [0, 1])
@pytest.mark.parametrize(
    ("build", "truth"),
    [
        # truth[bit] is the expected c[1] for c[0] == bit.
        (lambda _: True, (1, 1)),
        (lambda _: False, (0, 0)),
        (lambda b: b ^ True, (1, 0)),
        (lambda b: True ^ b, (1, 0)),
        (lambda b: b & True, (0, 1)),
    ],
)
def test_bool_literal_runs_as_its_bit_value(generate, bit, build, truth) -> None:
    # c[1] starts opposite to the expected result, so a literal read as 0 or left unset fails.
    q, c = QReg("q", 1), CReg("c", 2)
    want = truth[bit]
    program = Main(q, c, c[0].set(bit), c[1].set(1 - want), c[1].set(build(c[0])))
    result = sim(Qasm.from_string(generate(program))).quantum(state_vector()).seed(7).run(1).to_dict()
    assert result["c"] == [bit + 2 * want]
