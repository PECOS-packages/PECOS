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

"""Deprecated emitters still spell literals correctly and restore loop bindings."""

import numpy as np
import pytest
from pecos.slr import CReg, For, Main, QReg
from pecos.slr.cops import NEG
from pecos.slr.gen_codes import QASMGenerator, QuantumCircuitGenerator
from pecos.slr.gen_codes.gen_qasm import _param_qasm
from pecos.slr.qeclib import qubit as qb
from pecos.slr.vars import LoopVar


@pytest.mark.parametrize("value", [True, False, np.True_, np.False_])
@pytest.mark.parametrize("form", ["register", "bit", "binary", "unary", "parameter"])
def test_legacy_qasm_bool_literals(value, form: str) -> None:
    expected = str(int(value))
    if form == "parameter":
        assert _param_qasm(value) == expected
        return
    c = CReg("c", 2)
    operation, rendered = {
        "register": (c.set(value), f"c = {expected};"),
        "bit": (c[0].set(value), f"c[0] = {expected};"),
        "binary": (c.set(c + value), f"c = c + {expected};"),
        "unary": (c.set(NEG(value)), f"c = -{expected};"),
    }[form]
    generator = QASMGenerator(_internal=True)
    generator.generate_block(Main(c, operation))
    assert rendered in generator.get_output()


@pytest.mark.parametrize("form", ["bounds", "iterable"])
def test_legacy_quantum_circuit_empty_for(form: str) -> None:
    q = QReg("q", 1)
    loop = For("i", 0, 0) if form == "bounds" else For("i", range(0))
    generator = QuantumCircuitGenerator(_internal=True)
    generator.generate_block(Main(q, loop.Do(qb.X(q[0]))))
    assert len(generator.get_circuit()) == 0
    assert generator.loop_var_values == {}


@pytest.mark.parametrize("form", ["bounds", "iterable"])
@pytest.mark.parametrize("inner_count", [0, 1])
def test_legacy_nested_for_restores_outer_binding(form: str, inner_count: int) -> None:
    q, i = QReg("q", 2), LoopVar("i")
    inner = For(i, 0, inner_count) if form == "bounds" else For(i, range(inner_count))
    after_inner = qb.X(q[i])
    outer_values = []

    class BindingRecorder(QuantumCircuitGenerator):
        def _resolve_symbolic(self, op, loop_var, value):
            if op is after_inner:
                outer_values.append(self.loop_var_values[i])
            return super()._resolve_symbolic(op, loop_var, value)

    generator = BindingRecorder(_internal=True)
    generator.generate_block(Main(q, For(i, 0, 2).Do(inner.Do(qb.H(q[0])), after_inner)))
    assert outer_values == [0, 1]
    assert generator.loop_var_values == {}
    x_targets = [locations for symbol, locations, _ in generator.get_circuit().items() if symbol == "X"]
    assert x_targets == [{0}, {1}]
