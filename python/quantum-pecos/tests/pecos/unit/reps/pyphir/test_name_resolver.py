# Copyright 2025 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Tests for PyPHIR name resolver functionality."""

import pecos as pc
from pecos.reps.pyphir.name_resolver import sim_name_resolver
from pecos.reps.pyphir.op_types import QOp


def test_rzz2szz() -> None:
    """Verify that a RZZ(pi/2) gate will be resolved to a SZZ gate."""
    qop = QOp(name="RZZ", angles=(pc.f64.frac_pi_2,), args=[(0, 1), (2, 3)])
    assert sim_name_resolver(qop) == "SZZ"


def test_rzz2szzdg() -> None:
    """Verify that a RZZ(-pi/2) gate will be resolved to a SZZdg gate."""
    qop = QOp(name="RZZ", angles=(-pc.f64.frac_pi_2,), args=[(0, 1), (2, 3)])
    assert sim_name_resolver(qop) == "SZZdg"


def test_rzz2i() -> None:
    """Verify that a RZZ(0.0) gate will be resolved to an I gate."""
    qop = QOp(name="RZZ", angles=(0.0,), args=[(0, 1), (2, 3)])
    assert sim_name_resolver(qop) == "I"


def test_rzz2rzz() -> None:
    """Verify that a RZZ(pi/4) gate will be resolved to a RZZ gate since it is non-Clifford."""
    qop = QOp(name="RZZ", angles=(0.0,), args=[(0, 1), (2, 3)])
    assert sim_name_resolver(qop) == "I"


def test_rz2sz() -> None:
    """Verify that a RZ(pi/2) gate will be resolved to a SZ gate."""
    qop = QOp(name="RZ", angles=(pc.f64.frac_pi_2,), args=[0, 1, 2, 3])
    assert sim_name_resolver(qop) == "SZ"


def test_rz2szdg() -> None:
    """Verify that a RZ(-pi/2) gate will be resolved to a SZdg gate."""
    qop = QOp(name="RZ", angles=(-pc.f64.frac_pi_2,), args=[0, 1, 2, 3])
    assert sim_name_resolver(qop) == "SZdg"


def test_rz2i() -> None:
    """Verify that a RZ(0.0) gate will be resolved to an I gate."""
    qop = QOp(name="RZ", angles=(0.0,), args=[0, 1, 2, 3])
    assert sim_name_resolver(qop) == "I"


def test_rz2rz() -> None:
    """Verify that a RZ(pi/4) will give back RZ since it is non-Clifford."""
    qop = QOp(name="RZ", angles=(pc.f64.frac_pi_4,), args=[0, 1, 2, 3])
    assert sim_name_resolver(qop) == "RZ"


def test_r1xy2x() -> None:
    """Verify that a R1XY(pi, 0) will give back X."""
    qop = QOp(name="R1XY", angles=(pc.f64.pi, 0.0), args=[0, 1, 2, 3])
    assert sim_name_resolver(qop) == "X"


def test_r1xy2sydg() -> None:
    """Verify that a R1XY(-pi/2,pi/2) will give back SYdg."""
    qop = QOp(
        name="R1XY",
        angles=(-pc.f64.frac_pi_2, pc.f64.frac_pi_2),
        args=[0, 1, 2, 3],
    )
    assert sim_name_resolver(qop) == "SYdg"


def test_r1xy2i() -> None:
    """Verify that a R1XY(0, 0) will give back I."""
    qop = QOp(name="R1XY", angles=(0.0, 0.0), args=[0, 1, 2, 3])
    assert sim_name_resolver(qop) == "I"
