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
import pytest
from pecos.reps.pyphir.name_resolver import sim_name_resolver
from pecos.reps.pyphir.op_types import QOp


@pytest.mark.parametrize(
    ("gate", "args", "positive", "negative"),
    [
        pytest.param("RZZ", [(0, 1), (2, 3)], "SZZ", "SZZdg", id="RZZ"),
        pytest.param("RZ", [0, 1, 2, 3], "SZ", "SZdg", id="RZ"),
    ],
)
@pytest.mark.parametrize(
    ("angle", "resolution"),
    [
        pytest.param(0.0, "I", id="identity"),
        pytest.param(pc.f64.frac_pi_2, "positive", id="positive-clifford"),
        pytest.param(-pc.f64.frac_pi_2, "negative", id="negative-clifford"),
        pytest.param(pc.f64.frac_pi_4, "unchanged", id="positive-non-clifford"),
        pytest.param(-pc.f64.frac_pi_4, "unchanged", id="negative-non-clifford"),
    ],
)
def test_z_rotation_resolution(gate, args, positive, negative, angle, resolution) -> None:
    """Only Clifford rotations lower to discrete simulator gates."""
    expected = {"I": "I", "positive": positive, "negative": negative, "unchanged": gate}[resolution]
    assert sim_name_resolver(QOp(name=gate, angles=(angle,), args=args)) == expected


def test_rxy1q2x() -> None:
    """Verify that a R1XY(pi, 0) will give back X."""
    qop = QOp(name="R1XY", angles=(pc.f64.pi, 0.0), args=[0, 1, 2, 3])
    assert sim_name_resolver(qop) == "X"


def test_rxy1q2sydg() -> None:
    """Verify that a R1XY(-pi/2,pi/2) will give back SYdg."""
    qop = QOp(
        name="R1XY",
        angles=(-pc.f64.frac_pi_2, pc.f64.frac_pi_2),
        args=[0, 1, 2, 3],
    )
    assert sim_name_resolver(qop) == "SYdg"


def test_rxy1q2i() -> None:
    """Verify that a R1XY(0, 0) will give back I."""
    qop = QOp(name="R1XY", angles=(0.0, 0.0), args=[0, 1, 2, 3])
    assert sim_name_resolver(qop) == "I"


def test_rxy1q_non_clifford_uses_canonical_simulator_name() -> None:
    """A non-Clifford XY-plane rotation resolves to the canonical PECOS name, not the PHIR spelling."""
    qop = QOp(name="R1XY", angles=(0.123, 0.456), args=[])
    assert sim_name_resolver(qop) == "RXY1Q"
