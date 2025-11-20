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

"""Regression tests for R1XY gate operations with conditional execution.

These tests verify correct behavior of R1XY gates, particularly in conditional
contexts and with various angle combinations.

Background:
    GitHub issue #81 revealed a bug where R1XY gates with theta=2.0 were incorrectly
    classified as identity gates due to an operator precedence error:
    - Buggy: `theta % 2 * np.pi` → `(theta % 2) * np.pi` → 0.0 for theta=2.0
    - Fixed: `theta % (2 * np.pi)` → correctly checks if theta is a multiple of 2π

    This caused conditional R1XY gates with angles summing to 4π to not cancel
    properly, producing random results instead of deterministic zeros.
"""


import pecos as pc
from pecos.engines.hybrid_engine import HybridEngine


def test_r1xy_angles_summing_to_4pi_cancel() -> None:
    """Test that two R1XY gates with angles summing to 4π cancel completely.

    Regression test for GitHub issue #81.
    Uses angles 0.6366197723675814π and 3.3633802276324185π which sum to 4π.
    """
    phir = """{
    "format": "PHIR/JSON",
    "version": "0.1.0",
    "metadata": {"source": "test"},
    "ops": [
        {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 1},
        {"data": "cvar_define", "data_type": "i64", "variable": "a", "size": 1},
        {"data": "cvar_define", "data_type": "i64", "variable": "t", "size": 1},
        {
            "block": "if",
            "condition": {"cop": "==", "args": [["t", 0], 0]},
            "true_branch": [
                {"qop": "R1XY", "angles": [[0.6366197723675814, 0.0], "pi"], "args": [["q", 0]]}
            ]
        },
        {"qop": "R1XY", "angles": [[3.3633802276324185, 0.0], "pi"], "args": [["q", 0]]},
        {"qop": "Measure", "returns": [["a", 0]], "args": [["q", 0]]}
    ]
}"""

    engine = HybridEngine(qsim="state-vector")
    results = engine.run(phir, shots=100)

    zeros = results["a"].count("0")
    ones = results["a"].count("1")

    assert ones == 0, f"Expected all measurements to be 0, but got {ones} ones"
    assert zeros == 100, f"Expected 100 zeros, but got {zeros}"


def test_r1xy_theta_2pi_classified_as_identity() -> None:
    """Test that R1XY with theta=2π is correctly classified as identity.

    This verifies the operator precedence fix: theta % (2*pi) should be 0 for theta=2π.
    """
    phir = """{
    "format": "PHIR/JSON",
    "version": "0.1.0",
    "metadata": {"source": "test"},
    "ops": [
        {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 1},
        {"data": "cvar_define", "data_type": "i64", "variable": "a", "size": 1},
        {"qop": "R1XY", "angles": [[6.283185307179586, 0.0], "rad"], "args": [["q", 0]]},
        {"qop": "Measure", "returns": [["a", 0]], "args": [["q", 0]]}
    ]
}"""

    engine = HybridEngine(qsim="state-vector")
    results = engine.run(phir, shots=100)

    # R1XY with theta=2π should be identity, leaving qubit in |0⟩
    zeros = results["a"].count("0")
    assert (
        zeros == 100
    ), f"R1XY(2π, 0) should be identity, but got {100-zeros} non-zero results"


def test_r1xy_theta_2_not_identity() -> None:
    """Test that R1XY with theta=2.0 is NOT classified as identity.

    This is the specific bug case from issue #81 where theta=2.0 was incorrectly
    treated as identity due to `(theta % 2) * pi = 0`.
    """
    phir = """{
    "format": "PHIR/JSON",
    "version": "0.1.0",
    "metadata": {"source": "test"},
    "ops": [
        {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 1},
        {"data": "cvar_define", "data_type": "i64", "variable": "a", "size": 1},
        {"qop": "R1XY", "angles": [[2.0, 0.0], "rad"], "args": [["q", 0]]},
        {"qop": "Measure", "returns": [["a", 0]], "args": [["q", 0]]}
    ]
}"""

    engine = HybridEngine(qsim="state-vector")
    results = engine.run(phir, shots=100)

    # R1XY(2.0, 0) should rotate the qubit, not be identity
    # We expect a mix of 0s and 1s, not all 0s
    results["a"].count("0")
    ones = results["a"].count("1")

    # With a 2 radian rotation, we should NOT get all zeros
    assert ones > 0, "R1XY(2.0, 0) should not be identity - expected some 1s in results"


def test_r1xy_angles_summing_to_2pi_return_to_identity() -> None:
    """Test that R1XY gates with angles summing to 2π return to identity.

    Two R1XY gates with theta values summing to 2π should cancel (identity up to global phase).
    """
    phir = """{
    "format": "PHIR/JSON",
    "version": "0.1.0",
    "metadata": {"source": "test"},
    "ops": [
        {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 1},
        {"data": "cvar_define", "data_type": "i64", "variable": "a", "size": 1},
        {"data": "cvar_define", "data_type": "i64", "variable": "t", "size": 1},
        {
            "block": "if",
            "condition": {"cop": "==", "args": [["t", 0], 0]},
            "true_branch": [
                {"qop": "R1XY", "angles": [[1.0, 0.0], "pi"], "args": [["q", 0]]}
            ]
        },
        {"qop": "R1XY", "angles": [[1.0, 0.0], "pi"], "args": [["q", 0]]},
        {"qop": "Measure", "returns": [["a", 0]], "args": [["q", 0]]}
    ]
}"""

    engine = HybridEngine(qsim="state-vector")
    results = engine.run(phir, shots=100)

    # R1XY(π, 0) + R1XY(π, 0) = rotation by 2π = identity (up to global phase)
    # Qubit should remain in |0⟩
    zeros = results["a"].count("0")
    assert (
        zeros == 100
    ), f"Expected all measurements to be 0 (2π rotation = identity), but got {100-zeros} ones"


def test_conditional_r1xy_with_false_condition() -> None:
    """Test that conditional R1XY is not executed when condition is false."""
    phir = """{
    "format": "PHIR/JSON",
    "version": "0.1.0",
    "metadata": {"source": "test"},
    "ops": [
        {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 1},
        {"data": "cvar_define", "data_type": "i64", "variable": "a", "size": 1},
        {"data": "cvar_define", "data_type": "i64", "variable": "t", "size": 1},
        {
            "block": "if",
            "condition": {"cop": "==", "args": [["t", 0], 1]},
            "true_branch": [
                {"qop": "R1XY", "angles": [[1.0, 0.0], "pi"], "args": [["q", 0]]}
            ]
        },
        {"qop": "Measure", "returns": [["a", 0]], "args": [["q", 0]]}
    ]
}"""

    engine = HybridEngine(qsim="state-vector")
    results = engine.run(phir, shots=100)

    # Condition is false (t[0]=0, not 1), so R1XY should not execute
    # Qubit should remain in |0⟩
    zeros = results["a"].count("0")
    assert (
        zeros == 100
    ), f"Expected all zeros (R1XY not executed), but got {100-zeros} ones"


def test_r1xy_alternative_angles_summing_to_4pi() -> None:
    """Test another set of angles summing to 4π (1.9 rad and 4π-1.9 rad).

    This was reported as working correctly in issue #81 comments.
    """
    four_pi_minus_1_9 = 4 * pc.f64.pi - 1.9

    phir = f"""{{
    "format": "PHIR/JSON",
    "version": "0.1.0",
    "metadata": {{"source": "test"}},
    "ops": [
        {{"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 1}},
        {{"data": "cvar_define", "data_type": "i64", "variable": "a", "size": 1}},
        {{"data": "cvar_define", "data_type": "i64", "variable": "t", "size": 1}},
        {{
            "block": "if",
            "condition": {{"cop": "==", "args": [["t", 0], 0]}},
            "true_branch": [
                {{"qop": "R1XY", "angles": [[1.9, 0.0], "rad"], "args": [["q", 0]]}}
            ]
        }},
        {{"qop": "R1XY", "angles": [[{four_pi_minus_1_9}, 0.0], "rad"], "args": [["q", 0]]}},
        {{"qop": "Measure", "returns": [["a", 0]], "args": [["q", 0]]}}
    ]
}}"""

    engine = HybridEngine(qsim="state-vector")
    results = engine.run(phir, shots=100)

    zeros = results["a"].count("0")
    ones = results["a"].count("1")

    assert (
        ones == 0
    ), f"Expected all measurements to be 0 (4π cancellation), but got {ones} ones"
    assert zeros == 100, f"Expected 100 zeros, but got {zeros}"
