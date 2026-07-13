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

"""Integration tests for PHIR 64-bit value handling."""

from pecos.engines.hybrid_engine import HybridEngine


def twos_complement_bits(value: int, width: int) -> str:
    """Expected two's-complement bit string of ``value`` at ``width`` bits."""
    return format(value & ((1 << width) - 1), f"0{width}b")


def test_setting_cvar() -> None:
    """Test setting classical variables in PHIR with 64-bit values."""
    phir = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "cvar_define", "data_type": "i32", "variable": "var_i32"},
            {
                "data": "cvar_define",
                "data_type": "u32",
                "variable": "var_u32",
                "size": 32,
            },
            {"data": "cvar_define", "data_type": "i64", "variable": "var_i64"},
            {
                "data": "cvar_define",
                "data_type": "u64",
                "variable": "var_u64",
                "size": 64,
            },
            {"data": "cvar_define", "data_type": "i32", "variable": "var_i32neg"},
            {"data": "cvar_define", "data_type": "i64", "variable": "var_i64neg"},
            {"cop": "=", "returns": ["var_i32"], "args": [2**31 - 1]},
            {"cop": "=", "returns": ["var_u32"], "args": [2**32 - 1]},
            {"cop": "=", "returns": ["var_i64"], "args": [2**63 - 1]},
            {"cop": "=", "returns": ["var_u64"], "args": [2**64 - 1]},
            {"cop": "=", "returns": ["var_i32neg"], "args": [-(2**31)]},
            {"cop": "=", "returns": ["var_i64neg"], "args": [-(2**63)]},
        ],
    }

    results = HybridEngine(qsim="stabilizer").run(program=phir, shots=5)
    results_dict = results

    # Registers render as fixed-width two's-complement bit strings. Negative
    # signed values show the sign bit as "1" -- never a "-" prefix.
    expected = {
        "var_i32": twos_complement_bits(2**31 - 1, 32),
        "var_u32": twos_complement_bits(2**32 - 1, 32),
        "var_i64": twos_complement_bits(2**63 - 1, 64),
        "var_u64": twos_complement_bits(2**64 - 1, 64),
        "var_i32neg": twos_complement_bits(-(2**31), 32),
        "var_i64neg": twos_complement_bits(-(2**63), 64),
    }
    for name, bits in expected.items():
        assert all(shot == bits for shot in results_dict[name]), name
