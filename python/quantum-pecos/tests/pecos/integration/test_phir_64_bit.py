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


def bin2int(result: list[str]) -> int:
    """Convert binary string to integer."""
    return int(result[0], base=2)


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

    assert bin2int(results_dict["var_i32"]) == 2**31 - 1
    assert bin2int(results_dict["var_u32"]) == 2**32 - 1
    assert bin2int(results_dict["var_i64"]) == 2**63 - 1
    assert bin2int(results_dict["var_u64"]) == 2**64 - 1
    assert bin2int(results_dict["var_i32neg"]) == -(2**31)
    assert bin2int(results_dict["var_i64neg"]) == -(2**63)
