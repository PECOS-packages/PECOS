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

"""Integration tests for PHIR classical register setting."""

from pecos.engines.hybrid_engine import HybridEngine


def test_setting_bits() -> None:
    """Test setting individual bits in classical registers."""
    phir = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "cvar_define", "data_type": "u32", "variable": "c", "size": 3},
            # c[0], c[1], c[2] = 1, 0, 1
            {"cop": "=", "returns": [["c", 0], ["c", 1], ["c", 2]], "args": [1, 0, 1]},
        ],
    }

    results = HybridEngine(qsim="stabilizer").run(program=phir, shots=5)
    results_dict = results

    assert results_dict["c"].count("101") == len(results_dict["c"])


def test_setting_cvar() -> None:
    """Test setting classical variables in PHIR."""
    phir = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "cvar_define", "data_type": "u32", "variable": "a", "size": 3},
            {"data": "cvar_define", "data_type": "u32", "variable": "b", "size": 3},
            {"data": "cvar_define", "data_type": "u32", "variable": "c", "size": 3},
            # a, b, c = 0, 1, 2
            {"cop": "=", "returns": ["a", "b", "c"], "args": [0, 1, 2]},
        ],
    }

    results = HybridEngine(qsim="stabilizer").run(program=phir, shots=5)
    results_dict = results

    assert results_dict["a"].count("000") == len(results_dict["a"])
    assert results_dict["b"].count("001") == len(results_dict["b"])
    assert results_dict["c"].count("010") == len(results_dict["c"])


def test_setting_expr() -> None:
    """Test setting expressions in classical registers."""
    phir = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "cvar_define", "data_type": "u32", "variable": "a", "size": 3},
            {"data": "cvar_define", "data_type": "u32", "variable": "b", "size": 3},
            {"data": "cvar_define", "data_type": "u32", "variable": "c", "size": 3},
            # a, b, c = 0+1, a+1, c[1]+2
            {
                "cop": "=",
                "returns": ["a", "b", "c"],
                "args": [
                    {"cop": "+", "args": [0, 1]},
                    {"cop": "+", "args": ["a", 1]},
                    {"cop": "+", "args": [["c", 1], 2]},
                ],
            },
        ],
    }

    results = HybridEngine(qsim="stabilizer").run(program=phir, shots=5)
    results_dict = results

    assert results_dict["a"].count("001") == len(results_dict["a"])
    assert results_dict["b"].count("001") == len(results_dict["b"])
    assert results_dict["c"].count("010") == len(results_dict["c"])


def test_setting_mixed() -> None:
    """Test setting mixed types in classical registers."""
    phir = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "cvar_define", "data_type": "u32", "variable": "a", "size": 3},
            {"data": "cvar_define", "data_type": "u32", "variable": "b", "size": 3},
            {"data": "cvar_define", "data_type": "u32", "variable": "c", "size": 3},
            {"data": "cvar_define", "data_type": "u32", "variable": "d", "size": 3},
            # a[0], b, c, d[2] = 1, 2, c[1]+2, a[0] + 1
            {
                "cop": "=",
                "returns": [
                    ["a", 0],
                    "b",
                    "c",
                    ["d", 2],
                ],
                "args": [
                    1,
                    3,
                    {"cop": "+", "args": [["c", 1], 2]},
                    {"cop": "+", "args": [["a", 0], 1]},
                ],
            },
        ],
    }

    results = HybridEngine(qsim="stabilizer").run(program=phir, shots=5)
    results_dict = results

    assert results_dict["a"].count("001") == len(results_dict["a"])
    assert results_dict["b"].count("011") == len(results_dict["b"])
    assert results_dict["c"].count("010") == len(results_dict["c"])
    assert results_dict["d"].count("100") == len(results_dict["d"])
