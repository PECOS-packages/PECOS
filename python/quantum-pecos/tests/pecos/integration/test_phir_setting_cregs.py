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


def test_negative_signed_register_is_twos_complement() -> None:
    """Negative signed registers print two's-complement bits, not a "-" sign.

    A full-width signed register (size == the backing type width) can hold a
    negative value. Its bit string must show the sign bit as "1"/"0" -- never
    Python's sign-and-magnitude "-..." -- and stays at the backing width.
    """
    phir = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "cvar_define", "data_type": "i32", "variable": "w", "size": 32},
            {"data": "cvar_define", "data_type": "i64", "variable": "d", "size": 64},
            # w = d = 0 - 1  ->  -1  ->  all ones in two's complement
            {
                "cop": "=",
                "returns": ["w", "d"],
                "args": [
                    {"cop": "-", "args": [0, 1]},
                    {"cop": "-", "args": [0, 1]},
                ],
            },
        ],
    }

    results = HybridEngine(qsim="stabilizer").run(program=phir, shots=3)

    for bits in results["w"]:
        assert bits == "1" * 32
    for bits in results["d"]:
        assert bits == "1" * 64
    assert not any("-" in bits for bits in results["w"] + results["d"])


def test_signed_register_prints_sign_bit_width() -> None:
    """A signed size-n register prints n+1 bits: n data bits plus a sign bit.

    A size-31 i32-backed register is non-negative (its data bits are masked to
    31 bits), so it prints 32 bits with a leading "0" sign bit. An unsigned
    register has no sign bit and stays exactly `size` wide.
    """
    phir = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "cvar_define", "data_type": "i32", "variable": "s", "size": 31},
            {"data": "cvar_define", "data_type": "u32", "variable": "u", "size": 31},
            {"cop": "=", "returns": ["s", "u"], "args": [5, 5]},
        ],
    }

    results = HybridEngine(qsim="stabilizer").run(program=phir, shots=3)

    for bits in results["s"]:
        assert bits == "0" + "0" * 26 + "00101"  # 32 bits: sign "0" + 31 data bits
        assert len(bits) == 32
    for bits in results["u"]:
        assert bits == "0" * 26 + "00101"  # 31 bits, no sign bit
        assert len(bits) == 31
