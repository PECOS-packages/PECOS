from pypecos.engines.hybrid_engine import HybridEngine


def test_setting_bits1():
    phir = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "cvar_define", "data_type": "u32", "variable": "c", "size": 3},
            # c[0], c[1], c[2] = True, False, True
            {
                "cop": "=",
                "returns": [["c", 0], ["c", 1], ["c", 2]],
                "args": [True, False, True],
            },
        ],
    }

    results = HybridEngine(qsim="stabilizer").run(program=phir, shots=5)

    assert results["c"].count("101") == len(results["c"])


def test_setting_bits2():
    phir = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "cvar_define", "data_type": "u32", "variable": "c", "size": 3},
            # c[0], c[1], c[2] = 0, 1, 1
            {"cop": "=", "returns": [["c", 0], ["c", 1], ["c", 2]], "args": [0, 1, 1]},
        ],
    }

    results = HybridEngine(qsim="stabilizer").run(program=phir, shots=5)

    assert results["c"].count("110") == len(results["c"])


def test_setting_cvar():
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

    assert results["a"].count("000") == len(results["a"])
    assert results["b"].count("001") == len(results["b"])
    assert results["c"].count("010") == len(results["c"])


def test_setting_expr():
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

    assert results["a"].count("001") == len(results["a"])
    assert results["b"].count("001") == len(results["b"])
    assert results["c"].count("010") == len(results["c"])


def test_setting_mixed():
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

    assert results["a"].count("001") == len(results["a"])
    assert results["b"].count("011") == len(results["b"])
    assert results["c"].count("010") == len(results["c"])
    assert results["d"].count("100") == len(results["d"])
