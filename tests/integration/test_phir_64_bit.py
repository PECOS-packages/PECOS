from pecos.engines.hybrid_engine import HybridEngine


def bin2int(result: list[str]) -> int:
    return int(result[0], base=2)


def test_setting_cvar():
    phir = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "cvar_define", "data_type": "i32", "variable": "vi32", "size": 32},
            {"data": "cvar_define", "data_type": "u32", "variable": "vu32", "size": 32},
            {"data": "cvar_define", "data_type": "i64", "variable": "vi64", "size": 64},
            {"data": "cvar_define", "data_type": "u64", "variable": "vu64", "size": 64},
            {"data": "cvar_define", "data_type": "i32", "variable": "vi32neg", "size": 32},
            {"data": "cvar_define", "data_type": "i64", "variable": "vi64neg", "size": 64},
            {"cop": "=", "returns": ["vi32"], "args": [2**31 - 1]},
            {"cop": "=", "returns": ["vu32"], "args": [2**32 - 1]},
            {"cop": "=", "returns": ["vi64"], "args": [2**63 - 1]},
            {"cop": "=", "returns": ["vu64"], "args": [2**64 - 1]},
            {"cop": "=", "returns": ["vi32neg"], "args": [-(2**31)]},
            {"cop": "=", "returns": ["vi64neg"], "args": [-(2**63)]},
        ],
    }

    results = HybridEngine(qsim="stabilizer").run(program=phir, shots=5)

    assert bin2int(results["vi32"]) == 2**31 - 1
    assert bin2int(results["vu32"]) == 2**32 - 1
    assert bin2int(results["vi64"]) == 2**63 - 1
    assert bin2int(results["vu64"]) == 2**64 - 1
    assert bin2int(results["vi32neg"]) == -(2**31)
    assert bin2int(results["vi64neg"]) == -(2**63)
