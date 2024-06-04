from pecos.engines.hybrid_engine import HybridEngine


def test_setting_cvar():
    phir = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "cvar_define", "data_type": "i32", "variable": "var_i32"},
            {"data": "cvar_define", "data_type": "u32", "variable": "var_u32", "size": 32},
            {"data": "cvar_define", "data_type": "i64", "variable": "var_i64"},
            {"data": "cvar_define", "data_type": "u64", "variable": "var_u64", "size": 64},
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

    results = HybridEngine(qsim="stabilizer").run(program=phir, return_int=True)

    assert results["var_i32"] == [2**31 - 1]
    assert results["var_u32"] == [2**32 - 1]
    assert results["var_i64"] == [2**63 - 1]
    assert results["var_u64"] == [2**64 - 1]
    assert results["var_i32neg"] == [-(2**31)]
    assert results["var_i64neg"] == [-(2**63)]
