"""Test PHIR JSON + Wasmtime integration using Rust backend.

This test file demonstrates the new capability to use the Rust PhirJsonEngine
with Wasmtime for foreign function calls, mirroring the pattern used for QASM.
"""

import json
import os
import tempfile


from pecos_rslib import phir_json_engine
from pecos_rslib.programs import PhirJson
from pecos_rslib import sim


def test_phir_wasm_basic_ffcall() -> None:
    """Test basic WASM foreign function call from PHIR JSON."""
    # Create a simple WAT module with add and subtract functions
    wat_content = """
    (module
      (func $init (export "init"))
      (func $add (export "add") (param i32 i32) (result i32)
        local.get 0
        local.get 1
        i32.add
      )
      (func $sub (export "sub") (param i32 i32) (result i32)
        local.get 0
        local.get 1
        i32.sub
      )
    )
    """

    # Save WAT file - Rust Wasmtime will compile it automatically
    with tempfile.NamedTemporaryFile(suffix=".wat", delete=False, mode="w") as f:
        f.write(wat_content)
        wasm_path = f.name

    try:
        # Create PHIR JSON program with foreign function calls
        phir_json = {
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "metadata": {
                "num_qubits": 0,
                "source_program_type": ["Test", ["PECOS", "0.7.0"]],
            },
            "ops": [
                # Define classical variables
                {
                    "data": "cvar_define",
                    "data_type": "i32",
                    "variable": "a",
                    "size": 32,
                },
                {
                    "data": "cvar_define",
                    "data_type": "i32",
                    "variable": "b",
                    "size": 32,
                },
                {
                    "data": "cvar_define",
                    "data_type": "i32",
                    "variable": "result",
                    "size": 32,
                },
                # Set a = 10
                {"cop": "=", "args": [10], "returns": ["a"]},
                # Set b = 7
                {"cop": "=", "args": [7], "returns": ["b"]},
                # result = add(a, b)  -- should be 17
                {
                    "cop": "ffcall",
                    "function": "add",
                    "args": ["a", "b"],
                    "returns": ["result"],
                },
                # Export result
                {"cop": "Result", "args": ["result"], "returns": ["output"]},
            ],
        }

        # Create PHIR program
        prog = PhirJson.from_json(json.dumps(phir_json))

        # Create engine with WASM support using the same pattern as QASM
        engine = phir_json_engine().wasm(wasm_path).program(prog)

        # Run simulation
        results = sim(prog).classical(engine).run(10).to_dict()

        # Check results
        assert "output" in results
        assert all(val == 17 for val in results["output"])

    finally:
        if os.path.exists(wasm_path):
            os.remove(wasm_path)


def test_phir_wasm_conditional_ffcall() -> None:
    """Test conditional foreign function calls in PHIR JSON."""
    wat_content = """
    (module
      (func $init (export "init"))
      (func $double (export "double") (param i32) (result i32)
        local.get 0
        i32.const 2
        i32.mul
      )
      (func $triple (export "triple") (param i32) (result i32)
        local.get 0
        i32.const 3
        i32.mul
      )
    )
    """

    with tempfile.NamedTemporaryFile(suffix=".wat", delete=False, mode="w") as f:
        f.write(wat_content)
        wasm_path = f.name

    try:
        # PHIR program with conditional ffcall
        phir_json = {
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "metadata": {
                "num_qubits": 0,
                "source_program_type": ["Test", ["PECOS", "0.7.0"]],
            },
            "ops": [
                {
                    "data": "cvar_define",
                    "data_type": "i32",
                    "variable": "x",
                    "size": 32,
                },
                {
                    "data": "cvar_define",
                    "data_type": "i32",
                    "variable": "result",
                    "size": 32,
                },
                {
                    "data": "cvar_define",
                    "data_type": "i32",
                    "variable": "condition",
                    "size": 32,
                },
                # Set x = 5
                {"cop": "=", "args": [5], "returns": ["x"]},
                # Set condition = 1
                {"cop": "=", "args": [1], "returns": ["condition"]},
                # if (condition == 1) result = double(x)
                {
                    "block": "if",
                    "condition": {"cop": "==", "args": ["condition", 1]},
                    "true_branch": [
                        {
                            "cop": "ffcall",
                            "function": "double",
                            "args": ["x"],
                            "returns": ["result"],
                        }
                    ],
                    "false_branch": [
                        {
                            "cop": "ffcall",
                            "function": "triple",
                            "args": ["x"],
                            "returns": ["result"],
                        }
                    ],
                },
                {"cop": "Result", "args": ["result"], "returns": ["output"]},
            ],
        }

        prog = PhirJson.from_json(json.dumps(phir_json))
        engine = phir_json_engine().wasm(wasm_path).program(prog)
        results = sim(prog).classical(engine).run(10).to_dict()

        # Should execute double(5) = 10
        assert all(val == 10 for val in results["output"])

    finally:
        if os.path.exists(wasm_path):
            os.remove(wasm_path)


def test_phir_wasm_with_quantum_ops() -> None:
    """Test PHIR JSON with both quantum operations and WASM foreign function calls."""
    wat_content = """
    (module
      (func $init (export "init"))
      (func $is_zero (export "is_zero") (param i32) (result i32)
        ;; Return 1 if input is 0, else return 0
        local.get 0
        i32.const 0
        i32.eq
      )
    )
    """

    with tempfile.NamedTemporaryFile(suffix=".wat", delete=False, mode="w") as f:
        f.write(wat_content)
        wasm_path = f.name

    try:
        # PHIR program with quantum ops and ffcall
        phir_json = {
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "metadata": {
                "num_qubits": 1,
                "source_program_type": ["Test", ["PECOS", "0.7.0"]],
            },
            "ops": [
                # Define qubit and classical variables
                {
                    "data": "qvar_define",
                    "data_type": "qubits",
                    "variable": "q",
                    "size": 1,
                },
                {"data": "cvar_define", "data_type": "i32", "variable": "m", "size": 1},
                {
                    "data": "cvar_define",
                    "data_type": "i32",
                    "variable": "check",
                    "size": 32,
                },
                # Measure qubit (initially |0>)
                {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
                # check = is_zero(m)  -- should be 1 since m=0
                {
                    "cop": "ffcall",
                    "function": "is_zero",
                    "args": ["m"],
                    "returns": ["check"],
                },
                # Export check
                {"cop": "Result", "args": ["check"], "returns": ["output"]},
            ],
        }

        prog = PhirJson.from_json(json.dumps(phir_json))
        engine = phir_json_engine().wasm(wasm_path).program(prog)

        # Need to specify quantum engine for quantum operations
        from pecos_rslib import state_vector

        results = sim(prog).classical(engine).quantum(state_vector()).run(10).to_dict()

        # check should be 1 (is_zero(0) = 1)
        assert all(val == 1 for val in results["output"])

    finally:
        if os.path.exists(wasm_path):
            os.remove(wasm_path)


if __name__ == "__main__":
    # Run tests
    test_phir_wasm_basic_ffcall()
    print("test_phir_wasm_basic_ffcall passed")

    test_phir_wasm_conditional_ffcall()
    print("test_phir_wasm_conditional_ffcall passed")

    test_phir_wasm_with_quantum_ops()
    print("test_phir_wasm_with_quantum_ops passed")

    print("\nAll PHIR + Wasmtime tests passed!")
