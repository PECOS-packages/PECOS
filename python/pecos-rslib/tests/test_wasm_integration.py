"""Test WASM integration with QASM simulation."""

import pytest
import os
import tempfile
from pecos_rslib.qasm_sim import qasm_sim


def create_add_wat():
    """Create a simple WAT file that adds two numbers."""
    wat_content = """
    (module
      (func $init (export "init"))
      (func $add (export "add") (param i32 i32) (result i32)
        local.get 0
        local.get 1
        i32.add
      )
    )
    """
    return wat_content


def test_qasm_wasm_basic():
    """Test basic WASM function call from QASM."""
    qasm = """
    OPENQASM 2.0;
    creg a[10];
    creg b[10];
    creg result[10];

    a = 5;
    b = 3;
    result = add(a, b);
    """

    # Create a temporary WAT file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".wat", delete=False) as f:
        f.write(create_add_wat())
        wat_path = f.name

    try:
        # Run the simulation with WASM
        results = qasm_sim(qasm).wasm(wat_path).run(10)

        # Check that all shots give the expected result
        for i in range(10):
            assert results["a"][i] == 5
            assert results["b"][i] == 3
            assert results["result"][i] == 8  # 5 + 3 = 8
    finally:
        # Clean up
        os.unlink(wat_path)


def test_qasm_wasm_with_quantum():
    """Test WASM integration with quantum operations."""
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    creg sum[10];

    h q[0];
    cx q[0], q[1];
    measure q -> c;

    sum = add(c[0], c[1]);
    """

    # Create a temporary WAT file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".wat", delete=False) as f:
        f.write(create_add_wat())
        wat_path = f.name

    try:
        # Run the simulation with WASM
        results = qasm_sim(qasm).seed(42).wasm(wat_path).run(1000)

        # Check quantum entanglement and WASM addition
        for i in range(1000):
            c_val = results["c"][i]
            sum_val = results["sum"][i]

            # Due to entanglement, c should be either 0 (00) or 3 (11)
            assert c_val in [0, 3]

            # sum should be 0 (0+0) or 2 (1+1)
            if c_val == 0:
                assert sum_val == 0
            else:  # c_val == 3
                assert sum_val == 2
    finally:
        # Clean up
        os.unlink(wat_path)


def test_qasm_wasm_void_function():
    """Test calling void WASM functions from QASM."""
    qasm = """
    OPENQASM 2.0;
    creg a[10];
    creg b[10];

    a = 5;
    b = 10;
    void_func(a, b);  // Call void function
    """

    wat_content = """
    (module
      (func $init (export "init"))
      (func $void_func (export "void_func") (param i32 i32)
        ;; Void function - does nothing but is valid
      )
    )
    """

    # Create a temporary WAT file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".wat", delete=False) as f:
        f.write(wat_content)
        wat_path = f.name

    try:
        # Run the simulation with WASM
        results = qasm_sim(qasm).wasm(wat_path).run(1)

        # Check that the values are unchanged
        assert results["a"][0] == 5
        assert results["b"][0] == 10
    finally:
        # Clean up
        os.unlink(wat_path)


def test_qasm_wasm_missing_init():
    """Test that WASM modules without init function are rejected."""
    qasm = """
    OPENQASM 2.0;
    creg a[10];
    a = 5;
    """

    wat_content = """
    (module
      (func $add (export "add") (param i32 i32) (result i32)
        local.get 0
        local.get 1
        i32.add
      )
    )
    """

    # Create a temporary WAT file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".wat", delete=False) as f:
        f.write(wat_content)
        wat_path = f.name

    try:
        # This should raise an error
        with pytest.raises(RuntimeError, match="init"):
            qasm_sim(qasm).wasm(wat_path).build()
    finally:
        # Clean up
        os.unlink(wat_path)


def test_qasm_wasm_missing_function():
    """Test that calling non-existent WASM functions raises an error."""
    qasm = """
    OPENQASM 2.0;
    creg a[10];
    creg b[10];
    creg result[10];

    a = 5;
    b = 3;
    result = multiply(a, b);  // This function doesn't exist
    """

    # Create a temporary WAT file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".wat", delete=False) as f:
        f.write(create_add_wat())
        wat_path = f.name

    try:
        # This should raise an error during build
        with pytest.raises(RuntimeError, match="multiply"):
            qasm_sim(qasm).wasm(wat_path).build()
    finally:
        # Clean up
        os.unlink(wat_path)


if __name__ == "__main__":
    test_qasm_wasm_basic()
    test_qasm_wasm_with_quantum()
    test_qasm_wasm_void_function()
    test_qasm_wasm_missing_init()
    test_qasm_wasm_missing_function()
    print("All tests passed!")
