"""Test WASM integration with QASM simulation using the correct API."""

import os
import tempfile

from pecos_rslib import qasm_engine
from pecos_rslib.programs import Qasm
from pecos_rslib import sim


def test_qasm_wasm_basic_classical() -> None:
    """Test basic WASM function call from QASM for classical computation."""
    # Create a simple WAT module with add function
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

    # Compile WAT to WASM
    # Save WAT file - Rust will compile it automatically
    with tempfile.NamedTemporaryFile(suffix=".wat", delete=False, mode="w") as f:
        f.write(wat_content)
        wasm_path = f.name

    try:
        # QASM that uses the WASM functions
        qasm = """
        OPENQASM 2.0;
        creg a[10];
        creg b[10];
        creg result[10];

        a = 5;
        b = 7;
        result = add(a, b);
        """

        prog = Qasm.from_string(qasm)

        # Create engine with WASM loaded, then set the program
        engine = qasm_engine().wasm(wasm_path).program(prog)

        # Use sim() with the configured engine
        results = sim(prog).classical(engine).run(10).to_dict()

        # Check that we got the expected result
        assert "a" in results
        assert "b" in results
        assert "result" in results

        # All shots should have result = 12 (5 + 7)
        for i in range(len(results["result"])):
            assert results["a"][i] == 5
            assert results["b"][i] == 7
            assert results["result"][i] == 12

    finally:
        # Clean up
        if os.path.exists(wasm_path):
            os.remove(wasm_path)


def test_qasm_wasm_with_quantum() -> None:
    """Test WASM function controlling quantum operations."""
    wat_content = """
    (module
      (func $init (export "init"))
      (func $add (export "add") (param i32 i32) (result i32)
        local.get 0
        local.get 1
        i32.add
      )
      (func $should_flip (export "should_flip") (param i32) (result i32)
        ;; Return 1 if input > 5, else 0
        local.get 0
        i32.const 5
        i32.gt_s
      )
    )
    """

    # Save WAT file - Rust will compile it automatically
    with tempfile.NamedTemporaryFile(suffix=".wat", delete=False, mode="w") as f:
        f.write(wat_content)
        wasm_path = f.name

    try:
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[2];
        creg c[2];
        creg check[1];

        // Check if we should flip first qubit
        check = should_flip(7);  // 7 > 5, returns 1
        if (check == 1) x q[0];

        // Check if we should flip second qubit
        check = should_flip(3);  // 3 <= 5, returns 0
        if (check == 1) x q[1];

        measure q -> c;
        """

        prog = Qasm.from_string(qasm)

        # Create engine with WASM support
        engine = qasm_engine().program(prog).wasm(wasm_path)

        # Run simulation
        results = sim(prog).classical(engine).run(10).to_dict()

        # First qubit should be 1, second should be 0
        # So c should be 1 (binary 01)
        assert all(val == 1 for val in results["c"])

    finally:
        if os.path.exists(wasm_path):
            os.remove(wasm_path)


def test_wasm_fibonacci() -> None:
    """Test WASM with Fibonacci calculation."""
    wat_content = """
    (module
      (func $init (export "init"))

      ;; Iterative Fibonacci
      (func $fib (export "fib") (param i32) (result i32)
        (local $a i32)
        (local $b i32)
        (local $temp i32)
        (local $i i32)

        ;; Handle base cases
        local.get 0
        i32.const 2
        i32.lt_s
        if
          local.get 0
          return
        end

        ;; Initialize
        i32.const 0
        local.set $a
        i32.const 1
        local.set $b
        i32.const 2
        local.set $i

        ;; Loop
        loop
          ;; temp = a + b
          local.get $a
          local.get $b
          i32.add
          local.set $temp

          ;; a = b
          local.get $b
          local.set $a

          ;; b = temp
          local.get $temp
          local.set $b

          ;; i++
          local.get $i
          i32.const 1
          i32.add
          local.set $i

          ;; Continue if i <= n
          local.get $i
          local.get 0
          i32.le_s
          br_if 0
        end

        local.get $b
      )
    )
    """

    # Save WAT file - Rust will compile it automatically
    with tempfile.NamedTemporaryFile(suffix=".wat", delete=False, mode="w") as f:
        f.write(wat_content)
        wasm_path = f.name

    try:
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[2];
        creg c[2];
        creg fib_result[10];

        // Calculate fib(7) = 13
        fib_result = fib(7);

        // Set qubits based on result
        if (fib_result == 13) x q[0];

        // Calculate fib(10) = 55
        fib_result = fib(10);
        if (fib_result == 55) x q[1];

        measure q -> c;
        """

        prog = Qasm.from_string(qasm)

        # Create engine with WASM
        engine = qasm_engine().program(prog).wasm(wasm_path)

        results = sim(prog).classical(engine).run(10).to_dict()

        # Both conditions are true, so both qubits should be 1
        assert all(val == 3 for val in results["c"])  # 0b11 = 3

    finally:
        if os.path.exists(wasm_path):
            os.remove(wasm_path)


def test_wasm_with_multiple_functions() -> None:
    """Test WASM module with multiple functions of different signatures."""
    wat_content = """
    (module
      (func $init (export "init"))

      ;; No parameters, returns constant
      (func $get_constant (export "get_constant") (result i32)
        i32.const 42
      )

      ;; Single parameter
      (func $double (export "double") (param i32) (result i32)
        local.get 0
        i32.const 2
        i32.mul
      )

      ;; Three parameters
      (func $sum3 (export "sum3") (param i32 i32 i32) (result i32)
        local.get 0
        local.get 1
        i32.add
        local.get 2
        i32.add
      )
    )
    """

    # Save WAT file - Rust will compile it automatically
    with tempfile.NamedTemporaryFile(suffix=".wat", delete=False, mode="w") as f:
        f.write(wat_content)
        wasm_path = f.name

    try:
        qasm = """
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[3];
        creg c[3];
        creg temp[10];

        // Test get_constant (no params)
        temp = get_constant();
        if (temp == 42) x q[0];

        // Test double (1 param)
        temp = double(21);
        if (temp == 42) x q[1];

        // Test sum3 (3 params)
        temp = sum3(10, 20, 12);
        if (temp == 42) x q[2];

        measure q -> c;
        """

        prog = Qasm.from_string(qasm)
        engine = qasm_engine().program(prog).wasm(wasm_path)

        results = sim(prog).classical(engine).run(10).to_dict()

        # All conditions should be true
        assert all(val == 7 for val in results["c"])  # 0b111 = 7

    finally:
        if os.path.exists(wasm_path):
            os.remove(wasm_path)
