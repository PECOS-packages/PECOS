"""Example of using WebAssembly functions with QASM simulation.

This example demonstrates how to call WebAssembly functions from QASM code,
enabling custom classical computations within quantum circuits.
"""

import os
import tempfile

from pecos_rslib import qasm_engine, sim
from pecos_rslib.programs import Qasm


def create_math_wat() -> str:
    """Create a WAT file with various mathematical functions."""
    return """
    (module
      ;; Required init function
      (func $init (export "init"))

      ;; Add two numbers
      (func $add (export "add") (param i32 i32) (result i32)
        local.get 0
        local.get 1
        i32.add
      )

      ;; Multiply two numbers
      (func $multiply (export "multiply") (param i32 i32) (result i32)
        local.get 0
        local.get 1
        i32.mul
      )

      ;; Square a number
      (func $square (export "square") (param i32) (result i32)
        local.get 0
        local.get 0
        i32.mul
      )

      ;; Void function for side effects (in real use, might update memory)
      (func $process (export "process") (param i32 i32))
    )
    """


def example_basic_wasm() -> None:
    """Basic example of calling WASM functions from QASM."""
    print("=== Basic WASM Function Calls ===")

    qasm = """
    OPENQASM 2.0;
    creg a[10];
    creg b[10];
    creg sum[10];
    creg product[10];
    creg a_squared[10];

    // Initialize values
    a = 7;
    b = 3;

    // Call WASM functions
    sum = add(a, b);
    product = multiply(a, b);
    a_squared = square(a);

    // Call void function
    process(a, b);
    """

    # Create temporary WAT file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".wat", delete=False) as f:
        f.write(create_math_wat())
        wat_path = f.name

    try:
        # Run simulation with WASM
        engine_builder = qasm_engine().program(Qasm.from_string(qasm)).wasm(wat_path)
        results = engine_builder.to_sim().run(5)

        # Display results
        for shot in range(5):
            print(f"\nShot {shot}:")
            print(f"  a = {results['a'][shot]}")
            print(f"  b = {results['b'][shot]}")
            print(f"  sum = {results['sum'][shot]}")
            print(f"  product = {results['product'][shot]}")
            print(f"  a_squared = {results['a_squared'][shot]}")

    finally:
        os.unlink(wat_path)


def example_quantum_with_wasm() -> None:
    """Example combining quantum operations with WASM computations."""
    print("\n=== Quantum Circuit with WASM Processing ===")

    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";

    qreg q[3];
    creg c[3];
    creg parity[1];
    creg weighted_sum[10];

    // Create superposition
    h q[0];
    h q[1];
    h q[2];

    // Measure
    measure q -> c;

    // Process measurement results with WASM
    // Calculate weighted sum: c[0]*4 + c[1]*2 + c[2]*1
    // Note: We need to do this step by step as nested function calls aren't supported
    creg temp1[10];
    creg temp2[10];
    creg temp3[10];
    creg temp4[10];

    temp1 = multiply(c[0], 4);  // c[0] * 4
    temp2 = multiply(c[1], 2);  // c[1] * 2
    temp3 = add(temp1, temp2);   // (c[0]*4) + (c[1]*2)
    weighted_sum = add(temp3, c[2]); // + c[2]
    """

    # Create temporary WAT file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".wat", delete=False) as f:
        f.write(create_math_wat())
        wat_path = f.name

    try:
        # Run simulation with WASM
        results = sim(qasm).seed(42).wasm(wat_path).run(20)

        # Count occurrences of each weighted sum
        weighted_counts = {}
        for shot in range(20):
            c_val = results["c"][shot]
            weighted = results["weighted_sum"][shot]

            # Verify the calculation
            expected = (
                ((c_val >> 0) & 1) * 4 + ((c_val >> 1) & 1) * 2 + ((c_val >> 2) & 1) * 1
            )
            assert (
                weighted == expected
            ), f"Mismatch: got {weighted}, expected {expected}"

            weighted_counts[weighted] = weighted_counts.get(weighted, 0) + 1

        print("\nWeighted sum distribution:")
        for value in sorted(weighted_counts.keys()):
            count = weighted_counts[value]
            binary = f"{value:03b}"
            print(f"  {value} (binary: {binary}): {count} times")

    finally:
        os.unlink(wat_path)


def example_error_handling() -> None:
    """Example showing error handling for WASM integration."""
    print("\n=== Error Handling Examples ===")

    # Example 1: Missing function
    qasm_missing_func = """
    OPENQASM 2.0;
    creg a[10];
    a = divide(10, 2);  // This function doesn't exist
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".wat", delete=False) as f:
        f.write(create_math_wat())
        wat_path = f.name

    try:
        print("\n1. Trying to call non-existent function 'divide'...")
        try:
            sim(qasm_missing_func).wasm(wat_path).build()
        except RuntimeError as e:
            print(f"   Expected error: {e}")

    finally:
        os.unlink(wat_path)

    # Example 2: Missing init function
    wat_no_init = """
    (module
      (func $add (export "add") (param i32 i32) (result i32)
        local.get 0
        local.get 1
        i32.add
      )
    )
    """

    qasm_simple = """
    OPENQASM 2.0;
    creg a[10];
    a = 5;
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".wat", delete=False) as f:
        f.write(wat_no_init)
        wat_path = f.name

    try:
        print("\n2. Trying to use WASM module without init function...")
        try:
            sim(qasm_simple).wasm(wat_path).build()
        except RuntimeError as e:
            print(f"   Expected error: {e}")

    finally:
        os.unlink(wat_path)


if __name__ == "__main__":
    example_basic_wasm()
    example_quantum_with_wasm()
    example_error_handling()
    print("\nAll examples completed successfully!")
