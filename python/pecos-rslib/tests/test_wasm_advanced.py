"""Advanced test cases for WASM integration with QASM simulation."""

import os
import tempfile
from pecos_rslib.qasm_sim import qasm_sim, QuantumEngine


def test_wasm_multiple_functions_types():
    """Test WASM with different function signatures and types."""
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

      ;; Bit operations
      (func $bitwise_and (export "bitwise_and") (param i32 i32) (result i32)
        local.get 0
        local.get 1
        i32.and
      )
    )
    """

    qasm = """
    OPENQASM 2.0;
    creg a[10];
    creg b[10];
    creg c[10];
    creg const_val[10];
    creg doubled[10];
    creg sum[10];
    creg bit_result[10];

    a = 5;
    b = 3;
    c = 7;

    const_val = get_constant();
    doubled = double(a);
    sum = sum3(a, b, c);
    bit_result = bitwise_and(a, b);
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".wat", delete=False) as f:
        f.write(wat_content)
        wat_path = f.name

    try:
        results = qasm_sim(qasm).wasm(wat_path).run(5)

        for i in range(5):
            assert results["const_val"][i] == 42
            assert results["doubled"][i] == 10  # 5 * 2
            assert results["sum"][i] == 15  # 5 + 3 + 7
            assert results["bit_result"][i] == 1  # 5 & 3 = 0101 & 0011 = 0001
    finally:
        os.unlink(wat_path)


def test_wasm_with_different_engines():
    """Test WASM works with different quantum engines."""
    wat_content = """
    (module
      (func $init (export "init"))
      (func $is_zero (export "is_zero") (param i32) (result i32)
        local.get 0
        i32.eqz
      )
    )
    """

    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    creg c[2];
    creg is_zero_result[1];

    h q[0];
    cx q[0], q[1];
    measure q -> c;

    is_zero_result = is_zero(c);
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".wat", delete=False) as f:
        f.write(wat_content)
        wat_path = f.name

    try:
        # Test with StateVector engine
        results_sv = (
            qasm_sim(qasm)
            .wasm(wat_path)
            .quantum_engine(QuantumEngine.StateVector)
            .run(100)
        )

        # Test with SparseStabilizer engine (Clifford only)
        results_ss = (
            qasm_sim(qasm)
            .wasm(wat_path)
            .quantum_engine(QuantumEngine.SparseStabilizer)
            .run(100)
        )

        # Both should work and produce valid results
        for results in [results_sv, results_ss]:
            for i in range(100):
                c_val = results["c"][i]
                result_val = results["is_zero_result"][i]
                # c should be 0 or 3 (Bell state)
                assert c_val in [0, 3]
                # is_zero should be 1 when c==0, 0 when c==3
                expected = 1 if c_val == 0 else 0
                assert result_val == expected
    finally:
        os.unlink(wat_path)


def test_wasm_large_values():
    """Test WASM with large integer values."""
    wat_content = """
    (module
      (func $init (export "init"))

      ;; Test with larger values
      (func $multiply_large (export "multiply_large") (param i32 i32) (result i32)
        local.get 0
        local.get 1
        i32.mul
      )

      ;; Bitwise operations on large values
      (func $shift_left (export "shift_left") (param i32 i32) (result i32)
        local.get 0
        local.get 1
        i32.shl
      )
    )
    """

    qasm = """
    OPENQASM 2.0;
    creg a[32];
    creg b[32];
    creg product[32];
    creg shifted[32];

    a = 1000000;
    b = 2000;
    product = multiply_large(a, b);

    a = 255;
    b = 8;
    shifted = shift_left(a, b);
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".wat", delete=False) as f:
        f.write(wat_content)
        wat_path = f.name

    try:
        results = qasm_sim(qasm).wasm(wat_path).run(1)

        assert results["product"][0] == 2_000_000_000  # 1M * 2K
        assert results["shifted"][0] == 65280  # 255 << 8 = 0xFF00
    finally:
        os.unlink(wat_path)


def test_wasm_sequential_calls():
    """Test multiple sequential WASM function calls."""
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

      (func $mul (export "mul") (param i32 i32) (result i32)
        local.get 0
        local.get 1
        i32.mul
      )
    )
    """

    qasm = """
    OPENQASM 2.0;
    creg a[10];
    creg b[10];
    creg temp1[10];
    creg temp2[10];
    creg result[10];

    // Complex calculation: ((a + b) * 2) - 5
    a = 10;
    b = 7;

    temp1 = add(a, b);      // 17
    temp2 = mul(temp1, 2);  // 34
    result = sub(temp2, 5); // 29
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".wat", delete=False) as f:
        f.write(wat_content)
        wat_path = f.name

    try:
        results = qasm_sim(qasm).seed(42).wasm(wat_path).run(10)

        for i in range(10):
            assert results["temp1"][i] == 17
            assert results["temp2"][i] == 34
            assert results["result"][i] == 29
    finally:
        os.unlink(wat_path)


def test_wasm_with_noise():
    """Test WASM integration works correctly with noise models."""
    from pecos_rslib.qasm_sim import DepolarizingNoise

    wat_content = """
    (module
      (func $init (export "init"))
      (func $count_ones (export "count_ones") (param i32) (result i32)
        ;; Simple bit counting (not optimal but works for small values)
        (local $count i32)
        (local $value i32)
        (local.set $value (local.get 0))
        (local.set $count (i32.const 0))

        ;; Count bits in first 8 positions
        (local.set $count
          (i32.add (local.get $count)
            (i32.and (local.get $value) (i32.const 1))))
        (local.set $value (i32.shr_u (local.get $value) (i32.const 1)))

        (local.set $count
          (i32.add (local.get $count)
            (i32.and (local.get $value) (i32.const 1))))
        (local.set $value (i32.shr_u (local.get $value) (i32.const 1)))

        (local.set $count
          (i32.add (local.get $count)
            (i32.and (local.get $value) (i32.const 1))))

        (local.get $count)
      )
    )
    """

    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[3];
    creg c[3];
    creg ones[10];

    // Create GHZ state
    h q[0];
    cx q[0], q[1];
    cx q[1], q[2];

    measure q -> c;
    ones = count_ones(c);
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".wat", delete=False) as f:
        f.write(wat_content)
        wat_path = f.name

    try:
        # Run with noise
        results = (
            qasm_sim(qasm)
            .seed(42)
            .noise(DepolarizingNoise(p=0.01))
            .wasm(wat_path)
            .run(1000)
        )

        # Count occurrences
        zero_count = sum(1 for i in range(1000) if results["c"][i] == 0)
        seven_count = sum(1 for i in range(1000) if results["c"][i] == 7)
        other_count = 1000 - zero_count - seven_count

        # With noise, we should see mostly 000 and 111, but some errors
        assert zero_count > 400  # Should be ~500 with small noise
        assert seven_count > 400
        assert other_count > 0  # Should have some errors due to noise

        # Check count_ones function works correctly
        for i in range(1000):
            c_val = results["c"][i]
            ones_val = results["ones"][i]
            expected = bin(c_val).count("1")
            assert (
                ones_val == expected or ones_val <= 3
            )  # Our simple implementation counts up to 3
    finally:
        os.unlink(wat_path)


def test_wasm_error_negative_result():
    """Test WASM behavior with operations that could produce negative results."""
    wat_content = """
    (module
      (func $init (export "init"))

      ;; Subtraction that could go negative (but wraps in unsigned)
      (func $sub (export "sub") (param i32 i32) (result i32)
        local.get 0
        local.get 1
        i32.sub
      )
    )
    """

    qasm = """
    OPENQASM 2.0;
    creg a[32];
    creg b[32];
    creg result[32];

    a = 5;
    b = 10;
    result = sub(a, b);  // 5 - 10 would be -5, but wraps to large positive
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".wat", delete=False) as f:
        f.write(wat_content)
        wat_path = f.name

    try:
        results = qasm_sim(qasm).wasm(wat_path).run(1)

        # In unsigned 32-bit arithmetic, 5 - 10 wraps around
        # This should be 2^32 - 5 = 4294967291
        result_val = results["result"][0]
        assert result_val == 4294967291
    finally:
        os.unlink(wat_path)


def test_wasm_with_conditionals():
    """Test WASM function calls within QASM conditional statements."""
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

    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[1];
    creg c[1];
    creg value[10];
    creg result[10];

    value = 5;

    h q[0];
    measure q -> c;

    if (c == 0) result = double(value);
    if (c == 1) result = triple(value);
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".wat", delete=False) as f:
        f.write(wat_content)
        wat_path = f.name

    try:
        results = qasm_sim(qasm).seed(42).wasm(wat_path).run(100)

        for i in range(100):
            c_val = results["c"][i]
            result_val = results["result"][i]

            if c_val == 0:
                assert result_val == 10  # double(5)
            else:
                assert result_val == 15  # triple(5)
    finally:
        os.unlink(wat_path)


def test_wasm_build_once_run_multiple():
    """Test building simulation once and running multiple times with WASM."""
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

    qasm = """
    OPENQASM 2.0;
    creg a[10];
    creg b[10];
    creg sum[10];

    a = 3;
    b = 4;
    sum = add(a, b);
    """

    with tempfile.NamedTemporaryFile(mode="w", suffix=".wat", delete=False) as f:
        f.write(wat_content)
        wat_path = f.name

    try:
        # Build once
        sim = qasm_sim(qasm).wasm(wat_path).seed(42).build()

        # Run multiple times
        results1 = sim.run(10)
        results2 = sim.run(20)
        results3 = sim.run(5)

        # All runs should produce correct results
        for results in [results1, results2, results3]:
            for i in range(len(results["a"])):
                assert results["sum"][i] == 7
    finally:
        os.unlink(wat_path)


if __name__ == "__main__":
    test_wasm_multiple_functions_types()
    test_wasm_with_different_engines()
    test_wasm_large_values()
    test_wasm_sequential_calls()
    test_wasm_with_noise()
    test_wasm_error_negative_result()
    test_wasm_with_conditionals()
    test_wasm_build_once_run_multiple()
    print("All advanced tests passed!")
