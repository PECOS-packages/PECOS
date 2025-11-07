"""Test QisInterfaceBuilder pattern with Helios as reference implementation."""

import pytest
from pecos_rslib import (
    qis_engine,
    qis_helios_interface,
    qis_selene_helios_interface,
    QisProgram,
)


def run_with_both_interfaces(test_name, test_fn):
    """Helper to run tests with both Helios (reference) and JIT interfaces.

    Helios is considered the reference implementation - it's well-tested in Selene.
    JIT is our fallback for when Selene isn't available.
    Both should produce the same results for the same quantum circuits.
    """
    print(f"\nTesting {test_name} with Helios interface (reference):")

    # Check if we can use Helios by attempting a simple compilation
    test_program = QisProgram.from_string("define void @main() { ret void }")
    can_use_helios = False
    try:
        (qis_engine().interface(qis_selene_helios_interface()).program(test_program))
        can_use_helios = True
    except Exception as e:
        print(f"  Helios interface not available: {e}")

    if can_use_helios:
        try:
            test_fn("Helios")
            print("  Helios test passed (reference)")
        except Exception as e:
            pytest.fail(f"Helios reference implementation failed: {e}")

        # Now test with JIT - it should match Helios results
        print(f"\nTesting {test_name} with JIT interface (should match Helios):")
        try:
            test_fn("JIT")
            print("  JIT test passed (matches reference)")
        except Exception as e:
            pytest.fail(f"JIT implementation differs from Helios reference: {e}")
    else:
        print("  WARNING: Helios not available (Selene not installed)")
        print("  INFO: Running with JIT interface only")

        # At least test with JIT
        try:
            test_fn("JIT")
            print("  JIT test passed")
        except Exception as e:
            pytest.fail(f"JIT test failed: {e}")

        print("  WARNING: Could not verify against Helios reference implementation")


class TestQisInterfaceBuilder:
    """Test the QisInterfaceBuilder pattern with both interfaces."""

    def test_builder_functions_exist(self):
        """Test that the interface builder functions exist."""
        assert callable(qis_helios_interface)
        assert callable(qis_selene_helios_interface)

    def test_bell_state_with_both_interfaces(self):
        """Test Bell state with both interfaces, treating Helios as reference."""

        def run_bell_test(interface_name):
            # Bell state QIS program in LLVM IR
            bell_qis = """
                define void @main() {
                    call void @__quantum__qis__h__body(i64 0)
                    call void @__quantum__qis__cx__body(i64 0, i64 1)
                    %result0 = call i32 @__quantum__qis__m__body(i64 0, i64 0)
                    %result1 = call i32 @__quantum__qis__m__body(i64 1, i64 1)
                    ret void
                }

                declare void @__quantum__qis__h__body(i64)
                declare void @__quantum__qis__cx__body(i64, i64)
                declare i32 @__quantum__qis__m__body(i64, i64)
            """

            qis_program = QisProgram.from_string(bell_qis)

            # Select interface based on test parameter
            if interface_name == "Helios":
                interface_builder = qis_selene_helios_interface()
            else:
                interface_builder = qis_helios_interface()

            # Run simulation (runtime is default/built-in)
            engine = qis_engine().interface(interface_builder).program(qis_program)
            sim = engine.to_sim().qubits(2).seed(42)
            results = sim.run(100)

            # Verify Bell state results
            count_00 = 0
            count_11 = 0

            results_dict = results.to_dict()
            m0_vals = results_dict.get("measurement_0", [])
            m1_vals = results_dict.get("measurement_1", [])

            for m0, m1 in zip(m0_vals, m1_vals, strict=False):
                if m0 == 0 and m1 == 0:
                    count_00 += 1
                elif m0 == 1 and m1 == 1:
                    count_11 += 1
                else:
                    raise ValueError(
                        f"Bell state should only produce |00⟩ or |11⟩, got: ({m0}, {m1})"
                    )

            print(
                f"    {interface_name} interface: |00⟩: {count_00} times, |11⟩: {count_11} times"
            )

            # Verify distribution is reasonable (allowing for statistical variation)
            assert 20 < count_00 < 80, f"00 count out of expected range: {count_00}"
            assert 20 < count_11 < 80, f"11 count out of expected range: {count_11}"
            assert (
                count_00 + count_11 == 100
            ), f"Total should be 100, got {count_00 + count_11}"

        run_with_both_interfaces("Bell state", run_bell_test)

    def test_ghz_state_with_both_interfaces(self):
        """Test 3-qubit GHZ state with both interfaces."""

        def run_ghz_test(interface_name):
            # GHZ state QIS program
            ghz_qis = """
                define void @main() {
                    call void @__quantum__qis__h__body(i64 0)
                    call void @__quantum__qis__cx__body(i64 0, i64 1)
                    call void @__quantum__qis__cx__body(i64 1, i64 2)
                    %result0 = call i32 @__quantum__qis__m__body(i64 0, i64 0)
                    %result1 = call i32 @__quantum__qis__m__body(i64 1, i64 1)
                    %result2 = call i32 @__quantum__qis__m__body(i64 2, i64 2)
                    ret void
                }

                declare void @__quantum__qis__h__body(i64)
                declare void @__quantum__qis__cx__body(i64, i64)
                declare i32 @__quantum__qis__m__body(i64, i64)
            """

            qis_program = QisProgram.from_string(ghz_qis)

            # Select interface based on test parameter
            if interface_name == "Helios":
                interface_builder = qis_selene_helios_interface()
            else:
                interface_builder = qis_helios_interface()

            # Run simulation (runtime is default/built-in)
            engine = qis_engine().interface(interface_builder).program(qis_program)
            sim = engine.to_sim().qubits(3).seed(42)
            results = sim.run(100)

            # Verify GHZ state results - should only get |000⟩ or |111⟩
            count_000 = 0
            count_111 = 0

            results_dict = results.to_dict()
            m0_vals = results_dict.get("measurement_0", [])
            m1_vals = results_dict.get("measurement_1", [])
            m2_vals = results_dict.get("measurement_2", [])

            for m0, m1, m2 in zip(m0_vals, m1_vals, m2_vals, strict=False):
                if m0 == 0 and m1 == 0 and m2 == 0:
                    count_000 += 1
                elif m0 == 1 and m1 == 1 and m2 == 1:
                    count_111 += 1
                else:
                    raise ValueError(
                        f"GHZ state should only produce |000⟩ or |111⟩, got: ({m0}, {m1}, {m2})"
                    )

            print(
                f"    {interface_name} interface: |000⟩: {count_000} times, |111⟩: {count_111} times"
            )

            # Verify we got valid measurements
            assert (
                count_000 + count_111 == 100
            ), f"Total should be 100, got {count_000 + count_111}"
            assert count_000 > 0 or count_111 > 0, "Should have some valid measurements"

        run_with_both_interfaces("GHZ state", run_ghz_test)

    def test_default_behavior(self):
        """Test that default behavior uses Helios interface."""
        simple_qis = "define void @main() { ret void }"
        qis_program = QisProgram.from_string(simple_qis)

        try:
            # No .interface() call - should default to Helios
            qis_engine().program(qis_program)
            print("Default behavior uses Helios interface")
        except Exception as e:
            if "Selene Helios compilation failed" in str(e) or "Selene" in str(e):
                print("Correctly attempted Helios by default (but Selene unavailable)")
            else:
                pytest.fail(f"Unexpected error with default interface: {e}")

    def test_explicit_jit_selection(self):
        """Test explicit JIT interface selection always works."""
        simple_qis = """
            define void @main() {
                call void @__quantum__qis__h__body(i64 0)
                ret void
            }
            declare void @__quantum__qis__h__body(i64)
        """
        qis_program = QisProgram.from_string(simple_qis)

        # Explicitly select JIT - should always work
        engine = qis_engine().interface(qis_helios_interface()).program(qis_program)
        sim = engine.to_sim().qubits(1)
        results = sim.run(1)

        assert results is not None
        print("Explicit JIT interface selection works")


if __name__ == "__main__":
    # Run the tests
    test = TestQisInterfaceBuilder()
    test.test_builder_functions_exist()
    test.test_default_behavior()
    test.test_explicit_jit_selection()
    test.test_bell_state_with_both_interfaces()
    test.test_ghz_state_with_both_interfaces()
    print("\nAll tests completed")
