"""Test QisInterfaceBuilder pattern - Helios and Selene Helios interfaces."""

import pytest
from pecos_rslib import (
    qis_engine,
    qis_helios_interface,
    qis_selene_helios_interface,
    Qis,
)


class TestQisInterfaceBuilder:
    """Test the QisInterfaceBuilder pattern."""

    def test_builder_functions_exist(self):
        """Test that the interface builder functions exist."""
        assert callable(qis_helios_interface)
        assert callable(qis_selene_helios_interface)

    def test_helios_interface_creation(self):
        """Test that Helios interface can be created."""
        interface = qis_helios_interface()
        assert interface is not None

    def test_selene_helios_interface_creation(self):
        """Test that Selene Helios interface can be created."""
        interface = qis_selene_helios_interface()
        assert interface is not None

    def test_bell_state_with_helios(self):
        """Test Bell state simulation with Helios interface."""
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

        qis_program = Qis.from_string(bell_qis)
        interface_builder = qis_helios_interface()

        # Run simulation
        engine = qis_engine().interface(interface_builder).program(qis_program)
        sim = engine.to_sim().qubits(2).seed(42)
        results = sim.run(100)

        # Verify Bell state results
        count_00, count_11 = _count_bell_results(results)

        # Verify distribution is reasonable (allowing for statistical variation)
        assert 20 < count_00 < 80, f"00 count out of expected range: {count_00}"
        assert 20 < count_11 < 80, f"11 count out of expected range: {count_11}"
        assert (
            count_00 + count_11 == 100
        ), f"Total should be 100, got {count_00 + count_11}"

    def test_bell_state_with_selene_helios(self):
        """Test Bell state simulation with Selene Helios interface."""
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

        qis_program = Qis.from_string(bell_qis)
        interface_builder = qis_selene_helios_interface()

        # Run simulation
        engine = qis_engine().interface(interface_builder).program(qis_program)
        sim = engine.to_sim().qubits(2).seed(42)
        results = sim.run(100)

        # Verify Bell state results
        count_00, count_11 = _count_bell_results(results)

        # Verify distribution is reasonable (allowing for statistical variation)
        assert 20 < count_00 < 80, f"00 count out of expected range: {count_00}"
        assert 20 < count_11 < 80, f"11 count out of expected range: {count_11}"
        assert (
            count_00 + count_11 == 100
        ), f"Total should be 100, got {count_00 + count_11}"

    def test_ghz_state_with_helios(self):
        """Test 3-qubit GHZ state with Helios interface."""
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

        qis_program = Qis.from_string(ghz_qis)
        interface_builder = qis_helios_interface()

        # Run simulation
        engine = qis_engine().interface(interface_builder).program(qis_program)
        sim = engine.to_sim().qubits(3).seed(42)
        results = sim.run(100)

        # Verify GHZ state results
        count_000, count_111 = _count_ghz_results(results)

        # Verify we got valid measurements
        assert (
            count_000 + count_111 == 100
        ), f"Total should be 100, got {count_000 + count_111}"
        assert count_000 > 0 or count_111 > 0, "Should have some valid measurements"

    def test_ghz_state_with_selene_helios(self):
        """Test 3-qubit GHZ state with Selene Helios interface."""
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

        qis_program = Qis.from_string(ghz_qis)
        interface_builder = qis_selene_helios_interface()

        # Run simulation
        engine = qis_engine().interface(interface_builder).program(qis_program)
        sim = engine.to_sim().qubits(3).seed(42)
        results = sim.run(100)

        # Verify GHZ state results
        count_000, count_111 = _count_ghz_results(results)

        # Verify we got valid measurements
        assert (
            count_000 + count_111 == 100
        ), f"Total should be 100, got {count_000 + count_111}"
        assert count_000 > 0 or count_111 > 0, "Should have some valid measurements"

    def test_missing_interface_gives_helpful_error(self):
        """Test that missing interface gives a helpful error message."""
        simple_qis = """
            define void @main() {
                call void @__quantum__qis__h__body(i64 0)
                ret void
            }
            declare void @__quantum__qis__h__body(i64)
        """
        qis_program = Qis.from_string(simple_qis)

        # No .interface() call - should give helpful error, not silent fallback
        with pytest.raises(RuntimeError) as exc_info:
            qis_engine().program(qis_program)

        error_msg = str(exc_info.value)
        # Error message should guide the user on how to fix it
        assert "interface" in error_msg.lower()
        assert "runtime" in error_msg.lower() or "helios" in error_msg.lower()

    def test_explicit_helios_selection(self):
        """Test explicit Helios interface selection works."""
        simple_qis = """
            define void @main() {
                call void @__quantum__qis__h__body(i64 0)
                ret void
            }
            declare void @__quantum__qis__h__body(i64)
        """
        qis_program = Qis.from_string(simple_qis)

        # Explicitly select Helios
        engine = qis_engine().interface(qis_helios_interface()).program(qis_program)
        sim = engine.to_sim().qubits(1)
        results = sim.run(1)

        assert results is not None

    def test_explicit_selene_helios_selection(self):
        """Test explicit Selene Helios interface selection works."""
        simple_qis = """
            define void @main() {
                call void @__quantum__qis__h__body(i64 0)
                ret void
            }
            declare void @__quantum__qis__h__body(i64)
        """
        qis_program = Qis.from_string(simple_qis)

        # Explicitly select Selene Helios
        engine = (
            qis_engine().interface(qis_selene_helios_interface()).program(qis_program)
        )
        sim = engine.to_sim().qubits(1)
        results = sim.run(1)

        assert results is not None


def _count_bell_results(results):
    """Count Bell state measurement outcomes."""
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

    return count_00, count_11


def _count_ghz_results(results):
    """Count GHZ state measurement outcomes."""
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

    return count_000, count_111


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
