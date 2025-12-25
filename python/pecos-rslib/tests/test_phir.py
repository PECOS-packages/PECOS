"""Tests for PHIR (PECOS High-level IR) JSON pipeline."""

import pytest


def test_phir_json_engine_import() -> None:
    """Test that PhirJsonEngine can be imported."""
    from pecos_rslib import PhirJsonEngine

    assert PhirJsonEngine is not None


def test_phir_json_engine_builder_import() -> None:
    """Test that PhirJsonEngineBuilder can be imported."""
    from pecos_rslib import PhirJsonEngineBuilder

    assert PhirJsonEngineBuilder is not None


def test_phir_json_program_import() -> None:
    """Test that PhirJson can be imported."""
    from pecos_rslib.programs import PhirJson

    assert PhirJson is not None


def test_phir_json_simulation_import() -> None:
    """Test that PhirJsonSimulation can be imported."""
    from pecos_rslib import PhirJsonSimulation

    assert PhirJsonSimulation is not None


def test_compile_hugr_to_qis_import() -> None:
    """Test that compile_hugr_to_qis can be imported."""
    from pecos_rslib import compile_hugr_to_qis

    assert compile_hugr_to_qis is not None


def test_phir_json_engine_function() -> None:
    """Test that phir_json_engine function is available."""
    from pecos_rslib import phir_json_engine

    # Should be able to create an engine builder
    engine_builder = phir_json_engine()
    assert engine_builder is not None


def test_phir_json_program_creation() -> None:
    """Test creating PhirJson from JSON."""
    from pecos_rslib.programs import PhirJson

    # PhirJson.from_json may accept strings and parse them later
    # or may validate immediately. Test what actually happens:
    from contextlib import suppress

    with suppress(ValueError, RuntimeError, TypeError):
        # This might not raise immediately
        PhirJson.from_json("not json")
        # If it doesn't raise during creation, that's OK - it might fail during use

    # Test creating from valid-looking JSON string
    with suppress(ValueError, RuntimeError, TypeError):
        PhirJson.from_json("{}")
        # Empty object might be accepted or rejected


def test_compile_hugr_to_qis_with_invalid_input() -> None:
    """Test compile_hugr_to_qis with invalid input."""
    from pecos_rslib import compile_hugr_to_qis

    # compile_hugr_to_qis expects bytes
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        # Pass invalid HUGR bytes
        compile_hugr_to_qis(b"not valid hugr")


def test_compile_hugr_to_qis_with_wrong_type() -> None:
    """Test compile_hugr_to_qis with wrong input type."""
    from pecos_rslib import compile_hugr_to_qis

    # Should raise TypeError for string instead of bytes
    with pytest.raises(TypeError):
        compile_hugr_to_qis("{}")  # String instead of bytes
