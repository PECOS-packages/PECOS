"""Tests for PHIR JSON pipeline."""

import pytest

from contextlib import suppress


def test_phir_json_engine_function() -> None:
    """Test that phir_json_engine returns a builder."""
    from pecos_rslib import phir_json_engine

    engine_builder = phir_json_engine()
    assert engine_builder is not None


def test_phir_json_program_creation() -> None:
    """Test creating PhirJson from JSON."""
    from pecos_rslib.programs import PhirJson

    with suppress(ValueError, RuntimeError, TypeError):
        PhirJson.from_json("not json")

    with suppress(ValueError, RuntimeError, TypeError):
        PhirJson.from_json("{}")


def test_compile_hugr_to_qis_with_invalid_input() -> None:
    """Test compile_hugr_to_qis rejects invalid HUGR bytes."""
    from pecos_rslib import compile_hugr_to_qis

    with pytest.raises((RuntimeError, ValueError, TypeError)):
        compile_hugr_to_qis(b"not valid hugr")


def test_compile_hugr_to_qis_with_wrong_type() -> None:
    """Test compile_hugr_to_qis rejects string input."""
    from pecos_rslib import compile_hugr_to_qis

    with pytest.raises(TypeError):
        compile_hugr_to_qis("{}")
