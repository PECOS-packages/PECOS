"""Tests for PHIR JSON pipeline."""

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
