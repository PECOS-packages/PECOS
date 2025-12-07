"""Test the PHIR JSON unified API Python bindings."""

from pecos import PhirJson, phir_json_engine


def test_phir_json_program_creation() -> None:
    """Test creating PhirJson from string and JSON."""
    json_str = """{
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "metadata": {},
        "ops": [
            {"data": "cvar_define", "data_type": "u32", "variable": "result", "size": 1},
            {"cop": "Result", "args": [0], "returns": [["result", 0]]}
        ]
    }"""

    # Test from_string
    program1 = PhirJson.from_string(json_str)

    # Test from_json (should be the same)
    program2 = PhirJson.from_json(json_str)

    # Both should work
    assert program1 is not None
    assert program2 is not None


def test_phir_json_engine_builder() -> None:
    """Test creating a PHIR JSON engine builder."""
    json_str = """{
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "metadata": {},
        "ops": [
            {"data": "cvar_define", "data_type": "u32", "variable": "result", "size": 1},
            {"cop": "=", "returns": [["result", 0]], "args": [1]},
            {"cop": "Result", "args": [["result", 0]], "returns": [["result", 0]]}
        ]
    }"""

    # Use Python PhirJson type with pecos phir_json_engine
    program = PhirJson.from_json(json_str)

    # Create engine builder
    builder = phir_json_engine().program(program)

    # Convert to simulation builder
    sim_builder = builder.to_sim()

    # Set some options
    sim_builder = sim_builder.seed(42).workers(1)

    # Run simulation
    result = sim_builder.run(10)

    # Check we got a ShotVec
    assert hasattr(result, "to_dict")
    result_dict = result.to_dict()

    # Should have 'result' key
    assert "result" in result_dict
    assert len(result_dict["result"]) == 10


def test_phir_json_unified_api_full() -> None:
    """Test the full unified API pattern."""
    json_str = """{
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "metadata": {},
        "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
            {"data": "cvar_define", "data_type": "u32", "variable": "m", "size": 2},
            {"qop": "H", "args": [["q", 0]]},
            {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
            {"qop": "Measure", "args": [["q", 1]], "returns": [["m", 1]]},
            {"cop": "Result", "args": ["m"], "returns": ["m"]}
        ]
    }"""

    # One-liner unified API using pecos PhirJson
    result = (
        phir_json_engine()
        .program(PhirJson.from_json(json_str))
        .to_sim()
        .seed(42)
        .run(100)
    )

    # Check result
    result_dict = result.to_dict()
    assert "m" in result_dict
    assert len(result_dict["m"]) == 100

    # All measurements should be integers
    for val in result_dict["m"]:
        assert isinstance(val, int)
        assert 0 <= val <= 3  # 2 bits, so values 0-3
