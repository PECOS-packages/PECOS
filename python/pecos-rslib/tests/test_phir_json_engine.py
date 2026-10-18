# PECOS/python/pecos-rslib/tests/test_phir_json_engine.py
"""Tests for PHIR-JSON engine integration with Rust-based simulators.

This module contains test cases for verifying the integration between PHIR (PECOS High-level Intermediate
Representation) and the Rust-based quantum simulators, ensuring proper execution of quantum programs and
correct simulation results.
"""

import json

import pytest
from pecos_rslib import PhirJsonEngine


# Helper function to create a PhirJsonEngine instance with a simple test program
def create_test_bell_program() -> str:
    """Create a simple PHIR-JSON program for testing register mapping.

    This function returns a PHIR JSON program that creates a Bell state,
    measures two qubits, and maps the results to both 'm' and 'output' registers.
    """
    return json.dumps(
        {
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "metadata": {"description": "Bell state with register mapping"},
            "ops": [
                {
                    "data": "qvar_define",
                    "data_type": "qubits",
                    "variable": "q",
                    "size": 2,
                },
                {"data": "cvar_define", "data_type": "i64", "variable": "m", "size": 2},
                {
                    "data": "cvar_define",
                    "data_type": "i64",
                    "variable": "output",
                    "size": 2,
                },
                {"qop": "H", "args": [["q", 0]]},
                {"qop": "CX", "args": [["q", 0], ["q", 1]]},
                {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
                {"qop": "Measure", "args": [["q", 1]], "returns": [["m", 1]]},
            ],
        },
    )


def test_phir_minimal() -> None:
    """Test with a minimal PHIR-JSON program to verify basic functionality."""
    phir_json = json.dumps(
        {
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "metadata": {"generated_by": "PECOS version 0.6.0.dev8"},
            "ops": [
                {
                    "data": "qvar_define",
                    "data_type": "qubits",
                    "variable": "q",
                    "size": 2,
                },
                {
                    "data": "cvar_define",
                    "data_type": "i64",
                    "variable": "m",
                    "size": 2,
                },
                {
                    "qop": "Measure",
                    "args": [["q", 0]],
                    "returns": [["m", 0]],
                },
            ],
        },
    )

    # Create engine
    engine = PhirJsonEngine(phir_json)

    # Get commands
    commands = engine.process_program()
    assert len(commands) == 1, "Expected 1 quantum command"

    # Verify it's a measure gate
    cmd = commands[0]
    assert cmd["gate_type"] == "Measure", "Expected Measure gate"
    assert cmd["params"]["result_id"] == 0, "Expected result_id to be 0"
    assert cmd["qubits"][0] == 0, "Expected measurement on qubit 0"

    # Handle measurement
    engine.handle_measurement(1)  # Send a measurement result of 1

    # Get results
    results = engine.get_results()

    # Extract the measurement key
    assert len(results) > 0, "Expected at least one measurement result"
    measurement_key = "m"

    # Verify the result
    assert results[measurement_key] == 1, f"Expected {measurement_key} to preserve outcome 1"


def test_phir_invalid_json() -> None:
    """Test PHIR-JSON engine handling of invalid JSON input."""
    invalid_json = '{"format": "PHIR/JSON", "invalid": }'
    with pytest.raises(
        ValueError,
        match="(?i)parse|json",
    ):
        PhirJsonEngine(invalid_json)


def test_phir_empty_program() -> None:
    """Test PHIR-JSON engine processing of empty program."""
    phir = json.dumps(
        {
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "metadata": {"generated_by": "Test"},
            "ops": [],
        },
    )

    engine = PhirJsonEngine(phir)
    commands = engine.process_program()
    assert len(commands) == 0, "Expected empty command list"


def test_phir_full_circuit() -> None:
    """Test PHIR-JSON engine processing of complete quantum circuit."""
    phir = json.dumps(
        {
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "metadata": {"generated_by": "PECOS version 0.6.0.dev8"},
            "ops": [
                {
                    "data": "qvar_define",
                    "data_type": "qubits",
                    "variable": "q",
                    "size": 2,
                },
                {"data": "cvar_define", "data_type": "i64", "variable": "c", "size": 2},
                {"qop": "Measure", "args": [["q", 0]], "returns": [["c", 0]]},
                {"qop": "Measure", "args": [["q", 1]], "returns": [["c", 1]]},
            ],
        },
    )

    # Create engine
    engine = PhirJsonEngine(phir)

    # Generate a batch before delivering outcomes in command order.
    engine.process_program()
    engine.handle_measurement(1)
    engine.handle_measurement(0)

    # Get final results
    results = engine.get_results()

    assert len(results) > 0, "Expected measurement results"


def test_phir_full() -> None:
    """Test with a full PHIR-JSON program."""
    phir = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "metadata": {"generated_by": "PECOS version 0.6.0.dev8"},
        "ops": [
            {
                "data": "qvar_define",
                "data_type": "qubits",
                "variable": "q",
                "size": 2,
            },
            {
                "data": "cvar_define",
                "data_type": "i64",
                "variable": "m",
                "size": 2,
            },
            {
                "qop": "Measure",
                "args": [["q", 0]],
                "returns": [["m", 0]],
            },
        ],
    }

    phir_json = json.dumps(phir)
    engine = PhirJsonEngine(phir_json)
    results = engine.results_dict
    assert isinstance(results, dict)


def test_register_mapping_simulation() -> None:
    """A native batch accepts scalar outcomes in order without rewriting them."""
    phir = json.dumps(
        {
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "metadata": {"description": "Simple measurement test"},
            "ops": [
                {
                    "data": "qvar_define",
                    "data_type": "qubits",
                    "variable": "q",
                    "size": 2,
                },
                {"data": "cvar_define", "data_type": "i64", "variable": "m", "size": 2},
                {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
                {"qop": "Measure", "args": [["q", 1]], "returns": [["m", 1]]},
            ],
        },
    )

    # Create engine with validation disabled
    engine = PhirJsonEngine.create_with_validation_disabled(phir)

    # Process the program to get quantum commands
    commands = engine.process_program()

    assert [command["qubits"] for command in commands] == [[0], [1]]
    assert all(command["gate_type"] == "Measure" for command in commands)
    engine.handle_measurement(1)
    with pytest.raises(ValueError, match="1 of 2"):
        engine.process_program()
    engine.handle_measurement(0)
    assert engine.process_program() == []
    assert engine.get_results()["m"] == 1


def test_broadcast_measurement_destinations() -> None:
    """Scalar delivery preserves every target and reversed classical destination."""
    source = {
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 4},
            {"data": "cvar_define", "data_type": "u32", "variable": "m", "size": 4},
            {"qop": "X", "args": [["q", 0], ["q", 2]]},
            {"qop": "CX", "args": [[["q", 0], ["q", 1]], [["q", 2], ["q", 3]]]},
            {
                "qop": "Measure",
                "args": [["q", index] for index in range(4)],
                "returns": [["m", 3 - index] for index in range(4)],
            },
            {"cop": "Result", "args": ["m"], "returns": ["out"]},
        ],
    }
    engine = PhirJsonEngine(json.dumps(source))
    commands = engine.process_program()
    cx_qubits = [qubit for command in commands if command["gate_type"] == "CX" for qubit in command["qubits"]]
    assert cx_qubits == [0, 1, 2, 3]
    for outcome in [1, 0, 1, 1]:
        engine.handle_measurement(outcome)
    assert engine.process_program() == []
    assert engine.get_results()["out"] == 11
    assert engine.results_dict == engine.get_results()
