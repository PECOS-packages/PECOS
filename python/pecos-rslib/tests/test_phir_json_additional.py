"""Additional PHIR-JSON tests that work with current constraints."""

import json


def test_phir_json_measurement_only() -> None:
    """Test PHIR-JSON with only measurements (no Result instruction needed)."""
    from pecos_rslib import PhirJsonEngine

    # Create a minimal PHIR-JSON program without Result instruction
    # This should work with current validation
    phir_json = json.dumps(
        {
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "metadata": {"description": "Minimal measurement test"},
            "ops": [
                {
                    "data": "qvar_define",
                    "data_type": "qubits",
                    "variable": "q",
                    "size": 1,
                },
                {
                    "data": "cvar_define",
                    "data_type": "i64",
                    "variable": "m",
                    "size": 1,
                },
                {
                    "qop": "Measure",
                    "args": [["q", 0]],
                    "returns": [["m", 0]],
                },
            ],
        },
    )

    engine = PhirJsonEngine(phir_json)
    commands = engine.process_program()
    assert len(commands) == 1
    assert commands[0]["gate_type"] == "Measure"


def test_phir_json_validation_requirements() -> None:
    """Variable declarations emit no gates; measurement emits one command."""
    from pecos_rslib import PhirJsonEngine

    # Test various PHIR-JSON structures to understand what's required
    test_cases = [
        # Case 1: Absolutely minimal
        {
            "name": "empty_ops",
            "phir": {"format": "PHIR/JSON", "version": "0.1.0", "ops": []},
            "expected_commands": [],
        },
        # Case 2: Just variable definitions
        {
            "name": "just_vars",
            "expected_commands": [],
            "phir": {
                "format": "PHIR/JSON",
                "version": "0.1.0",
                "ops": [
                    {
                        "data": "qvar_define",
                        "data_type": "qubits",
                        "variable": "q",
                        "size": 1,
                    },
                    {
                        "data": "cvar_define",
                        "data_type": "i64",
                        "variable": "m",
                        "size": 1,
                    },
                ],
            },
        },
        # Case 3: With measurement
        {
            "name": "with_measurement",
            "expected_commands": [{"gate_type": "Measure", "params": {"result_id": 0}, "qubits": [0]}],
            "phir": {
                "format": "PHIR/JSON",
                "version": "0.1.0",
                "ops": [
                    {
                        "data": "qvar_define",
                        "data_type": "qubits",
                        "variable": "q",
                        "size": 1,
                    },
                    {
                        "data": "cvar_define",
                        "data_type": "i64",
                        "variable": "m",
                        "size": 1,
                    },
                    {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
                ],
            },
        },
    ]

    for case in test_cases:
        engine = PhirJsonEngine(json.dumps(case["phir"]))
        assert engine.process_program() == case["expected_commands"], case["name"]
