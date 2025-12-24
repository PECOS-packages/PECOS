"""Test to understand HUGR 0.13 structure from guppylang."""

import json
import tempfile

import pytest


def test_hugr_json_structure() -> None:
    """Examine HUGR JSON structure from guppylang."""
    try:
        from guppylang import guppy
        from guppylang.std.quantum import h, measure, qubit
    except ImportError:
        pytest.skip("guppylang not available")

    @guppy
    def simple_circuit() -> bool:
        q = qubit()
        h(q)
        return measure(q)

    # Compile to HUGR
    hugr = simple_circuit.compile()

    # Get JSON/string representation (use to_str if available)
    if hasattr(hugr, "to_str"):
        hugr_str = hugr.to_str()
        # Check if it's the envelope format with header
        if hugr_str.startswith("HUGRiHJv"):
            # Skip header (8 bytes), format byte (1 byte), and find JSON start
            json_start = hugr_str.find("{", 9)
            if json_start != -1:
                hugr_str = hugr_str[json_start:]
            else:
                msg = "Could not find JSON start in HUGR envelope"
                raise ValueError(msg)
    else:
        hugr_str = hugr.to_json()

    hugr_dict = json.loads(hugr_str)

    if "modules" in hugr_dict:
        for _i, module in enumerate(hugr_dict["modules"]):
            if "nodes" in module:
                # Print first few nodes
                for _j, _node in enumerate(module["nodes"][:5]):

                    pass

                    # Save to file for inspection
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(hugr_dict, f, indent=2)
