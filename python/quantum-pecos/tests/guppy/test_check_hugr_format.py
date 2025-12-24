"""Check HUGR format from guppylang."""

import json

import pytest


def test_check_hugr_format() -> None:
    """Check what HUGR format guppylang produces."""
    try:
        from guppylang import guppy
        from guppylang.std.quantum import h, measure, qubit
    except ImportError:
        pytest.skip("guppylang not available")

    @guppy
    def simple() -> bool:
        q = qubit()
        h(q)
        return measure(q)

    # Compile to HUGR
    hugr = simple.compile()

    # Check binary format
    hugr.to_bytes()

    # Check JSON/string format
    # Note: to_str() returns HUGR envelope format with header, while to_json() returns pure JSON
    if hasattr(hugr, "to_str"):
        hugr_str = hugr.to_str()
        # Check if it's the envelope format with header
        if hugr_str.startswith("HUGRiHJv"):
            # Skip header (8 bytes), format byte (1 byte), and extra byte (1 byte)
            json_start = hugr_str.find("{", 9)  # Find the start of JSON after header
            if json_start != -1:
                hugr_str = hugr_str[json_start:]
            else:
                msg = "Could not find JSON start in HUGR envelope"
                raise ValueError(msg)
    else:
        hugr_str = hugr.to_json()

    hugr_dict = json.loads(hugr_str)

    # Check if it's a single HUGR or a Package
    if "modules" in hugr_dict or "nodes" in hugr_dict:
        pass

    # Save JSON for inspection
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(hugr_dict, f, indent=2)
