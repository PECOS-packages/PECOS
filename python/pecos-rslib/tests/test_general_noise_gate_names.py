"""General noise uses the shared gate parser at the Python boundary."""

import pytest
from pecos_rslib import general_noise


@pytest.mark.parametrize("gate", ["PZ", "MZ", "CNOT", "pz", "mz", "cnot", "PREP", "MEASURE", "CX", "RESET", "CZ", "SX"])
def test_noiseless_gate_accepts_shared_names(gate: str) -> None:
    """Canonical names, aliases, and case variants all preserve immutable chaining."""
    builder = general_noise()
    updated = builder.with_noiseless_gate(gate)
    assert updated is not builder
    assert updated.with_p1(0.01) is not None


def test_noiseless_gate_rejects_unknown_name() -> None:
    """Invalid names still raise the Python API's existing ValueError."""
    with pytest.raises(ValueError, match="Invalid gate type: unknown"):
        general_noise().with_noiseless_gate("unknown")
