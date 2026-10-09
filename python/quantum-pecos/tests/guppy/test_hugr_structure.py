"""Test to understand HUGR 0.13 structure from guppylang."""


def test_hugr_json_structure() -> None:
    """Examine HUGR structure from guppylang."""
    from guppylang import guppy
    from guppylang.std.quantum import h, measure, qubit

    @guppy
    def simple_circuit() -> bool:
        q = qubit()
        h(q)
        return measure(q).read()

    # Compile to HUGR (a hugr.package.Package)
    pkg = simple_circuit.compile()

    # Inspect the Package structure directly
    assert len(pkg.modules) >= 1, "HUGR should contain at least one module"
    assert sum(1 for _ in pkg.modules[0].nodes()) > 0, "First module should contain at least one node"
