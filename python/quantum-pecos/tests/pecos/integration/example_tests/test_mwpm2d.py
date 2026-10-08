"""MWPM2D must correct physical errors using measured ancilla labels."""

import pecos as pc
import pytest
from pecos.decoders.mwpm2d import precomputing
from pecos.engines.std_output import StdOutput


@pytest.fixture(params=[pc.qeccs.Surface4444, pc.qeccs.SurfaceMedial4444])
def qecc(request, distance):
    """Exercise both precomputation paths."""
    return request.param(distance=distance)


@pytest.fixture(params=[3, 5])
def distance(request):
    """Use distances that can correct any single data error."""
    return request.param


@pytest.mark.parametrize("pauli", ["X", "Z", "Y"])
def test_single_data_errors(qecc, distance, pauli) -> None:
    """Correct every data error in both logical bases, including boundary errors."""
    decoder = pc.decoders.MWPM2D(qecc)
    runner = pc.circuit_runners.Standard(seed=1)
    extraction = pc.circuits.LogicalCircuit(suppress_warning=True)
    extraction.append(qecc.gate("I", num_syn_extract=1))
    logical_ops = qecc.instruction("instr_syn_extract").final_logical_ops[0]

    for qudit in sorted(qecc.data_qudit_set):
        error = pc.circuits.QuantumCircuit([{pauli: {qudit}}])
        for basis, logical in [("|0>", "Z"), ("|+>", "X")]:
            context = (qecc.name, distance, pauli, qudit, basis)
            init = pc.circuits.LogicalCircuit(suppress_warning=True)
            init.append(qecc.gate(f"ideal init {basis}"))
            state = pc.simulators.SparseStabPy(qecc.num_qudits)
            prepared, _ = runner.run(state, init)
            assert not prepared.simplified(last=True), context
            initial_sign = state.logical_sign(logical_ops[logical])
            runner.run(state, error)
            measurements, _ = runner.run(state, extraction)
            assert measurements.simplified(last=True), context
            recovery = decoder.decode(measurements)
            key = frozenset(measurements.simplified(last=True))
            assert decoder.recorded_recovery[key] is recovery
            assert decoder.decode(measurements) is recovery
            assert set().union(*recovery.active_qudits) <= qecc.data_qudit_set, context
            assert pc.quantum.commute.qubit_pauli(
                logical_ops[logical],
                error,
            ) == pc.quantum.commute.qubit_pauli(logical_ops[logical], recovery), context
            runner.run(state, recovery)
            assert state.logical_sign(logical_ops[logical]) == initial_sign, context
            residual, _ = runner.run(state, extraction)
            assert not residual.simplified(last=True), context


@pytest.mark.parametrize("entry_point", ["precompute", "code", "identity"])
def test_precompute_node_maps(qecc, entry_point) -> None:
    """Every entry point retains disjoint ancilla maps and physical recovery paths."""
    instr = qecc.instruction("instr_syn_extract")
    medial = qecc.name == "Medial 4.4.4.4 Surface Code"
    if entry_point == "precompute":
        compute = precomputing.precompute
    elif entry_point == "code":
        compute = precomputing.code_surface4444medial if medial else precomputing.code_surface4444
    else:
        compute = precomputing.surface4444medial_identity if medial else precomputing.surface4444_identity
    info = compute(instr)
    real_labels = {}
    for check_type in ["X", "Z"]:
        data = info[check_type]
        node_map = data["node_map"]
        real_labels[check_type] = node_map.keys() & qecc.ancilla_qudit_set
        expected = {
            params["ancillas"]
            for symbol, _, params in instr.abstract_circuit.items()
            if symbol == f"{check_type} check"
        }
        assert real_labels[check_type] == expected
        assert set(node_map.values()) == set(data["dist_graph"].nodes())
        assert len(set(node_map.values())) == len(node_map)
        assert {node_map[label] for label in expected} == data["virtual_edge_data"].keys()
        for edge in data["virtual_edge_data"].values():
            assert set(edge["data_path"]) <= qecc.data_qudit_set
        for n1, n2, _ in data["dist_graph"].edges():
            assert set(data["dist_graph"].edge_attrs(n1, n2)["data_path"]) <= qecc.data_qudit_set
    assert not real_labels["X"] & real_labels["Z"]
    assert real_labels["X"] | real_labels["Z"] == qecc.ancilla_qudit_set


def test_unknown_syndrome_labels(qecc) -> None:
    """Reject all unknown labels, even alongside a valid syndrome or a graph node ID."""
    decoder = pc.decoders.MWPM2D(qecc)
    unknown = {0, "unknown", qecc.num_qudits + 100}
    measurements = StdOutput()
    measurements.record(dict.fromkeys(unknown | {min(qecc.ancilla_qudit_set)}, 1), 0)
    with pytest.raises(ValueError, match="Unknown syndrome labels") as exc:
        decoder.decode(measurements)
    for label in unknown:
        assert repr(label) in str(exc.value)
    assert not decoder.recorded_recovery


def test_empty_syndrome_cache(qecc) -> None:
    """Empty measurements produce an empty recovery cached under the original syndrome."""
    decoder = pc.decoders.MWPM2D(qecc)
    measurements = StdOutput()
    recovery = decoder.decode(measurements)
    assert not set().union(*recovery.active_qudits)
    assert decoder.recorded_recovery == {frozenset(): recovery}
    assert decoder.decode(measurements) is recovery


def test_ancilla_in_both_check_types() -> None:
    """Precomputation rejects an ambiguous real syndrome label."""
    with pytest.raises(ValueError, match=r"both check types: \[1\]"):
        precomputing.invert_data(
            {"X": {}, "Z": {}},
            {7: [1, "vx"]},
            {7: [1, "vz"]},
            {},
            {},
            pc.graph.Graph(),
            pc.graph.Graph(),
            {"vx"},
            {"vz"},
        )
