"""Open-channel signed-flow contracts, with independent matrix Clifford oracles."""

from dataclasses import replace

import numpy as np
import pytest
from pecos.qec.color import ColorExtraction, color_code_implementation, color_code_spec
from pecos.qec.generic import Gadget, Gate, NamedParity, NamedPauli, PauliFlow


def _pauli(axis, role="data"):
    return NamedPauli(axis, ((role, axis),))


def _unitary(gates=(), **kwargs):
    return Gadget("unitary", ("data",), ("data",), ("data",),
                  tuple((Gate(g, ("data",)),) for g in gates), **kwargs)


def _cliffords():
    # Dense H/S matrices and breadth-first closure modulo global phase. No Stim
    # tableaus, flow generators, or PECOS gate propagation used for the oracle.
    h = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    s = np.diag([1, 1j])
    pending = [((), np.eye(2, dtype=complex))]
    seen, result = set(), []
    while pending:
        gates, u = pending.pop(0)
        first = next(v for v in u.flatten() if abs(v) > 1e-8)
        canonical = tuple(np.round(u.flatten() / (first / abs(first)), 8))
        if canonical in seen:
            continue
        seen.add(canonical)
        result.append((gates, u))
        pending.extend(((*gates, name), gate @ u) for name, gate in (("H", h), ("S", s)))
    assert len(result) == 24
    return result


@pytest.mark.parametrize(("gates", "matrix"), _cliffords(), ids=lambda v: str(v) if isinstance(v, tuple) else "matrix")
def test_all_single_qubit_clifford_flows_match_dense_matrix_signs(gates, matrix):
    paulis = {"X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]), "Z": np.diag([1, -1])}
    contracts, expected = [], []
    for incoming, p in paulis.items():
        conjugated = matrix @ p @ matrix.conj().T
        for outgoing, q in paulis.items():
            coefficient = np.trace(q @ conjugated).real / 2
            for sign in (False, True):
                contracts.append(PauliFlow(f"{incoming}-{outgoing}-{sign}", _pauli(incoming), _pauli(outgoing),
                                           NamedParity(constant=sign)))
                expected.append(bool(abs(coefficient - (-1 if sign else 1)) < 1e-8))
    report = _unitary(gates).audit_flows(contracts).to_dict()
    assert [row["valid"] for row in report["contracts"]] == expected


@pytest.mark.parametrize(("incoming", "outgoing"), [(False, False), (False, True), (True, False), (True, True)])
def test_boundary_h_frames_include_y_signs(incoming, outgoing):
    gadget = _unitary(input_h_frame=("data",) if incoming else (), output_h_frame=("data",) if outgoing else ())
    flow = PauliFlow("Y transport", _pauli("Y"), _pauli("Y"), NamedParity(constant=incoming != outgoing))
    audit = gadget.audit_flows([flow, replace(flow, name="wrong", parity=flow.parity ^ 1)])
    assert [r["valid"] for r in audit.to_dict()["contracts"]] == [True, False]
    assert "wrong sign" in audit.describe()


def _zz():
    return Gadget("ZZ extraction", ("left", "right", "ancilla"), ("left", "right"), ("left", "right"),
                  ((Gate("R", "ancilla"),), (Gate("CX", ("left", "ancilla")),),
                   (Gate("CX", ("right", "ancilla")),), (Gate("M", "ancilla", "bit"),)))


def test_measurement_backaction_and_arbitrary_open_inputs_are_checked():
    zz = NamedPauli("ZZ", (("left", "Z"), ("right", "Z")))
    xx = NamedPauli("XX", (("left", "X"), ("right", "X")))
    x = _pauli("X", "left")
    bit = NamedParity("bit")
    audit = _zz().audit_flows([
        PauliFlow("measure", input=zz, parity=bit),
        PauliFlow("project", output=zz, parity=bit),
        PauliFlow("commuting", xx, xx),
        PauliFlow("backaction", x, x),
        PauliFlow("random bit", parity=bit),
        PauliFlow("wrong sign", input=zz, parity=bit ^ 1),
    ])
    assert [row["valid"] for row in audit.to_dict()["contracts"]] == [True, True, True, False, False, False]
    with pytest.raises(ValueError, match="backaction, random bit, wrong sign"):
        audit.require_valid()
    report = _unitary().audit_flows([PauliFlow("unprepared", output=_pauli("Z")),
                                    PauliFlow("unmeasured", input=_pauli("Z"))]).to_dict()
    assert not any(row["valid"] for row in report["contracts"])


def test_measure_reset_reuse_and_constant_checks():
    gadget = Gadget("reuse", ("data",), ("data",), ("data",),
                    ((Gate("M", "data", "old"),), (Gate("R", "data"),),
                     (Gate("X", "data"),), (Gate("M", "data", "one"),), (Gate("R", "data"),)))
    contracts = [PauliFlow("old Z measured", input=_pauli("Z"), parity=NamedParity("old")),
                 PauliFlow("new Z prepared", output=_pauli("Z")),
                 PauliFlow("one is one", parity=NamedParity("one") ^ 1),
                 PauliFlow("no constant", parity=NamedParity("one")),
                 PauliFlow("no old X transport", _pauli("X"), _pauli("X"))]
    assert [r["valid"] for r in gadget.audit_flows(contracts).to_dict()["contracts"]] == [
        True, True, True, False, False,
    ]


def test_relabel_changes_carriers_without_inserting_swap():
    gadget = Gadget("relabel", ("a", "b"), ("a", "b"), ("a", "b"), ((Gate("RELABEL", ("a", "b")),),))
    contracts = [PauliFlow("transport", _pauli("Y", "a"), _pauli("Y", "b")),
                 PauliFlow("wrong role", _pauli("Y", "a"), _pauli("Y", "a"))]
    report = gadget.audit_flows(contracts).to_dict()
    assert [r["valid"] for r in report["contracts"]] == [True, False]
    assert report["output_carriers"] == {"a": 1, "b": 0}
    assert "SWAP" not in report["analysis_circuit"]
    assert "RELABEL" not in report["analysis_circuit"]


def test_empty_ports_identity_and_impossible_sign_are_distinct():
    gadget = Gadget("empty", (), (), (), ((),))
    report = gadget.audit_flows([PauliFlow("identity"), PauliFlow("impossible", parity=NamedParity(constant=True))])
    assert [r["valid"] for r in report.to_dict()["contracts"]] == [True, False]


@pytest.mark.parametrize(("kind", "distance"), [
    (kind, d) for kind in ColorExtraction
    for d in ((3, 7, 11) if kind in (ColorExtraction.BRICKWORK_CX, ColorExtraction.BRICKWORK_CZ) else (3, 5, 7))
])
def test_paper_extraction_phases_have_all_signed_contracts(kind, distance):
    family = "488" if kind in (ColorExtraction.BRICKWORK_CX, ColorExtraction.BRICKWORK_CZ) else "666"
    implementation = color_code_implementation(color_code_spec(family, distance), kind)
    for phase in range(implementation.period):
        contracts = implementation.flow_contracts(phase)
        gadget = implementation.gadget(phase)
        assert {f.name: f.parity for f in contracts} == {f.name: f.parity for f in gadget.flows}
        audit = implementation.audit_flows(phase)
        assert audit.all_valid, audit.describe()
        audit.require_valid()
        assert audit.to_dict()["gadget"]["citations"]
        bad = replace(contracts[0], parity=contracts[0].parity ^ 1)
        assert not gadget.audit_flows([bad]).all_valid


def test_paper_missing_physical_gate_is_detected():
    impl = color_code_implementation(color_code_spec("488", 3), ColorExtraction.BRICKWORK_CZ)
    gadget = impl.gadget()
    index = next(i for i, moment in enumerate(gadget.moments) if any(g.name == "CZ" for g in moment))
    moment = gadget.moments[index]
    damaged = replace(gadget, moments=(*gadget.moments[:index], moment[1:], *gadget.moments[index+1:]))
    assert not damaged.audit_flows(impl.flow_contracts()).all_valid


@pytest.mark.parametrize("contracts", [[], ["not a flow"], [PauliFlow("a"), PauliFlow("a")],
                                      [PauliFlow("bad input", input=_pauli("X", "missing"))],
                                      [PauliFlow("bad output", output=_pauli("Z", "missing"))],
                                      [PauliFlow("bad outcome", parity=NamedParity("missing"))]])
def test_invalid_contracts_rejected_before_verification(contracts):
    with pytest.raises(ValueError, match=r"nonempty|unique|boundary ports|unknown local outcome"):
        _unitary().audit_flows(contracts)


def test_scratch_ports_cannot_be_used_as_quantum_endpoints():
    with pytest.raises(ValueError, match="boundary ports"):
        _zz().audit_flows([PauliFlow("scratch", output=_pauli("Z", "ancilla"))])


def test_invalid_types_and_detached_report():
    with pytest.raises(TypeError, match="NamedPauli"):
        PauliFlow("bad", input="X")
    with pytest.raises(TypeError, match="NamedParity"):
        PauliFlow("bad", parity=1)
    with pytest.raises(ValueError, match="Names"):
        PauliFlow("")
    flow = PauliFlow("identity", _pauli("X"), _pauli("X"))
    report = _unitary().audit_flows([flow])
    data = report.to_dict()
    data["contracts"][0]["valid"] = False
    assert report.all_valid
    assert not data["complete_channel_specification_verified"]
    assert not data["fault_tolerance_verified"]
    assert not data["noise_equivalence_verified"]
