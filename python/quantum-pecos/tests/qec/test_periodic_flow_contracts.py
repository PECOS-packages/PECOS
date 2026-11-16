"""Explicit signed extraction contracts for every periodic logical pair."""

from dataclasses import replace

import pytest
from pecos.qec.color import periodic_488_code, periodic_488_implementation


@pytest.mark.parametrize("distance", [4, 8, 12])
@pytest.mark.parametrize("gate_set", ["cnot", "native-cz"])
def test_all_published_checks_logical_pairs_and_constraints_are_verified(distance, gate_set):
    impl = periodic_488_implementation(periodic_488_code(distance), gate_set=gate_set)
    contracts = impl.flow_contracts()
    assert {c.name: c.parity for c in contracts} == {f.name: f.parity for f in impl.gadget().flows}
    assert sum(c.name.startswith("input:") for c in contracts) == 2 * len(impl.code.faces)
    assert sum(c.name.startswith("output:") for c in contracts) == 2 * len(impl.code.faces)
    assert {c.name for c in contracts if c.name.startswith("logical:")} == {
        f"logical:{axis}[i]".replace("[i]", f"[{i}]") for axis in "XZ" for i in range(4)
    }
    assert all(":" in c.input.name for c in contracts if c.name.startswith("input:"))
    audit = impl.audit_flows()
    assert audit.all_valid, audit.describe()
    audit.require_valid()
    assert audit.to_dict()["gadget"]["citations"]
    if gate_set == "native-cz":
        assert impl.gadget().input_h_frame
        assert any(c.parity.constant for c in contracts)


@pytest.mark.parametrize("gate_set", ["cnot", "native-cz"])
def test_each_logical_pair_wrong_sign_and_wrong_output_are_rejected(gate_set):
    impl = periodic_488_implementation(periodic_488_code(4), gate_set=gate_set)
    logicals = [c for c in impl.flow_contracts() if c.name.startswith("logical:")]
    bad = [replace(c, parity=c.parity ^ 1) for c in logicals]
    audit = impl.gadget().audit_flows(bad)
    assert all(r["support_valid"] and not r["valid"] for r in audit.to_dict()["contracts"])
    wrong = replace(logicals[0], output=logicals[2].output)
    assert not impl.gadget().audit_flows([wrong]).all_valid


def test_missing_cz_is_rejected_without_changing_published_contracts():
    impl = periodic_488_implementation(periodic_488_code(4), gate_set="native-cz")
    gadget = impl.gadget()
    layer = next(i for i, m in enumerate(gadget.moments) if any(g.name == "CZ" for g in m))
    j = next(i for i, g in enumerate(gadget.moments[layer]) if g.name == "CZ")
    moments = list(gadget.moments)
    moments[layer] = (*moments[layer][:j], *moments[layer][j+1:])
    assert not replace(gadget, moments=tuple(moments)).audit_flows(impl.flow_contracts()).all_valid
