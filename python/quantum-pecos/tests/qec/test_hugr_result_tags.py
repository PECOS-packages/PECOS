"""Static HUGR result-tag analysis, using the Rust oracle's Guppy fixtures."""

from pathlib import Path

import pytest
from hugr import Hugr, ops, tys
from hugr.build.dfg import Dfg
from hugr.envelope import EnvelopeConfig
from hugr.package import Package
from pecos.qec._hugr_result_tags import (
    extract_result_tag_measurements,
    has_nontrivial_control_flow,
    load_hugr_from_bytes,
    measurement_op_count,
)

FIXTURES = Path(__file__).parent / "fixtures" / "result_tags"


def _load(name: str) -> Hugr:
    return load_hugr_from_bytes((FIXTURES / f"{name}.hugr").read_bytes())


def _has_branch_or_loop(hugr: Hugr) -> bool:
    return any(
        isinstance(hugr[node].op, (ops.Conditional, ops.TailLoop))
        or (isinstance(hugr[node].op, ops.DataflowBlock) and len(hugr[node].op.sum_ty.variant_rows) > 1)
        for node in hugr
    )


def test_bodyless_declared_function_is_nontrivial_control_flow() -> None:
    hugr = _load("funcdecl")
    assert not _has_branch_or_loop(hugr)
    assert not any(isinstance(hugr[node].op, ops.CallIndirect) for node in hugr)
    assert has_nontrivial_control_flow(hugr)


def test_indirect_call_is_nontrivial_control_flow() -> None:
    """A real indirect call is the only reason this graph is nontrivial."""
    builder = Dfg(tys.FunctionType([tys.Bool], [tys.Bool]), tys.Bool)
    function, bit = builder.inputs()
    call = builder.add(ops.CallIndirect()(function, bit))
    builder.set_outputs(call)
    hugr = load_hugr_from_bytes(builder.hugr.to_bytes())
    assert any(isinstance(hugr[node].op, ops.CallIndirect) for node in hugr)
    assert not _has_branch_or_loop(hugr)
    assert not any(isinstance(hugr[node].op, ops.FuncDecl) for node in hugr)
    assert has_nontrivial_control_flow(hugr)


def test_scrambled_binds_each_tag_to_its_measurement() -> None:
    tags = extract_result_tag_measurements(_load("scrambled"))
    assert tags == {"tag_a": [0], "tag_b": [1], "tag_c": [2]}
    assert list(tags) == ["tag_a", "tag_b", "tag_c"]


def test_looped_tag_is_single_static_measure_op() -> None:
    hugr = _load("looped")
    assert extract_result_tag_measurements(hugr)["synx"] == [0]
    assert has_nontrivial_control_flow(hugr)


def test_straight_line_program_has_no_nontrivial_control_flow() -> None:
    assert not has_nontrivial_control_flow(_load("scrambled"))


def test_computed_and_constant_tags_are_excluded() -> None:
    tags = extract_result_tag_measurements(_load("computed"))
    assert tags["eq"] == [None]
    assert tags["const"] == [None]


def test_array_valued_tag_is_excluded() -> None:
    assert extract_result_tag_measurements(_load("arr"))["pair"] == [None]


@pytest.mark.parametrize(
    ("name", "count"),
    [("scrambled", 3), ("looped", 1), ("computed", 2), ("arr", 1), ("funcdecl", 1)],
)
def test_measurement_op_count(name: str, count: int) -> None:
    assert measurement_op_count(_load(name)) == count


@pytest.mark.parametrize("data", [b"", b"corrupt HUGR", b"HUGRiHJv\x00\x40{broken json"])
def test_loader_rejects_invalid_bytes(data: bytes) -> None:
    with pytest.raises(ValueError, match=r"^Failed to parse HUGR"):
        load_hugr_from_bytes(data)


def test_loader_rejects_empty_package() -> None:
    with pytest.raises(ValueError, match=r"^Failed to parse HUGR: Package contains no modules"):
        load_hugr_from_bytes(Package(modules=[]).to_bytes())


def test_loader_validates_every_module() -> None:
    """A parseable package with an invalid second module must still fail."""
    valid = Dfg()
    valid.set_outputs()
    invalid = Dfg(tys.Bool)
    (bit,) = invalid.inputs()
    invalid.set_outputs(bit)
    source = bit.out_port()
    (target,) = invalid.hugr.linked_ports(source)
    invalid.hugr.delete_link(source, target)
    data = Package(modules=[valid.hugr, invalid.hugr]).to_bytes()
    assert len(Package.from_bytes(data).modules) == 2
    with pytest.raises(ValueError, match=r"^Failed to parse HUGR: Error validating HUGR"):
        load_hugr_from_bytes(data)


def test_loader_valid_bytes_do_not_print_to_stdout(capfd: pytest.CaptureFixture[str]) -> None:
    assert measurement_op_count(_load("scrambled")) == 3
    assert capfd.readouterr().out == ""


def test_loader_uses_first_module() -> None:
    package = Package.from_bytes((FIXTURES / "scrambled.hugr").read_bytes())
    package.modules.append(_load("computed"))
    assert extract_result_tag_measurements(load_hugr_from_bytes(package.to_bytes(EnvelopeConfig()))) == {
        "tag_a": [0],
        "tag_b": [1],
        "tag_c": [2],
    }
