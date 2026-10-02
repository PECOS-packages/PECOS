"""Static HUGR result-tag analysis, using the Rust oracle's Guppy fixtures."""

from importlib.resources import as_file, files
from pathlib import Path

import pytest
from hugr import Hugr, ops, tys
from hugr.build.dfg import Dfg
from hugr.cli import convert
from hugr.envelope import EnvelopeConfig, EnvelopeFormat
from hugr.package import Package
from pecos.qec import _hugr_result_tags
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


@pytest.mark.parametrize("name", ["scrambled", "looped", "computed", "arr", "funcdecl"])
@pytest.mark.parametrize("envelope_format", [EnvelopeFormat.MODEL, EnvelopeFormat.JSON, EnvelopeFormat.S_EXPRESSION])
def test_loader_resolves_extensions_not_embedded_in_envelope(name: str, envelope_format: EnvelopeFormat) -> None:
    """Regression: the fixed registry must resolve extension-free envelopes."""
    embedded = _load(name)
    package = Package(modules=[embedded], extensions=[])
    data = package.to_bytes(
        EnvelopeConfig(format=EnvelopeFormat.MODEL if envelope_format == EnvelopeFormat.JSON else envelope_format),
    )
    if envelope_format == EnvelopeFormat.JSON:
        # hugr-py's JSON writer misserializes DataflowBlock outputs in 0.18.3.
        # The native converter preserves these graphs when producing JSON.
        with as_file(files("tket_exts").joinpath("data")) as extension_data:
            data = convert(data, format="json", extensions=[str(path) for path in extension_data.rglob("*.json")])
    if envelope_format != EnvelopeFormat.S_EXPRESSION:
        assert not Package.from_bytes(data).extensions
    loaded = load_hugr_from_bytes(data)
    assert measurement_op_count(loaded) == measurement_op_count(embedded)
    assert has_nontrivial_control_flow(loaded) == has_nontrivial_control_flow(embedded)
    assert extract_result_tag_measurements(loaded) == extract_result_tag_measurements(embedded)
    assert not any(isinstance(loaded[node].op, ops.Custom) for node in loaded)


def _unresolved_package() -> bytes:
    builder = Dfg()
    builder.add(ops.Custom("missing", tys.FunctionType([], []), "unknown.extension")())
    builder.set_outputs()
    return Package(modules=[builder.hugr]).to_bytes(EnvelopeConfig(format=EnvelopeFormat.JSON))


def test_loader_rejects_unknown_extension() -> None:
    with pytest.raises(ValueError, match=r"^Failed to parse HUGR"):
        load_hugr_from_bytes(_unresolved_package())


def test_loader_rejects_unresolved_operation_after_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the Python guard independently of the native validator's rejection."""
    monkeypatch.setattr(_hugr_result_tags, "validate", lambda *_args, **_kwargs: None)
    with pytest.raises(ValueError, match=r"^Failed to parse HUGR: unresolved operation unknown\.extension\.missing$"):
        load_hugr_from_bytes(_unresolved_package())
