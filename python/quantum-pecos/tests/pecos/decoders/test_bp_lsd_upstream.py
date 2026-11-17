"""Pin BP+LSD defaults and corrections against the upstream 2.4.1 oracle."""

import json
from functools import partial
from pathlib import Path

import numpy as np
import pytest
from pecos_rslib.decoders import BpLsdBuilder, SparseMatrix


@pytest.mark.parametrize("explicit_settings", [False, True], ids=["defaults", "explicit"])
def test_bp_lsd_matches_upstream(explicit_settings: bool) -> None:
    reference = json.loads(
        (Path(__file__).parent / "fixtures" / "bp_lsd_upstream_ldpc.json").read_text(),
    )
    matrix = np.zeros((reference["num_checks"], reference["num_bits"]), dtype=np.uint8)
    for column, rows in enumerate(reference["columns"]):
        matrix[rows, column] = 1
    builder = BpLsdBuilder(SparseMatrix(matrix.tolist()), error_rate=reference["priors"])
    if explicit_settings:
        settings = reference["settings"]
        builder = (
            builder.max_iter(settings["max_iter"])
            .bp_method(settings["bp_method"])
            .ms_scaling_factor(settings["ms_scaling_factor"])
            .schedule(settings["schedule"])
            .bits_per_step(settings["bits_per_step"])
            .lsd_order(settings["lsd_order"])
        )
    decoder = builder.build()
    mismatches = []
    for index, case in enumerate(reference["cases"]):
        syndrome = np.zeros(reference["num_checks"], dtype=np.uint8)
        syndrome[case["syndrome"]] = 1
        correction = np.flatnonzero(decoder.decode(syndrome.tolist()).decoding).tolist()
        if correction != case["correction"]:
            mismatches.append((index, case["bp_converged"], correction, case["correction"]))
    assert not mismatches, f"{len(mismatches)}/{len(reference['cases'])} mismatches: {mismatches}"


@pytest.mark.parametrize("argument", ["bits_per_step", "lsd_order"])
@pytest.mark.parametrize("value", [-1, 1.5, "bad"])
@pytest.mark.parametrize("entry_point", ["builder", "dem", "generic_dem", "spec"])
def test_bp_lsd_integer_options_reject_invalid_values(argument, value, entry_point) -> None:
    from pecos_rslib.decoders import BpLsdDecoder, DemAwareDecoder, bp_lsd

    dem = "error(0.1) D0 L0"
    if entry_point == "builder":
        builder = BpLsdBuilder(SparseMatrix([[1]]), error_rate=0.1)
        invoke = partial(getattr(builder, argument), value)
    elif entry_point == "dem":
        invoke = partial(BpLsdDecoder.from_dem, dem, **{argument: value})
    elif entry_point == "generic_dem":
        invoke = partial(DemAwareDecoder.from_dem, dem, decoder_type="bp_lsd", **{argument: value})
    else:
        invoke = partial(bp_lsd, **{argument: value})
    with pytest.raises(ValueError, match=argument) as exc_info:
        invoke()
    assert argument in str(exc_info.value)
    assert str(value) in str(exc_info.value)


@pytest.mark.parametrize("entry_point", ["dem", "generic_dem", "spec"])
def test_bp_lsd_product_sum_and_growth_options(entry_point) -> None:
    from pecos_rslib.decoders import BpLsdDecoder, DemAwareDecoder, bp_lsd

    options = {"bp_method": "product_sum", "ms_scaling_factor": 0.75, "bits_per_step": 0, "lsd_order": 2}
    dem = "error(0.1) D0 L0"
    if entry_point == "spec":
        spec = bp_lsd(**options)
        assert spec == bp_lsd(ms_scaling_factor=None, bits_per_step=0, lsd_order=2)
        assert "ms_scaling_factor=None" in repr(spec)
    else:
        decoder = (
            BpLsdDecoder.from_dem(dem, **options)
            if entry_point == "dem"
            else DemAwareDecoder.from_dem(dem, decoder_type="bp_lsd", **options)
        )
        assert list(decoder.decode_syndrome([1]).observable_flips) == [True]


@pytest.mark.parametrize("entry_point", ["builder", "dem", "generic_dem", "spec"])
@pytest.mark.parametrize(
    ("argument", "value"),
    [("ms_scaling_factor", "0.5"), ("bits_per_step", "2"), ("lsd_order", "3")],
)
def test_bp_lsd_errors_quote_string_values(entry_point, argument, value) -> None:
    from pecos_rslib.decoders import BpLsdDecoder, DemAwareDecoder, bp_lsd

    dem = "error(0.1) D0 L0"
    if entry_point == "builder":
        builder = BpLsdBuilder(SparseMatrix([[1]]), error_rate=0.1)
        invoke = partial(getattr(builder, argument), value)
    elif entry_point == "dem":
        invoke = partial(BpLsdDecoder.from_dem, dem, **{argument: value})
    elif entry_point == "generic_dem":
        invoke = partial(DemAwareDecoder.from_dem, dem, decoder_type="bp_lsd", **{argument: value})
    else:
        invoke = partial(bp_lsd, **{argument: value})
    with pytest.raises(ValueError, match=argument) as exc_info:
        invoke()
    assert repr(value) in str(exc_info.value)


@pytest.mark.parametrize("argument", ["bp_method", "ms_scaling_factor", "bits_per_step", "lsd_order"])
def test_generic_dem_rejects_lsd_options_with_repr(argument) -> None:
    from pecos_rslib.decoders import DemAwareDecoder

    with pytest.raises(ValueError, match="only supported") as exc_info:
        DemAwareDecoder.from_dem("error(0.1) D0", "bp_osd", **{argument: "x"})
    assert argument in str(exc_info.value)
    assert repr("x") in str(exc_info.value)


@pytest.mark.parametrize("options", [{"bp_method": "x"}, {"ms_scaling_factor": "0.5"}])
def test_generic_dem_validates_decoder_type_first(options) -> None:
    from pecos_rslib.decoders import DemAwareDecoder

    with pytest.raises(ValueError, match="Unknown decoder type: bogus"):
        DemAwareDecoder.from_dem("error(0.1) D0", "bogus", **options)


@pytest.mark.parametrize("entry_point", ["dem", "generic_dem", "spec"])
def test_bp_lsd_scaling_signature_default(entry_point) -> None:
    from inspect import signature

    from pecos_rslib.decoders import BpLsdDecoder, DemAwareDecoder, bp_lsd

    factories = {"dem": BpLsdDecoder.from_dem, "generic_dem": DemAwareDecoder.from_dem, "spec": bp_lsd}
    assert signature(factories[entry_point]).parameters["ms_scaling_factor"].default == 0.625


@pytest.mark.parametrize("entry_point", ["builder", "dem", "generic_dem", "spec"])
@pytest.mark.parametrize("value", ["unknown", 'bad "method"', "bad\nmethod"])
def test_bp_lsd_invalid_method_uses_python_repr(entry_point, value) -> None:
    from pecos_rslib.decoders import BpLsdDecoder, DemAwareDecoder, bp_lsd

    dem = "error(0.1) D0"
    if entry_point == "builder":
        invoke = partial(BpLsdBuilder(SparseMatrix([[1]]), error_rate=0.1).bp_method, value)
    elif entry_point == "dem":
        invoke = partial(BpLsdDecoder.from_dem, dem, bp_method=value)
    elif entry_point == "generic_dem":
        invoke = partial(DemAwareDecoder.from_dem, dem, "bp_lsd", bp_method=value)
    else:
        invoke = partial(bp_lsd, bp_method=value)
    with pytest.raises(ValueError, match="bp_method") as exc_info:
        invoke()
    message = str(exc_info.value)
    assert repr(value) in message
    for spelling in ["minimum_sum", "product_sum", "ms", "ps"]:
        assert repr(spelling) in message


def test_generic_dem_rejects_explicit_none_scaling_for_bp_osd() -> None:
    from pecos_rslib.decoders import DemAwareDecoder

    with pytest.raises(ValueError, match=r"ms_scaling_factor=None.*only supported"):
        DemAwareDecoder.from_dem("error(0.1) D0", "bp_osd", ms_scaling_factor=None)


def test_generic_dem_other_none_options_remain_omitted() -> None:
    from pecos_rslib.decoders import DemAwareDecoder

    decoder = DemAwareDecoder.from_dem(
        "error(0.1) D0",
        "bp_osd",
        bp_method=None,
        bits_per_step=None,
        lsd_order=None,
    )
    assert decoder is not None


@pytest.mark.parametrize(
    ("argument", "value"),
    [("ms_scaling_factor", "bad"), ("bits_per_step", -1), ("lsd_order", 1.5), ("bp_method", "bad")],
)
def test_generic_bp_lsd_validates_options_before_dem(argument, value) -> None:
    from pecos_rslib.decoders import DemAwareDecoder

    with pytest.raises(ValueError, match=argument):
        DemAwareDecoder.from_dem("not a DEM", "bp_lsd", **{argument: value})


def test_valid_numeric_options_do_not_call_repr() -> None:
    class IntegerWithoutRepr(int):
        def __repr__(self):
            msg = "repr must not be called on valid integers"
            raise AssertionError(msg)

    class FloatWithoutRepr(float):
        def __repr__(self):
            msg = "repr must not be called on valid factors"
            raise AssertionError(msg)

    builder = BpLsdBuilder(SparseMatrix([[1]]), error_rate=0.1)
    builder.bits_per_step(IntegerWithoutRepr(1)).lsd_order(IntegerWithoutRepr(0))
    builder.ms_scaling_factor(FloatWithoutRepr(0.5))
    assert builder.build() is not None


@pytest.mark.parametrize("first", ["bp_method", "ms_scaling_factor", "bits_per_step"])
def test_lsd_only_options_report_first_without_rendering_later(first) -> None:
    """Non-LSD decoder types report the first LSD-only keyword without rendering later ones."""
    from pecos_rslib.decoders import DemAwareDecoder

    class Unrenderable:
        def __repr__(self):
            msg = "later argument must not be rendered"
            raise AssertionError(msg)

    with pytest.raises(ValueError, match=first):
        DemAwareDecoder.from_dem("error(0.1) D0", "bp_osd", **{first: "bad", "lsd_order": Unrenderable()})
