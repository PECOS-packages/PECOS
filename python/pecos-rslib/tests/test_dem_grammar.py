# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Public DEM surfaces agree with the pinned Stim grammar oracle."""

from __future__ import annotations

import importlib.util
from functools import partial
from pathlib import Path

import pytest
import stim
import pecos_rslib.decoders as decoders
from pecos_rslib.decoders import DemAwareDecoder, PyMatchingDecoder, bp_osd, pymatching
from pecos_rslib.qec import DemSampler, ParsedDem, SampleBatch

ROOT = Path(__file__).resolve().parents[3]
SUPPORT = ROOT / "python/pecos-rslib/tests/dem_grammar_contract.py"
SPEC = importlib.util.spec_from_file_location("dem_grammar_contract", SUPPORT)
assert SPEC is not None
assert SPEC.loader is not None
CONTRACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTRACT)
# Dense per-detector storage is sized by the maximum index.
DENSE_INDEX_THRESHOLD = 1 << 24

FIXTURE = ROOT / "crates/pecos-decoder-core/tests/fixtures/stim_dem_grammar.tsv"

STIM_EXCEPTIONS = {
    "error(0.1) D0 D0": "a target may appear at most once per component",
    "error(0.1) D0 ^ D0": "component D0 is repeated",
}


def _unescape(text: str) -> str:
    chars = iter(text)
    result = []
    for char in chars:
        if char != "\\":
            result.append(char)
            continue
        escape = next(chars)
        if escape == "u":
            assert next(chars) == "{"
            digits = []
            for digit in chars:
                if digit == "}":
                    break
                digits.append(digit)
            result.append(chr(int("".join(digits), 16)))
        else:
            result.append({"n": "\n", "t": "\t", "\\": "\\"}[escape])
    return "".join(result)


ROWS = [line.split("\t") for line in FIXTURE.read_text().splitlines()[1:]]


@pytest.mark.parametrize("row", ROWS, ids=[row[0] for row in ROWS])
@pytest.mark.parametrize(
    "consumer", ["DemSampler", "ParsedDem", "DemAwareDecoder", "bp_osd", "pymatching", "PyMatchingDecoder"]
)
def test_public_dem_grammar(row: list[str], consumer: str) -> None:
    text = _unescape(row[0])
    accepted = row[1] == "accept" and text not in STIM_EXCEPTIONS
    expected = None
    needs_flattening = False
    if accepted:
        oracle = stim.DetectorErrorModel(_unescape(row[2] if len(row) > 2 else ""))
        expected = oracle.num_detectors, oracle.num_observables
        needs_flattening = any(
            isinstance(instruction, stim.DemRepeatBlock) or instruction.type == "shift_detectors"
            for instruction in oracle
        )
    if expected is not None and max(expected) - 1 >= DENSE_INDEX_THRESHOLD:
        if consumer == "PyMatchingDecoder" or (
            consumer in {"DemAwareDecoder", "bp_osd", "pymatching"} and max(expected) - 1 <= (1 << 32) - 1
        ):
            pytest.skip("dense per-detector storage is sized by the maximum index")
        # Larger indices still exercise each PECOS consumer's own overflow rejection.
    counts = None
    error = None
    try:
        if consumer == "DemSampler":
            model = DemSampler.from_dem_string(text)
            counts = model.num_detectors, model.num_observables
        elif consumer == "ParsedDem":
            model = ParsedDem.from_string(text)
            counts = model.num_detectors, model.num_observables
        elif consumer == "PyMatchingDecoder":
            model = PyMatchingDecoder.from_dem(text)
            counts = model.num_detectors, model.num_observables
        elif consumer == "DemAwareDecoder":
            model = DemAwareDecoder.from_dem(text, decoder_type="bp_osd")
            counts = model.num_detectors, model.num_observables
        else:
            # Index overflow is checked during construction before any syndrome
            # allocation. Ordinary models also exercise decoding a zero syndrome.
            detectors, observables = expected or (0, 0)
            events = [[0] * detectors] if detectors <= 4294967295 else []
            batch = SampleBatch(events, [0] * len(events), num_observables=observables)
            batch.decode(text, bp_osd() if consumer == "bp_osd" else pymatching(correlated=False))
    except (ValueError, RuntimeError, OverflowError) as exc:
        error = str(exc)
    if needs_flattening and consumer != "PyMatchingDecoder":
        assert error is not None
        assert "requires a flattened DEM:" in error
    if text in STIM_EXCEPTIONS:
        assert error is not None
        assert STIM_EXCEPTIONS[text] in error
    CONTRACT.assert_outcome(text, accepted, consumer, counts, expected, error)


def test_tagged_mechanism_is_sampled_and_counted() -> None:
    text = "ERROR[tag](1) d0 l0 # D99 L99"
    sampler = DemSampler.from_dem_string(text)
    assert sampler.sample(seed=7) == ([True], [True])
    assert ParsedDem.from_string(text).num_mechanisms == 1
    assert DemAwareDecoder.from_dem(text).num_mechanisms == 1


def test_empty_probability_has_no_sampling_effect() -> None:
    sampler = DemSampler.from_dem_string("error() D0 L0")
    assert sampler.sample(seed=7) == ([False], [False])


def test_zero_probability_mechanism_is_not_registered() -> None:
    sampler = DemSampler.from_dem_string("error(0) D0\nerror(1) D1")
    assert sampler.num_mechanisms == 1
    assert sampler.sample(seed=7) == ([False, True], [])


@pytest.mark.parametrize(
    "targets",
    ["D0 D1 ^ D1 D2", "D0 L0 ^ D1 L0", "D0 D1 ^ D1 D2 ^ D2 D0"],
)
def test_cross_component_effects_match_stim(targets: str) -> None:
    text = f"error(1) {targets}"
    sampler = DemSampler.from_dem_string(text)
    assert sampler.num_mechanisms == 1
    detectors, observables, _ = stim.DetectorErrorModel(text).compile_sampler().sample(4)
    for seed, (dets, obs) in enumerate(zip(detectors, observables, strict=True)):
        assert sampler.sample(seed=seed) == (dets.tolist(), obs.tolist())


def test_detector_free_observable_mechanism_keeps_zero_column() -> None:
    text = "error(0.1) L0\nerror(0.2) D0 L0"
    decoder = decoders.BpOsdDecoder.from_dem(text)
    assert decoder.num_mechanisms == 2
    assert decoder.num_detectors == 1
    with pytest.raises(RuntimeError, match=r"Column weight is zero"):
        decoders.UnionFindDecoder.from_dem(text)


@pytest.mark.parametrize(
    "text",
    [
        "error[unclosed(0.1) D0",
        r"error[bad\t](0.1) D0",
        "repeat 2",
        "repeat {",
        "error(0.1) D18446744073709551616",
        "pecos_observable{}",
        "@bad",
    ],
)
@pytest.mark.parametrize("consumer", ["DemSampler", "ParsedDem", "DemAwareDecoder", "bp_osd", "pymatching"])
def test_syntax_discriminant_survives_public_wrappers(text: str, consumer: str) -> None:
    test_public_dem_grammar([text.replace("\\", "\\\\"), "reject"], consumer)


@pytest.mark.parametrize(
    "text",
    [
        "\x0cerror(0.1) D0",
        "\x0berror(0.1) D0",
        "repeat 2{\n}",
        "repeat 2 {}",
        "repeat 0 {\n}",
    ],
)
@pytest.mark.parametrize("consumer", ["DemSampler", "ParsedDem", "DemAwareDecoder", "bp_osd", "pymatching"])
def test_whitespace_and_repeat_variants_on_public_surfaces(text: str, consumer: str) -> None:
    canonical = str(stim.DetectorErrorModel(text))
    test_public_dem_grammar([text, "accept", canonical], consumer)


DECODER_CLASSES = [
    decoder
    for name, decoder in sorted(vars(decoders).items())
    if name.endswith("Decoder") and isinstance(decoder, type) and hasattr(decoder, "from_dem")
]


@pytest.mark.parametrize("decoder", DECODER_CLASSES, ids=lambda decoder: decoder.__name__)
def test_exported_decoder_constructors_classify_dem_syntax(decoder: type) -> None:
    with pytest.raises(ValueError, match="Invalid DEM syntax:"):
        decoder.from_dem("@bad")


@pytest.mark.parametrize(
    "consumer",
    ["DemSampler", "ParsedDem", "DemAwareDecoder", "bp_osd", "pymatching", "PyMatchingDecoder", "TesseractDecoder"],
)
def test_duplicate_target_is_a_syntax_error(consumer: str) -> None:
    text = "error(0.1) D0 D0"
    if consumer == "DemSampler":
        construct = partial(DemSampler.from_dem_string, text)
    elif consumer == "ParsedDem":
        construct = partial(ParsedDem.from_string, text)
    elif consumer in {"bp_osd", "pymatching"}:
        construct = partial(
            SampleBatch([[0]], [0], num_observables=0).decode,
            text,
            bp_osd() if consumer == "bp_osd" else pymatching(correlated=False),
        )
    else:
        construct = partial(getattr(decoders, consumer).from_dem, text)
    with pytest.raises(ValueError, match=r"Invalid DEM syntax: detector D0 is listed twice"):
        construct()
