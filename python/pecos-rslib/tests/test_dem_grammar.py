# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Public DEM surfaces agree with the pinned Stim grammar oracle."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import stim
from pecos_rslib.decoders import DemAwareDecoder, bp_osd, pymatching
from pecos_rslib.qec import DemSampler, ParsedDem, SampleBatch

ROOT = Path(__file__).resolve().parents[3]
SUPPORT = ROOT / "python/pecos-rslib/tests/dem_grammar_contract.py"
SPEC = importlib.util.spec_from_file_location("dem_grammar_contract", SUPPORT)
assert SPEC is not None
assert SPEC.loader is not None
CONTRACT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTRACT)
FIXTURE = ROOT / "crates/pecos-decoder-core/tests/fixtures/stim_dem_grammar.tsv"


def _unescape(text: str) -> str:
    chars = iter(text)
    return "".join({"n": "\n", "t": "\t", "\\": "\\"}[next(chars)] if char == "\\" else char for char in chars)


ROWS = [line.split("\t") for line in FIXTURE.read_text().splitlines()[1:]]


@pytest.mark.parametrize("row", ROWS, ids=[row[0] for row in ROWS])
@pytest.mark.parametrize("consumer", ["DemSampler", "ParsedDem", "DemAwareDecoder", "bp_osd", "pymatching"])
def test_public_dem_grammar(row: list[str], consumer: str) -> None:
    text = _unescape(row[0])
    accepted = row[1] == "accept"
    expected = None
    needs_flattening = False
    if accepted:
        oracle = stim.DetectorErrorModel(_unescape(row[2]))
        expected = oracle.num_detectors, oracle.num_observables
        needs_flattening = any(
            isinstance(instruction, stim.DemRepeatBlock) or instruction.type == "shift_detectors"
            for instruction in oracle
        )
    counts = None
    error = None
    try:
        if consumer == "DemSampler":
            model = DemSampler.from_dem_string(text)
            counts = model.num_detectors, model.num_observables
        elif consumer == "ParsedDem":
            model = ParsedDem.from_string(text)
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
    if needs_flattening:
        assert error is not None
        assert "requires a flattened DEM:" in error
    CONTRACT.assert_outcome(accepted, consumer, counts, expected, error)


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
