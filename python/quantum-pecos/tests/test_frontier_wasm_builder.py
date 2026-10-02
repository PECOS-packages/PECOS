# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0
"""Exercise corpus ingestion at the Frontier Wasm build boundary."""

import importlib.util
import struct
from pathlib import Path

import pytest
from pecos.qec import SampleBatch

BUILDER_PATH = Path(__file__).resolve().parents[3] / "scripts" / "build_frontier_wasm.py"
SPEC = importlib.util.spec_from_file_location("frontier_wasm_builder", BUILDER_PATH)
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def test_corpus_packs_wide_words_and_bounds_shots(tmp_path: Path) -> None:
    """Preserve detector and observable bits across every ABI word boundary."""
    dem = "error(0.1) D128 L69\n"
    syndrome = [int(bit in (0, 31, 32, 127, 128)) for bit in range(129)]
    mask = (1 << 69) | (1 << 32) | 1
    corpus = tmp_path / "shots.pecos"
    SampleBatch([syndrome, [0] * 129], [mask, 0], num_observables=70).save(corpus, dem=dem)
    fixture = tmp_path / "shots.fwr"
    metadata = BUILDER.pack_replay(corpus, fixture, dem, max_shots=1)
    assert metadata == {"shots": 1, "detectors": 129, "observables": 70}
    assert struct.unpack("<4sIIII8I", fixture.read_bytes()) == (
        b"FWR1",
        2,
        1,
        129,
        70,
        0x80000001,
        1,
        0,
        0x80000000,
        1,
        1,
        1,
        32,
    )
    BUILDER.pack_replay(corpus, fixture, dem)
    assert len(fixture.read_bytes()) == 20 + 2 * 32
    assert fixture.read_bytes()[-32:] == bytes(32)


def test_corpus_rejects_different_same_width_model_and_corruption(tmp_path: Path) -> None:
    """Widths alone must not permit replay against a different model."""
    corpus = tmp_path / "shots.pecos"
    dem = "error(0.1) D0 L0\n"
    SampleBatch([[1]], [1], num_observables=1).save(corpus, dem=dem)
    fixture = tmp_path / "shots.fwr"
    with pytest.raises(ValueError, match="exactly match"):
        BUILDER.pack_replay(corpus, fixture, dem.replace("0.1", "0.2"))
    assert not fixture.exists()
    with pytest.raises(ValueError, match="positive"):
        BUILDER.pack_replay(corpus, fixture, dem, max_shots=0)
    damaged = bytearray(corpus.read_bytes())
    damaged[-1] ^= 1
    corpus.write_bytes(damaged)
    with pytest.raises(ValueError, match=r"checksum|SHA|hash|digest"):
        BUILDER.pack_replay(corpus, fixture, dem)


def test_corpus_rejects_observables_beyond_result_abi(tmp_path: Path) -> None:
    """A valid corpus may still exceed the Wasm correction width."""
    corpus = tmp_path / "wide.pecos"
    dem = "error(0.1) D0 L128\n"
    SampleBatch([[1]], [1 << 128], num_observables=129).save(corpus, dem=dem)
    with pytest.raises(ValueError, match="128-observable"):
        BUILDER.pack_replay(corpus, tmp_path / "wide.fwr", dem)
