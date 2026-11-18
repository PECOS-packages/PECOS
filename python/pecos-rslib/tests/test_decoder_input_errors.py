# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Caller input errors retain their category across decoder entry points."""

import pytest
from pecos_rslib import TickCircuit
from pecos_rslib.decoders import DemAwareDecoder, bp_osd
from pecos_rslib.qec import DemSampler, DetectorErrorModel, LogicalSubgraphDecoder, SampleBatch

DEM = "error(0.1) D0 L0\ndetector D1"


@pytest.mark.parametrize(
    ("spec", "message"),
    [(bp_osd(max_iter=2**31), "max_iter"), (bp_osd(error_rate=0.0), "no uncertain error mechanism")],
)
def test_spec_build_input_error(spec, message):
    with pytest.raises(ValueError, match=message):
        SampleBatch([[0, 0]], [0]).decode(DEM, spec, workers=1)


@pytest.mark.parametrize("workers", [1, 2])
def test_batch_dead_detector(workers):
    with pytest.raises(ValueError, match="shot 0:.*detector 1 fired"):
        SampleBatch([[0, 1]], [0]).decode(DEM, bp_osd(), workers=workers)


@pytest.mark.parametrize("workers", [1, 2])
def test_sampler_dead_detector(workers):
    sampler = DemSampler.from_dem_string("error(1) D1\nerror(0.1) D0 L0")
    with pytest.raises(ValueError, match="shot 0:.*detector 1 fired"):
        sampler.decode(DEM, 4, bp_osd(), seed=1, workers=workers)


def test_logical_dead_detector():
    decoder = LogicalSubgraphDecoder.from_membership("error(0.1) D1 L0", [[0, 1]], "bp_osd")
    with pytest.raises(ValueError, match="detector 0 fired"):
        decoder.decode(bytes([1, 0]))


def test_structured_wrong_length():
    circuit = TickCircuit()
    circuit.tick().pz([0])
    circuit.tick().mz([0])
    circuit.add_detector([-1], detector_id=2)
    dem = DetectorErrorModel.from_circuit(circuit, p1=0, p2=0, p_meas=0.1, p_prep=0)
    decoder = dem.build_decoder(bp_osd())
    with pytest.raises(ValueError, match="expected 3, got 1"):
        decoder.decode_syndrome([0])


@pytest.mark.parametrize("rate", [0.1, 0.0, 1.0, float("nan"), float("inf")])
def test_union_find_rejects_prior(rate):
    with pytest.raises(ValueError, match="error_rate.*no priors"):
        DemAwareDecoder.from_dem(DEM, "union_find", error_rate=rate)


def test_logical_decode_count_dead_detector():
    decoder = LogicalSubgraphDecoder.from_membership("error(0.1) D1 L0", [[0, 1]], "bp_osd")
    with pytest.raises(ValueError, match="detector 0 fired"):
        decoder.decode_count(SampleBatch([[1, 0]], [0]))


def test_logical_constructor_preserves_inner_parse_error():
    with pytest.raises(ValueError, match="Unsupported decoder_type"):
        LogicalSubgraphDecoder.from_membership(DEM, [[0]], "unknown_decoder")


def test_comparison_build_input_error():
    with pytest.raises(ValueError, match="no uncertain error mechanism"):
        SampleBatch([[0]], [0]).compare_decoders("error(0) D0 L0", "bp_osd", "bp_osd")


def test_comparison_counts_dead_detector_errors():
    result = SampleBatch([[0, 1]], [0]).compare_decoders(DEM, "bp_osd", "bp_osd")
    assert result.dut_error_reference_error == 1
