# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Caller input errors retain their category across decoder entry points."""

import pytest
from pecos_rslib import TickCircuit
from pecos_rslib.decoders import DemAwareDecoder, bp_osd, pymatching, tesseract, tesseract_trellis
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


@pytest.mark.parametrize(
    ("spec", "length"),
    [
        pytest.param(pymatching(correlated=False), 1, id="pymatching-short"),
        pytest.param(pymatching(correlated=False), 4, id="pymatching-long"),
        pytest.param(tesseract(), 1, id="tesseract-short"),
        pytest.param(tesseract(), 4, id="tesseract-long"),
        pytest.param(tesseract_trellis(), 1, id="tesseract-trellis-short"),
        pytest.param(tesseract_trellis(), 4, id="tesseract-trellis-long"),
    ],
)
def test_structured_backend_wrong_length(spec, length):
    circuit = TickCircuit()
    circuit.tick().pz([0])
    circuit.tick().mz([0])
    circuit.add_detector([-1], detector_id=2)
    dem = DetectorErrorModel.from_circuit(circuit, p1=0, p2=0, p_meas=0.1, p_prep=0)
    decoder = dem.build_decoder(spec)
    with pytest.raises(ValueError, match=f"expected 3, got {length}"):
        decoder.decode_syndrome([0] * length)


def test_spec_tesseract_rejects_wide_observables():
    with pytest.raises(ValueError, match="at most 64"):
        SampleBatch([[0]], [0]).decode("error(0.1) D0 L64", tesseract(), workers=1)


@pytest.mark.parametrize("length", [1, 4])
@pytest.mark.parametrize(
    "family",
    [
        "pymatching",
        "tesseract",
        "tesseract_trellis",
        "k_mwpm",
        "astar",
        "astar_full",
        "fusion_blossom",
        "perturbed_fb_corr",
        "bp_osd",
        "bp_lsd",
        "belief_find",
        "union_find",
        "relay_bp",
        "min_sum_bp",
        "pecos_uf",
        "belief_matching",
        "windowed:step=5,buf=5,inner=pecos_uf",
        "mwpf",
        "perturbed",
        "beamsearch",
        "ensemble:pecos_uf,bp_osd",
    ],
)
def test_all_spec_families_reject_wrong_length(family, length):
    from pecos_rslib.decoders import DecoderSpec

    circuit = TickCircuit()
    circuit.tick().pz([0, 1, 2])
    circuit.tick().mz([0, 1, 2])
    for detector in range(3):
        circuit.add_detector([detector - 3], coords=[detector, 0, 0], detector_id=detector, fault_sector="Z")
    dem = DetectorErrorModel.from_circuit(circuit, p1=0, p2=0, p_meas=0.1, p_prep=0)
    decoder = dem.build_decoder(DecoderSpec.parse(family))
    with pytest.raises(ValueError, match=f"expected 3, got {length}"):
        decoder.decode_syndrome([0] * length)


@pytest.mark.parametrize("family", ["tesseract", "tesseract_trellis", "bp_osd"])
def test_logical_subdem_preserves_unconnected_detector(family):
    decoder = LogicalSubgraphDecoder.from_membership("error(0.1) D1 L0", [[1, 0]], family)
    assert decoder.decode(bytes([0, 0])) == 0
    assert decoder.decode(bytes([0, 1])) == 1


@pytest.mark.parametrize(("x_length", "z_length"), [(0, 2), (2, 2), (1, 1), (1, 3)])
@pytest.mark.parametrize("batch", [False, True])
def test_css_wrong_lengths(x_length, z_length, batch):
    from pecos_rslib.qec import CssUfDecoder

    decoder = CssUfDecoder("error(0.1) D0 L0", "error(0.1) D0 L0\ndetector D1")
    if batch:
        with pytest.raises(ValueError, match="Invalid input dimensions"):
            decoder.decode_count_batch([[0] * (x_length + z_length)], [0], x_length)
    else:
        with pytest.raises(ValueError, match="Invalid input dimensions"):
            decoder.decode_css(bytes(x_length), bytes(z_length))


def test_css_batch_does_not_ignore_unpaired_syndromes():
    from pecos_rslib.qec import CssUfDecoder

    decoder = CssUfDecoder("error(0.1) D0 L0", "error(0.1) D0 L0")
    with pytest.raises(ValueError, match="Batch size mismatch"):
        decoder.decode_count_batch([[0, 0], [0]], [0], 1)


def test_css_empty_batch_still_validates_split():
    from pecos_rslib.qec import CssUfDecoder

    decoder = CssUfDecoder("error(0.1) D0 L0", "error(0.1) D0 L0")
    with pytest.raises(ValueError, match="expected 1, got 0"):
        decoder.decode_count_batch([], [], 0)
