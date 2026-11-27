# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Caller input errors retain their category across decoder entry points."""

import pytest
from pecos_rslib import TickCircuit
from pecos_rslib.decoders import (
    DemAwareDecoder,
    FusionBlossomDecoder,
    PyMatchingDecoder,
    bp_osd,
    pymatching,
    tesseract,
    tesseract_trellis,
)
from pecos_rslib.qec import (
    DemSampler,
    DetectorErrorModel,
    LogicalAlgorithmDecoder,
    LogicalCircuitDecoder,
    LogicalSubgraphDecoder,
    SampleBatch,
    WindowedLogicalSubgraphDecoder,
)

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


@pytest.mark.parametrize("width", [1, 3])
def test_comparison_rejects_detector_count_mismatch(width):
    # DEM has two detectors. Without the run-level check every shot would be
    # counted as a decode error on both sides instead of raising.
    with pytest.raises(ValueError, match=f"SampleBatch has {width} detectors, but the DUT decoder model has 2"):
        SampleBatch([[0] * width], [0]).compare_decoders(DEM, "pecos_uf", "bp_osd")


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


@pytest.mark.parametrize("family", ["tesseract", "tesseract_trellis", "bp_osd", "mwpf"])
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
        message = "Invalid input dimensions" if x_length == 1 else f"x_num_detectors is {x_length}"
        with pytest.raises(ValueError, match=message):
            decoder.decode_count_batch([[0] * (x_length + z_length)], [0], x_length)
    else:
        with pytest.raises(ValueError, match="Invalid input dimensions"):
            decoder.decode_css(bytes(x_length), bytes(z_length))


def test_css_batch_does_not_ignore_unpaired_syndromes():
    from pecos_rslib.qec import CssUfDecoder

    decoder = CssUfDecoder("error(0.1) D0 L0", "error(0.1) D0 L0")
    with pytest.raises(ValueError, match="true_obs_masks has 1 entries, but syndromes has 2"):
        decoder.decode_count_batch([[0, 0], [0]], [0], 1)


def test_css_empty_batch_still_validates_split():
    from pecos_rslib.qec import CssUfDecoder

    decoder = CssUfDecoder("error(0.1) D0 L0", "error(0.1) D0 L0")
    with pytest.raises(ValueError, match="x_num_detectors is 0, but the decoder's X DEM has 1 detectors"):
        decoder.decode_count_batch([], [], 0)


@pytest.mark.parametrize(("x_length", "z_length"), [(0, 2), (2, 2), (1, 1), (1, 3)])
def test_css_count_erasures_wrong_lengths(x_length, z_length):
    from pecos_rslib.qec import CssUfDecoder

    decoder = CssUfDecoder("error(0.1) D0 L0", "error(0.1) D0 L0\ndetector D1")
    with pytest.raises(ValueError, match="Invalid input dimensions"):
        decoder.count_erasures(bytes(x_length), bytes(z_length))
    assert decoder.count_erasures(bytes(1), bytes(2)) == 0


# A two-detector model whose logical subgraph holds only D0, so a zero-padded short
# syndrome or a truncated long one would decode without error.
_LOGICAL_DEM = """error(0.1) D0 L0
error(0.1) D0 D1
error(0.1) D1
detector(1, 0, 0) D0
detector(1, 0, 1) D1
"""
_LOGICAL_STAB_COORDS = [{"X": [(1.0, 0.0)], "Z": []}]


def _logical_descriptor():
    segment = {"num_detectors": 2, "dem": _LOGICAL_DEM, "stab_coords": _LOGICAL_STAB_COORDS}
    return {
        "full_dem": _LOGICAL_DEM,
        "segments": [segment],
        "num_observables": 1,
        "num_frame_slots": 2,
        "boundary_gates": [],
    }


def _logical_decoders():
    return [
        LogicalSubgraphDecoder(_LOGICAL_DEM, _LOGICAL_STAB_COORDS, "pymatching"),
        WindowedLogicalSubgraphDecoder(_LOGICAL_DEM, _LOGICAL_STAB_COORDS, 1, 1),
        LogicalAlgorithmDecoder(_logical_descriptor(), "pymatching"),
        LogicalCircuitDecoder(_logical_descriptor(), "unlimited", "pymatching"),
        LogicalCircuitDecoder(_logical_descriptor(), "windowed", "pymatching"),
    ]


@pytest.mark.parametrize("length", [1, 3])
def test_logical_decoders_reject_wrong_length(length):
    for decoder in _logical_decoders():
        assert decoder.decode(bytes(2)) == 0
        with pytest.raises(ValueError, match=f"expected 2, got {length}"):
            decoder.decode(bytes(length))


@pytest.mark.parametrize("length", [1, 3])
def test_logical_decode_count_rejects_wrong_length(length):
    batch = SampleBatch([[0] * length], [0])
    for decoder in _logical_decoders():
        with pytest.raises(ValueError, match=f"expected 2, got {length}"):
            decoder.decode_count(batch)


def test_logical_algorithm_short_syndrome_does_not_reuse_previous_shot():
    decoder = LogicalAlgorithmDecoder(_logical_descriptor(), "pymatching")
    decoder.feed_dense([1, 0])
    with pytest.raises(ValueError, match="expected 2, got 1"):
        decoder.feed_dense([0])
    with pytest.raises(ValueError, match="node index 2"):
        decoder.feed_sparse([(0, 0), (2, 1)])
    # Neither rejected call touched the buffer fed by the first shot.
    assert decoder.flush() == 1
    assert decoder.rounds_fed() == 0


def test_logical_algorithm_rejects_descriptor_decoder_width_mismatch():
    descriptor = _logical_descriptor()
    descriptor["segments"][0]["num_detectors"] = 3
    with pytest.raises(ValueError, match="segments total 3 detectors, but the full-circuit decoder expects 2"):
        LogicalAlgorithmDecoder(descriptor, "pymatching")


@pytest.mark.parametrize("budget", ["unlimited", "windowed"])
@pytest.mark.parametrize("segment_detectors", [1, 3])
def test_logical_circuit_rejects_descriptor_strategy_width_mismatch(budget, segment_detectors):
    # Below the DEM width, the windowed strategy used to zero-pad the missing
    # tail instead of rejecting the descriptor.
    descriptor = _logical_descriptor()
    descriptor["segments"][0]["num_detectors"] = segment_detectors
    with pytest.raises(
        ValueError,
        match=f"segments total {segment_detectors} detectors, but the decode strategy expects 2",
    ):
        LogicalCircuitDecoder(descriptor, budget, "pymatching")


# D1 is used by a mechanism but sits at coordinates no stabilizer claims, so
# partitioning this model fails before any inner decoder could be built.
_UNPARTITIONABLE_DEM = "error(0.1) D0 L0\nerror(0.1) D0 D1\ndetector(1, 0, 0) D0\ndetector(9, 9, 0) D1\n"


def test_logical_constructor_rejects_bad_inner_spec_before_building():
    with pytest.raises(ValueError, match="could not be classified against stab_coords"):
        LogicalSubgraphDecoder(_UNPARTITIONABLE_DEM, _LOGICAL_STAB_COORDS, "pymatching")
    # The spec is parsed before the model is partitioned, so its error wins.
    with pytest.raises(ValueError, match="Unsupported decoder_type: unknown_decoder"):
        LogicalSubgraphDecoder(_UNPARTITIONABLE_DEM, _LOGICAL_STAB_COORDS, "unknown_decoder")


# The p = 0.7 mechanism's matching weight is negative. Logical-subgraph extraction checks the
# caller's prior before the sub-DEM merges it with the projected D0 D1 mechanism.
_HIGH_PRIOR_DEM = _LOGICAL_DEM.replace("error(0.1) D0 L0", "error(0.7) D0 L0")
_MECHANISM_PRIOR = r"DEM error probability 0\.7 is above 0\.5, which gives a negative matching weight"
_EDGE_PRIOR = r"DEM edge probability 0\.7 is above 0\.5"
_CORRELATED_PM_PRIOR = r"DEM error probability 0\.7 is above 0\.5, which correlated PyMatching"


@pytest.mark.parametrize(
    ("family", "message"),
    [
        ("fusion_blossom", _MECHANISM_PRIOR),
        ("k_mwpm", _MECHANISM_PRIOR),
        ("perturbed_fb_corr", _EDGE_PRIOR),
        ("pymatching", _CORRELATED_PM_PRIOR),
        ("perturbed", _CORRELATED_PM_PRIOR),
        ("belief_matching", _MECHANISM_PRIOR),
        ("belief_matching_correlated", _MECHANISM_PRIOR),
        ("belief_matching_mgbp", _MECHANISM_PRIOR),
        pytest.param(f"belief_matching_hybrid:{_HIGH_PRIOR_DEM}", _MECHANISM_PRIOR, id="belief_matching_hybrid"),
    ],
)
def test_logical_subdem_prior_above_half_is_input_error(family, message):
    from pecos_rslib.decoders import DecoderSpec

    with pytest.raises(ValueError, match=_MECHANISM_PRIOR):
        LogicalSubgraphDecoder(_HIGH_PRIOR_DEM, _LOGICAL_STAB_COORDS, family)
    # Building the same backend directly classifies the prior the same way.
    with pytest.raises(ValueError, match=message):
        SampleBatch([[0, 0]], [0]).decode(_HIGH_PRIOR_DEM, DecoderSpec.parse(family), workers=1)


# Each prior is above 0.5, though their independent combination, 0.42, is not.
_PARALLEL_HIGH_PRIOR_DEM = _LOGICAL_DEM.replace("error(0.1) D0 D1", "error(0.7) D0 D1\nerror(0.7) D0 D1")


@pytest.mark.parametrize("family", ["fusion_blossom", "belief_matching"])
def test_parallel_priors_above_half_are_input_errors(family):
    from pecos_rslib.decoders import DecoderSpec

    with pytest.raises(ValueError, match=_MECHANISM_PRIOR):
        SampleBatch([[0, 0]], [0]).decode(_PARALLEL_HIGH_PRIOR_DEM, DecoderSpec.parse(family), workers=1)
    with pytest.raises(ValueError, match=_MECHANISM_PRIOR):
        LogicalSubgraphDecoder(_PARALLEL_HIGH_PRIOR_DEM, _LOGICAL_STAB_COORDS, family)


def test_windowed_union_find_rejects_parallel_priors_above_half():
    from pecos_rslib.decoders import DecoderSpec

    # Commit windows build their matching graphs from the raw columns, so they check
    # each mechanism's prior themselves.
    spec = DecoderSpec.parse("windowed:step=1,buf=1,inner=pecos_uf")
    with pytest.raises(ValueError, match=_MECHANISM_PRIOR):
        SampleBatch([[0, 0]], [0]).decode(_PARALLEL_HIGH_PRIOR_DEM, spec, workers=1)


# A certain mechanism gives a zero weight that the BP stage accepts, so only the matching
# prior check rejects it.
_CERTAIN_PRIOR_DEM = _LOGICAL_DEM.replace("error(0.1) D0 L0", "error(1) D0 L0")


@pytest.mark.parametrize(
    "family",
    [
        "belief_matching",
        "belief_matching_correlated",
        "belief_matching_mgbp",
        pytest.param(f"belief_matching_hybrid:{_CERTAIN_PRIOR_DEM}", id="belief_matching_hybrid"),
    ],
)
def test_belief_matching_certain_prior_is_input_error(family):
    from pecos_rslib.decoders import DecoderSpec

    with pytest.raises(ValueError, match=r"DEM error probability 1 is above 0\.5"):
        SampleBatch([[0, 0]], [0]).decode(_CERTAIN_PRIOR_DEM, DecoderSpec.parse(family), workers=1)


def test_raw_pymatching_invalid_error_probability():
    with pytest.raises(ValueError, match="error_probability"):
        PyMatchingDecoder.from_dem(DEM, error_probability=1.1)


@pytest.mark.parametrize("length", [1, 3])
@pytest.mark.parametrize(
    ("decoder_class", "message"),
    [
        (PyMatchingDecoder, "Invalid syndrome: expected length 2, got {length}$"),
        (FusionBlossomDecoder, "Invalid input dimensions: expected 2, got {length}$"),
    ],
)
def test_raw_matching_wrong_length(decoder_class, message, length):
    decoder = decoder_class.from_dem(DEM)
    decoder.decode_syndrome([1, 0])
    with pytest.raises(ValueError, match=message.format(length=length)):
        decoder.decode_syndrome([0] * length)


# PyMatching rejects a decomposed error with an undetectable component while it
# builds the correlated graph from DEM text.
_UNDETECTABLE_COMPONENT_DEM = "error(0.1) D0 ^ L0\ndetector D0"


@pytest.mark.parametrize("workers", [1, 2])
def test_correlated_pymatching_dem_rejection_is_input_error(workers):
    from pecos_rslib.decoders import DecoderSpec

    with pytest.raises(ValueError, match="undetectable component"):
        SampleBatch([[0]], [0]).decode(
            _UNDETECTABLE_COMPONENT_DEM, DecoderSpec.parse("pymatching_correlated"), workers=workers
        )


def test_raw_correlated_pymatching_dem_rejection_is_input_error():
    with pytest.raises(ValueError, match="undetectable component"):
        PyMatchingDecoder.from_dem_with_correlations(_UNDETECTABLE_COMPONENT_DEM)


# Every caller mechanism's prior is checked before projection, so the membership
# cannot hide a prior above 0.5.
@pytest.mark.parametrize(
    ("dem", "membership"),
    [
        pytest.param("error(0.1) D0 L0\nerror(0.9) D1", [[0]], id="projected-away"),
        pytest.param("error(0.1) D0 L0\nerror(0.9) D1", [[]], id="empty-membership"),
        pytest.param("error(0.1) D0 L0\nerror(0.9) D1 D2 D3", [[0, 1, 2, 3]], id="three-detectors"),
    ],
)
def test_logical_membership_does_not_hide_a_prior_above_half(dem, membership):
    with pytest.raises(ValueError, match=r"DEM error probability 0\.9 is above 0\.5"):
        LogicalSubgraphDecoder.from_membership(dem, membership, "pymatching")
