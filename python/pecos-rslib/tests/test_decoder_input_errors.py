# Copyright 2026 The PECOS Developers
# Licensed under the Apache License, Version 2.0

"""Caller input errors retain their category across decoder entry points."""

import sys

import pytest
from pecos_rslib import TickCircuit
from pecos_rslib.decoders import (
    CheckMatrix,
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

pytest_plugins = ("pytester",)

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


# PyMatching classifies upstream failures by phase: anything that fails while it
# parses DEM text or builds the graph or matcher is a rejection of the graph
# (`ValueError`); a failure while decoding a syndrome stays `RuntimeError`.
@pytest.mark.parametrize(
    "dem",
    [
        pytest.param("repeat 2 {\nerror(0.1) D0", id="unterminated-repeat"),
        pytest.param("error(0.1) D0\n}", id="uninitiated-repeat"),
        # The largest index the grammar accepts; its node vector exceeds the
        # allocation size limit at once (`length_error`). A smaller index such as
        # D4294967295 asks for about 137 GB, which may succeed or be OOM-killed
        # depending on the machine.
        pytest.param("error(0.1) D0 D1152921504606846975", id="huge-detector"),
    ],
)
def test_pymatching_graph_phase_rejection_is_input_error(dem):
    with pytest.raises(ValueError, match="Failed to parse DEM"):
        PyMatchingDecoder.from_dem(dem)


def test_pymatching_weight_rejection_happens_when_the_matcher_is_built():
    # The graph builds; upstream rejects the infinite weight when the matcher is
    # first built for decoding, which is still a rejection of the graph.
    with pytest.raises(ValueError, match="decoder failed on shot 0:.*maximum absolute edge weight"):
        SampleBatch([[1]], [0]).decode("error(1) D0", pymatching(correlated=False), workers=1)
    decoder = PyMatchingDecoder.from_dem("error(1) D0")
    with pytest.raises(ValueError, match="Failed to build the matcher: maximum absolute edge weight"):
        decoder.decode_syndrome([1])


def test_pymatching_error_probability_override_replaces_an_out_of_range_dem_weight():
    decoder = PyMatchingDecoder.from_dem("error(1) D0", error_probability=0.1)
    decoder.decode_syndrome([1])
    SampleBatch([[1]], [0]).decode("error(1) D0", pymatching(correlated=False, error_probability=0.1), workers=1)


def test_pymatching_graph_size_rejection_is_input_error():
    # 2**62 nodes and node 2**63 fail the allocation size check at once.
    with pytest.raises(ValueError, match="Failed to create the graph"):
        PyMatchingDecoder(2**62)
    decoder = PyMatchingDecoder(2)
    with pytest.raises(ValueError, match="Failed to add edge"):
        decoder.add_edge(2**63, 1, [])


def test_pymatching_matcher_build_rejection_is_input_error():
    # add_edge stores any finite weight; upstream rejects it when it builds the
    # matcher for the first decode.
    decoder = PyMatchingDecoder(2)
    decoder.add_edge(0, 1, [0], weight=1e12)
    with pytest.raises(ValueError, match="Failed to build the matcher: maximum absolute edge weight"):
        decoder.decode_syndrome([1, 1])


def test_pymatching_decode_phase_failure_stays_runtime_error():
    # An odd-parity syndrome with no boundary has no perfect matching.
    decoder = PyMatchingDecoder.from_dem("error(0.1) D0 D1")
    with pytest.raises(RuntimeError, match="No perfect matching"):
        decoder.decode_syndrome([1, 0])


def _value_error_in_subprocess(pytester, call, setup=""):
    """Run `call` in a fresh interpreter, since before #1771 it ended the process."""
    script = (
        "from pecos_rslib.decoders import PyMatchingDecoder\n"
        f"{setup}"
        "try:\n"
        f"    {call}\n"
        "except ValueError as error:\n"
        "    print('ValueError:', error)\n"
        "else:\n"
        "    print('no error')\n"
    )
    result = pytester.run(sys.executable, "-c", script, timeout=60)
    assert result.ret == 0, result.stderr.str()
    return result.stdout.str()


# Upstream sizes its node list as `node + 1`, which wraps for 2**64 - 1, and then
# indexes past the end.
@pytest.mark.parametrize(
    "call",
    [
        "PyMatchingDecoder(4).add_edge(2**64 - 1, 1, [0], 1.0)",
        "PyMatchingDecoder(4).add_edge(0, 2**64 - 1, [0], 1.0)",
        "PyMatchingDecoder(4).add_boundary_edge(2**64 - 1, [0], 1.0)",
    ],
)
def test_pymatching_boundary_node_marker_is_not_an_edge_node(pytester, call):
    output = _value_error_in_subprocess(pytester, call)
    assert "ValueError: node 18446744073709551615 is out of range: BOUNDARY_NODE_MARKER" in output


@pytest.mark.parametrize("add", ["add_edge", "add_boundary_edge"])
def test_pymatching_observable_usize_max_is_rejected_before_the_edge_is_added(add):
    decoder = PyMatchingDecoder(4)
    nodes = (0, 1) if add == "add_edge" else (0,)
    with pytest.raises(ValueError, match="observable 18446744073709551615 is out of range"):
        getattr(decoder, add)(*nodes, [2**64 - 1], 1.0)
    assert decoder.num_edges == 0


# Observable 2**63 gives an observable count no buffer can hold, on any machine
# and with any overcommit setting; the output is sized by the graph, so it is a
# rejection of the graph. Before #1771 decode_batch aborted the process.
@pytest.mark.parametrize("decode", ["decode_syndrome([1, 1, 0, 0])", "decode_batch([1, 1, 0, 0], num_shots=1)"])
def test_pymatching_unallocatable_observable_output_is_an_input_error(pytester, decode):
    output = _value_error_in_subprocess(
        pytester, f"d = PyMatchingDecoder(4); d.add_edge(0, 1, [2**63], 1.0); d.{decode}"
    )
    assert "ValueError: cannot allocate one row of" in output
    assert "sized by the graph's observable count" in output


def test_pymatching_batch_too_large_for_num_shots_is_an_input_error(pytester):
    # No detectors, so the empty shot buffer matches any num_shots.
    output = _value_error_in_subprocess(pytester, "PyMatchingDecoder(0).decode_batch([], num_shots=2**62)")
    assert "ValueError: Configuration error: cannot allocate" in output
    assert "pass a smaller num_shots" in output


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="limits the address space with RLIMIT_AS")
def test_pymatching_batch_rows_too_many_for_memory_are_an_input_error(pytester):
    # The flat output fits under the limit, but splitting it into one row per
    # shot does not; every row allocation must fail as an error, not abort.
    setup = (
        "import os, resource\n"
        "from pathlib import Path\n"
        "used = int(Path('/proc/self/statm').read_text().split()[0]) * os.sysconf('SC_PAGE_SIZE')\n"
        "limit = used + 600 * 1024 * 1024\n"
        "resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n"
        "resource.setrlimit(resource.RLIMIT_AS, (limit, limit))\n"
    )
    output = _value_error_in_subprocess(
        pytester, "PyMatchingDecoder(0).decode_batch([], num_shots=4_000_000)", setup=setup
    )
    assert (
        "ValueError: Configuration error: cannot allocate the contents of 4000000 output rows; pass a smaller num_shots"
        in output
    )


@pytest.mark.parametrize(
    ("repetitions", "message"),
    [
        (0, "repetitions must be at least 1, got 0"),
        (2**63, f"2 rows times {2**63} repetitions overflows usize; pass fewer repetitions"),
    ],
)
def test_pymatching_check_matrix_repetitions_are_input_errors(repetitions, message):
    matrix = CheckMatrix.from_dense([[1, 1, 0], [0, 1, 1]])
    with pytest.raises(ValueError, match=message):
        PyMatchingDecoder.from_check_matrix_with_repetitions(
            matrix, repetitions=repetitions, use_virtual_boundary=False
        )


def test_pymatching_unallocatable_node_names_the_node():
    decoder = PyMatchingDecoder(4)
    with pytest.raises(
        ValueError,
        match=(
            f"Failed to add edge: node {2**64 - 2} needs a graph of {2**64 - 1} nodes, " "which cannot be allocated"
        ),
    ):
        decoder.add_edge(2**64 - 2, 1, [0], 1.0)
    with pytest.raises(ValueError, match=f"Failed to add boundary edge: node {2**64 - 2} needs a graph"):
        decoder.add_boundary_edge(2**64 - 2, [0], 1.0)
    assert (decoder.num_nodes, decoder.num_edges) == (4, 0)
