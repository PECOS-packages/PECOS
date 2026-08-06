# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Regression tests for the decoder surface defects from issue #431."""

from __future__ import annotations

import pytest
from pecos.decoders import (
    MWPM2D,
    BpLsdDecoder,
    BpOsdBuilder,
    BpOsdDecoder,
    DemAwareDecoder,
    DemAwareResult,
    DummyDecoder,
    FusionBlossomDecoder,
    MinSumBpDecoder,
    PyMatchingDecoder,
    RelayBpDecoder,
    SparseMatrix,
    TesseractDecoder,
    UnionFindBuilder,
    UnionFindDecoder,
)
from pecos_rslib.qec import decoder_dem_requirement

_DEM = """error(0.1) D0 D1 L0
error(0.1) D1 L0
detector D0
detector D1
logical_observable L0"""

_ENCODING_DEM = """detector D0
detector D1
logical_observable L0
error(0.1) D0
error(0.1) D1 L0
"""

_SYNDROMES = ([0, 0], [1, 0], [0, 1], [1, 1])
_OBSERVABLE_MASKS_BEFORE = [0, 0, 1, 1]
_FAMILY_DECODERS = [
    (BpOsdDecoder, "bp_osd"),
    (BpLsdDecoder, "bp_lsd"),
    (UnionFindDecoder, "union_find"),
    (RelayBpDecoder, "relay_bp"),
    (MinSumBpDecoder, "min_sum_bp"),
]


def test_dem_aware_result_is_importable() -> None:
    # Decoding returns this type, so users must be able to name it.
    assert DemAwareResult.__name__ == "DemAwareResult"


def test_fusion_blossom_builds_from_a_dem() -> None:
    assert FusionBlossomDecoder.from_dem(_DEM) is not None
    assert FusionBlossomDecoder.from_dem(_DEM, correlated=True) is not None


def test_dense_and_sparse_names_disambiguate_the_same_list() -> None:
    decoder = TesseractDecoder.from_dem(_ENCODING_DEM)
    dense_result = decoder.decode_syndrome([1, 0])
    sparse_result = decoder.decode_from_defects([1, 0])

    assert dense_result.observables_mask == 0
    assert dense_result.cost == pytest.approx(2.197224577336219)
    assert not dense_result.low_confidence
    assert sparse_result.observables_mask == 1
    assert sparse_result.cost == pytest.approx(4.394449154672438)
    assert not sparse_result.low_confidence


def test_renamed_methods_preserve_captured_results() -> None:
    pymatching = PyMatchingDecoder.from_dem(_ENCODING_DEM)
    pymatching_result = pymatching.decode_syndrome([1, 0])
    assert pymatching_result.correction == [0]
    assert pymatching_result.weight == pytest.approx(4.394449154672439)

    fusion_blossom = FusionBlossomDecoder.from_dem(_ENCODING_DEM)
    fusion_blossom_result = fusion_blossom.decode_syndrome([1, 0])
    assert fusion_blossom_result.correction == [0]
    assert fusion_blossom_result.weight == pytest.approx(2.196)

    parity_check_matrix = SparseMatrix([[1, 0], [0, 1]])
    bp_osd = BpOsdBuilder(parity_check_matrix, error_rate=0.1).build()
    bp_osd_result = bp_osd.decode_syndrome([1, 0])
    assert (bp_osd_result.decoding, bp_osd_result.converged, bp_osd_result.iterations) == ([1, 0], True, 1)

    union_find = UnionFindBuilder(parity_check_matrix).build()
    union_find_result = union_find.decode_syndrome([1, 0])
    assert (union_find_result.decoding, union_find_result.converged, union_find_result.iterations) == ([1, 0], True, 1)


def test_affected_decoder_classes_do_not_expose_bare_decode() -> None:
    parity_check_matrix = SparseMatrix([[1, 0], [0, 1]])
    decoders = [
        PyMatchingDecoder.from_dem(_ENCODING_DEM),
        TesseractDecoder.from_dem(_ENCODING_DEM),
        FusionBlossomDecoder.from_dem(_ENCODING_DEM),
        BpOsdBuilder(parity_check_matrix, error_rate=0.1).build(),
        UnionFindBuilder(parity_check_matrix).build(),
    ]

    for decoder in decoders:
        assert not hasattr(decoder, "decode")
        with pytest.raises(AttributeError, match=r"decode_syndrome.*decode_from_defects"):
            decoder.decode()


@pytest.mark.parametrize(("decoder_class", "decoder_type"), _FAMILY_DECODERS)
def test_family_from_dem_matches_existing_wrapper(decoder_class: type, decoder_type: str) -> None:
    named_decoder = decoder_class.from_dem(_ENCODING_DEM)
    existing_decoder = DemAwareDecoder.from_dem(_ENCODING_DEM, decoder_type=decoder_type)

    named_results = [named_decoder.decode_syndrome(list(syndrome)) for syndrome in _SYNDROMES]
    existing_results = [existing_decoder.decode_syndrome(list(syndrome)) for syndrome in _SYNDROMES]

    assert all(isinstance(result, DemAwareResult) for result in named_results)
    named_masks = [result.observables_mask for result in named_results]
    existing_masks = [result.observables_mask for result in existing_results]
    assert named_masks == existing_masks == _OBSERVABLE_MASKS_BEFORE
    assert isinstance(named_decoder, DemAwareDecoder)
    assert named_decoder.num_detectors == existing_decoder.num_detectors == 2
    assert named_decoder.num_mechanisms == existing_decoder.num_mechanisms == 2
    assert named_decoder.num_observables == existing_decoder.num_observables == 1
    assert f"type={decoder_type}" in repr(named_decoder)
    assert not hasattr(named_decoder, "decode")


@pytest.mark.parametrize(("decoder_class", "decoder_type"), _FAMILY_DECODERS)
def test_family_from_dem_forwards_configuration(decoder_class: type, decoder_type: str) -> None:
    named_decoder = decoder_class.from_dem(_ENCODING_DEM, error_rate=0.2, max_iter=0)
    existing_decoder = DemAwareDecoder.from_dem(
        _ENCODING_DEM,
        decoder_type=decoder_type,
        error_rate=0.2,
        max_iter=0,
    )

    named_result = named_decoder.decode_syndrome([0, 1])
    existing_result = existing_decoder.decode_syndrome([0, 1])
    assert (
        named_result.observables_mask,
        named_result.converged,
        named_result.iterations,
    ) == (
        existing_result.observables_mask,
        existing_result.converged,
        existing_result.iterations,
    )


@pytest.mark.parametrize("decoder_type", [decoder_type for _, decoder_type in _FAMILY_DECODERS])
def test_decoder_type_still_accepts_all_five_family_values(decoder_type: str) -> None:
    decoder = DemAwareDecoder.from_dem(_ENCODING_DEM, decoder_type=decoder_type)
    assert decoder.decode_syndrome([0, 1]).observables_mask == 1


def test_legacy_measurement_protocol_decoders_keep_decode() -> None:
    assert callable(MWPM2D.decode)
    assert callable(DummyDecoder.decode)


# Every name `create_observable_decoder` accepts must also classify here. Add to
# both places when adding a decoder; the two lists drifted apart before.
@pytest.mark.parametrize(
    ("decoder_type", "requirement"),
    [
        ("pymatching", "graphlike"),
        ("fusion_blossom", "graphlike"),
        ("k_mwpm", "graphlike"),
        ("windowed", "graphlike"),
        ("beamsearch", "graphlike"),
        ("belief_matching", "graphlike"),
        ("belief_matching_correlated", "graphlike"),
        ("belief_matching_mgbp", "graphlike"),
        ("belief_matching_hybrid:inner=pymatching", "graphlike"),
        ("tesseract", "any"),
        ("astar", "any"),
        ("bp_osd", "any"),
        ("bp_lsd", "any"),
        ("belief_find", "any"),
        ("union_find", "any"),
        ("relay_bp", "any"),
        ("min_sum_bp", "any"),
        ("mwpf", "any"),
        ("pecos_uf:bp", "graphlike"),
    ],
)
def test_registry_names_have_a_dem_requirement(decoder_type: str, requirement: str) -> None:
    assert decoder_dem_requirement(decoder_type) == requirement


@pytest.mark.parametrize(
    ("spec", "requirement"),
    [
        pytest.param("perturbed", "graphlike", id="default-inner-pymatching"),
        pytest.param("perturbed:K=5,inner=pymatching", "graphlike", id="matching-inner"),
        pytest.param("perturbed:K=5,inner=tesseract", "any", id="hyperedge-inner"),
        pytest.param("perturbed:K=5,inner=bp_osd", "any", id="check-matrix-inner"),
    ],
)
def test_perturbed_requirement_follows_its_inner_decoder(spec: str, requirement: str) -> None:
    # "perturbed" wraps an arbitrary inner decoder, so a fixed classification
    # would be wrong for half its uses.
    assert decoder_dem_requirement(spec) == requirement


def test_unknown_decoder_still_raises() -> None:
    with pytest.raises(ValueError, match="Unknown decoder type"):
        decoder_dem_requirement("not_a_decoder")
