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

"""Decoder surface defects from issue #431.

Covers the public re-export, the DEM constructor that existed in Rust but was
unreachable from Python, and the drift between the decoder string registry and
the DEM-requirement query.
"""

from __future__ import annotations

import pytest
from pecos.decoders import DemAwareResult, FusionBlossomDecoder
from pecos_rslib.qec import decoder_dem_requirement

_DEM = "\n".join(
    [
        "error(0.1) D0 D1 L0",
        "error(0.1) D1 L0",
        "detector D0",
        "detector D1",
        "logical_observable L0",
    ],
)


def test_dem_aware_result_is_importable() -> None:
    # Decoding returns this type, so users must be able to name it.
    assert DemAwareResult.__name__ == "DemAwareResult"


def test_fusion_blossom_builds_from_a_dem() -> None:
    assert FusionBlossomDecoder.from_dem(_DEM) is not None
    assert FusionBlossomDecoder.from_dem(_DEM, correlated=True) is not None


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
