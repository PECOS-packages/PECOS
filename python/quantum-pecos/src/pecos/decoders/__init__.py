"""Quantum error correction decoders.

This package provides various decoders for quantum error correction codes.
"""

# Copyright 2018 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract
# DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
# the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

# Rust decoders (from pecos_rslib)
from importlib import import_module

from pecos_rslib.decoders import (
    BpLsdBuilder,
    BpLsdDecoder,
    BpOsdBuilder,
    BpOsdDecoder,
    BpResult,
    CheckMatrix,
    DecoderSpec,
    DemAwareDecoder,
    DemAwareResult,
    FusionBlossomDecoder,
    MinSumBpBuilder,
    MinSumBpDecoder,
    MwpmResult,
    ObservableFlips,
    PyMatchingDecoder,
    RelayBpBuilder,
    RelayBpDecoder,
    SparseMatrix,
    TesseractDecoder,
    TesseractResult,
    TesseractTrellisDecoder,
    TesseractTrellisResult,
    UnionFindBuilder,
    UnionFindDecoder,
    astar,
    astar_full,
    beamsearch,
    belief_find,
    belief_matching,
    bp_lsd,
    bp_osd,
    ensemble,
    fusion_blossom,
    k_mwpm,
    min_sum_bp,
    mwpf,
    pecos_uf,
    perturbed,
    perturbed_fb_corr,
    pymatching,
    relay_bp,
    tesseract,
    tesseract_trellis,
    union_find,
    windowed,
)

from pecos.decoders.dummy_decoder.dummy_decoder import DummyDecoder
from pecos.decoders.mwpm2d.mwpm2d import MWPM2D

__all__ = [
    "MWPM2D",
    "BpLsdBuilder",
    "BpLsdDecoder",
    "BpOsdBuilder",
    "BpOsdDecoder",
    "BpResult",
    "CheckMatrix",
    "DecoderSpec",
    "DemAwareDecoder",
    "DemAwareResult",
    "DummyDecoder",
    "FusionBlossomDecoder",
    "MinSumBpBuilder",
    "MinSumBpDecoder",
    "MwpmResult",
    "ObservableFlips",
    "PyMatchingDecoder",
    "RelayBpBuilder",
    "RelayBpDecoder",
    "SparseMatrix",
    "TesseractDecoder",
    "TesseractResult",
    "TesseractTrellisDecoder",
    "TesseractTrellisResult",
    "UnionFindBuilder",
    "UnionFindDecoder",
    "astar",
    "astar_full",
    "beamsearch",
    "belief_find",
    "belief_matching",
    "bp_lsd",
    "bp_osd",
    "ensemble",
    "fusion_blossom",
    "k_mwpm",
    "min_sum_bp",
    "mwpf",
    "pecos_uf",
    "perturbed",
    "perturbed_fb_corr",
    "pymatching",
    "relay_bp",
    "tesseract",
    "tesseract_trellis",
    "union_find",
    "windowed",
]


def __getattr__(name: str) -> object:
    """Load experimental decoder factories only when explicitly requested."""
    if name in {"frontier", "bp_trellis"}:
        try:
            experimental = import_module("pecos_rslib_exp")
        except ModuleNotFoundError as exc:
            if exc.name != "pecos_rslib_exp":
                raise
            message = (
                f"{name} requires the optional pecos-rslib-exp package, which is not published to PyPI; "
                "build it from a PECOS source checkout with `just build`"
            )
            raise ImportError(message) from exc
        try:
            return getattr(experimental, name)
        except AttributeError as exc:
            message = (
                f"the installed pecos-rslib-exp package does not provide {name}; upgrade it to match quantum-pecos"
            )
            raise ImportError(message) from exc
    message = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(message)
