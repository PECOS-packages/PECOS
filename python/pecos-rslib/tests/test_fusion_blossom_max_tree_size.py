# Copyright 2026 The PECOS Developers

"""Fusion Blossom max-tree-size Python surface tests."""

import inspect

from pecos_rslib.decoders import FusionBlossomDecoder


def test_manual_constructor_accepts_max_tree_size() -> None:
    """The manual constructor exposes an optional keyword-only limit."""
    signature = inspect.signature(FusionBlossomDecoder)

    assert "max_tree_size" in signature.parameters
    assert signature.parameters["max_tree_size"].default is None
    FusionBlossomDecoder(4)
    FusionBlossomDecoder(4, max_tree_size=1)


def test_check_matrix_constructor_accepts_max_tree_size() -> None:
    """The check-matrix constructor exposes an optional keyword-only limit."""
    assert hasattr(FusionBlossomDecoder, "from_check_matrix")
    signature = inspect.signature(FusionBlossomDecoder.from_check_matrix)

    assert "max_tree_size" in signature.parameters
    assert signature.parameters["max_tree_size"].default is None
    check_matrix = [[1, 1, 0], [0, 1, 1]]
    FusionBlossomDecoder.from_check_matrix(check_matrix)
    FusionBlossomDecoder.from_check_matrix(check_matrix, max_tree_size=1)
