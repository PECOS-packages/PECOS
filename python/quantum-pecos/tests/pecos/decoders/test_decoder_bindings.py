# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License.You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

# ruff: noqa: N806
# H is standard notation for parity check matrix in coding theory

"""Tests for PECOS decoder bindings exposed via pecos_rslib.decoders.

The API is designed to mirror the original library APIs:
- PyMatching: pymatching.Matching
- Fusion Blossom: fusion_blossom
- LDPC: ldpc.bposd_decoder, ldpc.bplsd_decoder
"""

import pytest


class TestMwpmResult:
    """Tests for unified MwpmResult type."""

    def test_result_attributes(self) -> None:
        """Test that MwpmResult has the expected attributes."""
        from pecos_rslib.decoders import CheckMatrix, PyMatchingDecoder

        matrix = CheckMatrix.from_dense([[1, 1, 0], [0, 1, 1]])
        decoder = PyMatchingDecoder.from_check_matrix(matrix)
        result = decoder.decode([0, 0])

        # Check attributes exist
        assert hasattr(result, "correction")
        assert hasattr(result, "weight")

        # Check types
        assert isinstance(result.correction, list)
        assert isinstance(result.weight, float)

    def test_result_to_list(self) -> None:
        """Test MwpmResult.to_list() method."""
        from pecos_rslib.decoders import CheckMatrix, PyMatchingDecoder

        matrix = CheckMatrix.from_dense([[1, 1, 0], [0, 1, 1]])
        decoder = PyMatchingDecoder.from_check_matrix(matrix)
        result = decoder.decode([0, 0])

        assert result.to_list() == result.correction

    def test_result_indexing(self) -> None:
        """Test MwpmResult supports indexing like a list."""
        from pecos_rslib.decoders import CheckMatrix, PyMatchingDecoder

        matrix = CheckMatrix.from_dense([[1, 1, 0], [0, 1, 1]])
        decoder = PyMatchingDecoder.from_check_matrix(matrix)
        result = decoder.decode([0, 0])

        assert len(result) == len(result.correction)
        if len(result) > 0:
            assert result[0] == result.correction[0]


class TestCheckMatrix:
    """Tests for CheckMatrix (used by PyMatching)."""

    def test_create_from_dense(self) -> None:
        """Test creating from dense matrix (like PyMatching's Matching(H))."""
        from pecos_rslib.decoders import CheckMatrix

        # Repetition code check matrix
        H = [[1, 1, 0], [0, 1, 1]]
        matrix = CheckMatrix.from_dense(H)

        assert matrix.rows == 2
        assert matrix.cols == 3
        assert matrix.nnz() == 4

    def test_create_from_coo(self) -> None:
        """Test creating from COO format."""
        from pecos_rslib.decoders import CheckMatrix

        matrix = CheckMatrix(
            rows=2,
            cols=3,
            row_indices=[0, 0, 1, 1],
            col_indices=[0, 1, 1, 2],
        )

        assert matrix.rows == 2
        assert matrix.cols == 3

    def test_with_weights(self) -> None:
        """Test adding weights (like PyMatching's weights parameter)."""
        from pecos_rslib.decoders import CheckMatrix

        matrix = CheckMatrix.from_dense([[1, 1, 0], [0, 1, 1]])
        weighted = matrix.with_weights([1.0, 2.0, 1.0])

        weights = weighted.weights()
        assert weights is not None
        assert len(weights) == 3
        assert weights[1] == 2.0


class TestPyMatchingDecoder:
    """Tests for PyMatchingDecoder (mirrors pymatching.Matching)."""

    def test_from_check_matrix(self) -> None:
        """Test construction from check matrix (like Matching(H))."""
        from pecos_rslib.decoders import CheckMatrix, PyMatchingDecoder

        H = CheckMatrix.from_dense([[1, 1, 0], [0, 1, 1]])
        decoder = PyMatchingDecoder.from_check_matrix(H)

        assert decoder.num_detectors == 2

    def test_decode_trivial(self) -> None:
        """Test decoding trivial (all-zero) syndrome."""
        from pecos_rslib.decoders import CheckMatrix, PyMatchingDecoder

        H = CheckMatrix.from_dense([[1, 1, 0], [0, 1, 1]])
        decoder = PyMatchingDecoder.from_check_matrix(H)

        result = decoder.decode([0, 0])

        # No errors - should have zero weight
        assert result.weight == 0.0

    def test_decode_single_error(self) -> None:
        """Test decoding syndrome from single error."""
        from pecos_rslib.decoders import CheckMatrix, PyMatchingDecoder

        # Repetition code: error on qubit 1 gives syndrome [1, 1]
        H = CheckMatrix.from_dense([[1, 1, 0], [0, 1, 1]])
        decoder = PyMatchingDecoder.from_check_matrix(H)

        result = decoder.decode([1, 1])
        assert result is not None
        assert result.weight > 0

    def test_manual_graph_construction(self) -> None:
        """Test building graph manually (like Matching.add_edge)."""
        from pecos_rslib.decoders import PyMatchingDecoder

        decoder = PyMatchingDecoder(num_nodes=3, num_observables=1)
        decoder.add_edge(0, 1, observables=[0], weight=1.0)
        decoder.add_edge(1, 2, observables=[], weight=1.0)
        decoder.add_boundary_edge(0, observables=[0], weight=1.0)
        decoder.add_boundary_edge(2, observables=[], weight=1.0)

        assert decoder.num_nodes >= 3
        assert decoder.num_edges >= 4


class TestFusionBlossomDecoder:
    """Tests for FusionBlossomDecoder (mirrors fusion_blossom)."""

    def test_create_decoder(self) -> None:
        """Test basic construction."""
        from pecos_rslib.decoders import FusionBlossomDecoder

        decoder = FusionBlossomDecoder(num_nodes=4, num_observables=1)
        assert decoder.num_nodes == 4

    def test_from_check_matrix(self) -> None:
        """Test construction from check matrix."""
        from pecos_rslib.decoders import FusionBlossomDecoder

        H = [[1, 1, 0], [0, 1, 1]]
        decoder = FusionBlossomDecoder.from_check_matrix(H)

        assert decoder.num_nodes == 2

    def test_decode_trivial(self) -> None:
        """Test decoding trivial syndrome."""
        from pecos_rslib.decoders import FusionBlossomDecoder

        decoder = FusionBlossomDecoder.from_check_matrix([[1, 1, 0], [0, 1, 1]])
        result = decoder.decode([0, 0])

        assert result.weight == 0.0

    def test_from_standard_code(self) -> None:
        """Test construction for standard codes (like CodeCapacityPlanarCode)."""
        from pecos_rslib.decoders import FusionBlossomDecoder

        decoder = FusionBlossomDecoder.from_standard_code(
            code_type="code_capacity_rotated",
            distance=3,
            error_rate=0.01,
        )
        assert decoder is not None
        assert decoder.num_nodes > 0

    def test_clear_for_reuse(self) -> None:
        """Test clear() for efficient decoder reuse."""
        from pecos_rslib.decoders import FusionBlossomDecoder

        decoder = FusionBlossomDecoder.from_check_matrix([[1, 1, 0], [0, 1, 1]])

        # Decode multiple syndromes with clear
        for _ in range(3):
            result = decoder.decode([0, 0])
            assert result is not None
            decoder.clear()


class TestBpResult:
    """Tests for unified BpResult type."""

    def test_result_attributes(self) -> None:
        """Test that BpResult has expected attributes."""
        from pecos_rslib.decoders import BpOsdDecoder, SparseMatrix

        H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        decoder = BpOsdDecoder(H, error_rate=0.01)
        result = decoder.decode([0, 0, 0])

        assert hasattr(result, "decoding")
        assert hasattr(result, "converged")
        assert hasattr(result, "iterations")

        assert isinstance(result.decoding, list)
        assert isinstance(result.converged, bool)
        assert isinstance(result.iterations, int)


class TestSparseMatrix:
    """Tests for SparseMatrix (used by LDPC decoders)."""

    def test_create_from_dense(self) -> None:
        """Test creation from dense matrix."""
        from pecos_rslib.decoders import SparseMatrix

        H = [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]
        matrix = SparseMatrix(H)

        assert matrix.rows == 3
        assert matrix.cols == 4
        assert matrix.nnz() == 6

    def test_create_from_coo(self) -> None:
        """Test creation from COO format."""
        from pecos_rslib.decoders import SparseMatrix

        matrix = SparseMatrix.from_coo(
            rows=2,
            cols=3,
            row_indices=[0, 0, 1, 1],
            col_indices=[0, 1, 1, 2],
        )

        assert matrix.rows == 2
        assert matrix.cols == 3


class TestBpOsdDecoder:
    """Tests for BpOsdDecoder (mirrors ldpc.bposd_decoder)."""

    def test_create_decoder(self) -> None:
        """Test construction (like ldpc's BpOsdDecoder)."""
        from pecos_rslib.decoders import BpOsdDecoder, SparseMatrix

        H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        decoder = BpOsdDecoder(
            H,
            error_rate=0.01,
            bp_method="product_sum",
            osd_method="osd0",
        )

        assert decoder is not None

    def test_decode_trivial(self) -> None:
        """Test decoding trivial syndrome."""
        from pecos_rslib.decoders import BpOsdDecoder, SparseMatrix

        H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        decoder = BpOsdDecoder(H, error_rate=0.01)

        result = decoder.decode([0, 0, 0])
        assert result.converged

    def test_bp_methods(self) -> None:
        """Test different BP methods."""
        from pecos_rslib.decoders import BpOsdDecoder, SparseMatrix

        H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])

        # product_sum
        decoder1 = BpOsdDecoder(H, error_rate=0.01, bp_method="product_sum")
        result1 = decoder1.decode([0, 0, 0])
        assert result1 is not None

        # minimum_sum
        decoder2 = BpOsdDecoder(H, error_rate=0.01, bp_method="minimum_sum")
        result2 = decoder2.decode([0, 0, 0])
        assert result2 is not None


class TestBpLsdDecoder:
    """Tests for BpLsdDecoder (mirrors ldpc.bplsd_decoder)."""

    def test_create_decoder(self) -> None:
        """Test construction."""
        from pecos_rslib.decoders import BpLsdDecoder, SparseMatrix

        H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        decoder = BpLsdDecoder(H, error_rate=0.01, lsd_order=0)

        assert decoder is not None

    def test_decode_trivial(self) -> None:
        """Test decoding trivial syndrome."""
        from pecos_rslib.decoders import BpLsdDecoder, SparseMatrix

        H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        decoder = BpLsdDecoder(H, error_rate=0.01)

        result = decoder.decode([0, 0, 0])
        assert result is not None


class TestUnionFindDecoder:
    """Tests for UnionFindDecoder."""

    def test_create_decoder(self) -> None:
        """Test construction."""
        from pecos_rslib.decoders import SparseMatrix, UnionFindDecoder

        H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        decoder = UnionFindDecoder(H, method="inversion")

        assert decoder is not None

    def test_decode_trivial(self) -> None:
        """Test decoding trivial syndrome."""
        from pecos_rslib.decoders import SparseMatrix, UnionFindDecoder

        H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        decoder = UnionFindDecoder(H)

        result = decoder.decode([0, 0, 0])
        assert result is not None

    def test_methods(self) -> None:
        """Test different UF methods."""
        from pecos_rslib.decoders import SparseMatrix, UnionFindDecoder

        H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])

        decoder_inv = UnionFindDecoder(H, method="inversion")
        result_inv = decoder_inv.decode([0, 0, 0])
        assert result_inv is not None

        decoder_peel = UnionFindDecoder(H, method="peeling")
        result_peel = decoder_peel.decode([0, 0, 0])
        assert result_peel is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
