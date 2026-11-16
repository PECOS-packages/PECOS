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

import sys

import pytest


class TestMwpmResult:
    """Tests for unified MwpmResult type."""

    def test_result_attributes(self) -> None:
        """Test that MwpmResult has the expected attributes."""
        from pecos_rslib.decoders import CheckMatrix, ObservableFlips, PyMatchingDecoder

        matrix = CheckMatrix.from_dense([[1, 1, 0], [0, 1, 1]])
        decoder = PyMatchingDecoder.from_check_matrix(matrix)
        result = decoder.decode_syndrome([0, 0])

        # Check attributes exist
        assert hasattr(result, "observable_flips")
        assert hasattr(result, "weight")

        # Check types
        assert isinstance(result.observable_flips, ObservableFlips)
        assert isinstance(result.weight, float)

    def test_observable_flips_materializes_to_list(self) -> None:
        """Test materializing MwpmResult.observable_flips as a list."""
        from pecos_rslib.decoders import CheckMatrix, PyMatchingDecoder

        matrix = CheckMatrix.from_dense([[1, 1, 0], [0, 1, 1]])
        decoder = PyMatchingDecoder.from_check_matrix(matrix)
        result = decoder.decode_syndrome([0, 0])

        assert list(result.observable_flips) == [bool(value) for value in result]

    def test_result_indexing(self) -> None:
        """Test MwpmResult supports indexing like a list."""
        from pecos_rslib.decoders import CheckMatrix, PyMatchingDecoder

        matrix = CheckMatrix.from_dense([[1, 1, 0], [0, 1, 1]])
        decoder = PyMatchingDecoder.from_check_matrix(matrix)
        result = decoder.decode_syndrome([0, 0])

        assert len(result) == len(result.observable_flips)
        if len(result) > 0:
            assert bool(result[0]) == result.observable_flips[0]


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

        result = decoder.decode_syndrome([0, 0])

        # No errors - should have zero weight
        assert result.weight == 0.0

    def test_decode_single_error(self) -> None:
        """Test decoding syndrome from single error."""
        from pecos_rslib.decoders import CheckMatrix, PyMatchingDecoder

        # Repetition code: error on qubit 1 gives syndrome [1, 1]
        H = CheckMatrix.from_dense([[1, 1, 0], [0, 1, 1]])
        decoder = PyMatchingDecoder.from_check_matrix(H)

        result = decoder.decode_syndrome([1, 1])
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

    def test_from_dem_with_correlations(self) -> None:
        """Test construction from DEM with decomposition correlations enabled."""
        from pecos_rslib.decoders import PyMatchingDecoder

        dem = "error(0.1) D0 D1 ^ D2 L0"
        decoder = PyMatchingDecoder.from_dem_with_correlations(dem)

        result = decoder.decode_syndrome([0, 0, 0])
        assert list(result.observable_flips) == [False]


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
        result = decoder.decode_syndrome([0, 0])

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
            result = decoder.decode_syndrome([0, 0])
            assert result is not None
            decoder.clear()


class TestBpResult:
    """Tests for unified BpResult type."""

    def test_result_attributes(self) -> None:
        """Test that BpResult has expected attributes."""
        from pecos_rslib.decoders import BpOsdBuilder, SparseMatrix

        H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        decoder = BpOsdBuilder(H, error_rate=0.01).build()
        result = decoder.decode_syndrome([0, 0, 0])

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
    """Tests for BpOsdDecoder via BpOsdBuilder."""

    def test_create_decoder(self) -> None:
        """Test construction via builder."""
        from pecos_rslib.decoders import BpOsdBuilder, SparseMatrix

        H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        decoder = BpOsdBuilder(H, error_rate=0.01).bp_method("product_sum").osd_method("osd0").build()

        assert decoder is not None

    def test_decode_trivial(self) -> None:
        """Test decoding trivial syndrome."""
        from pecos_rslib.decoders import BpOsdBuilder, SparseMatrix

        H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        decoder = BpOsdBuilder(H, error_rate=0.01).build()

        result = decoder.decode_syndrome([0, 0, 0])
        assert result.converged

    def test_bp_methods(self) -> None:
        """Test different BP methods."""
        from pecos_rslib.decoders import BpOsdBuilder, SparseMatrix

        H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])

        # product_sum
        decoder1 = BpOsdBuilder(H, error_rate=0.01).bp_method("product_sum").build()
        result1 = decoder1.decode_syndrome([0, 0, 0])
        assert result1 is not None

        # minimum_sum
        decoder2 = BpOsdBuilder(H, error_rate=0.01).bp_method("minimum_sum").build()
        result2 = decoder2.decode_syndrome([0, 0, 0])
        assert result2 is not None

    def test_per_column_error_channel(self) -> None:
        """A per-column channel must steer the decode, not just be accepted."""
        from pecos_rslib.decoders import BpOsdBuilder, SparseMatrix

        H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        # Syndrome [1, 1, 1] is explained by columns {1, 3} or {0, 2};
        # the channel priors decide which support wins.
        favor_1_3 = BpOsdBuilder(H, error_rate=[0.001, 0.3, 0.001, 0.3]).build()
        assert list(favor_1_3.decode_syndrome([1, 1, 1]).decoding) == [0, 1, 0, 1]
        favor_0_2 = BpOsdBuilder(H, error_rate=[0.3, 0.001, 0.3, 0.001]).build()
        assert list(favor_0_2.decode_syndrome([1, 1, 1]).decoding) == [1, 0, 1, 0]

    def test_error_channel_length_mismatch_rejected(self) -> None:
        """A channel whose length differs from the column count fails at build."""
        import pytest
        from pecos_rslib.decoders import BpOsdBuilder, SparseMatrix

        H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        with pytest.raises(ValueError, match="length must match"):
            BpOsdBuilder(H, error_rate=[0.01, 0.02]).build()

    @pytest.mark.parametrize("bad", [float("nan"), -0.1, 1.5])
    def test_non_probability_prior_rejected(self, bad: float) -> None:
        """A rate or channel entry outside [0, 1] fails at build, naming the argument."""
        from pecos_rslib.decoders import BpLsdBuilder, BpOsdBuilder, SparseMatrix

        H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        for builder in (BpOsdBuilder, BpLsdBuilder):
            with pytest.raises(ValueError, match=r"error_rate must be finite and in \[0, 1\]"):
                builder(H, error_rate=bad).build()
            with pytest.raises(ValueError, match=r"error_channel\[2\] must be finite"):
                builder(H, error_rate=[0.01, 0.01, bad, 0.01]).build()


class TestBpLsdDecoder:
    """Tests for BpLsdDecoder via BpLsdBuilder."""

    def test_create_decoder(self) -> None:
        """Test construction via builder."""
        from pecos_rslib.decoders import BpLsdBuilder, SparseMatrix

        H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        decoder = BpLsdBuilder(H, error_rate=0.01).lsd_order(0).build()

        assert decoder is not None

    def test_decode_trivial(self) -> None:
        """Test decoding trivial syndrome."""
        from pecos_rslib.decoders import BpLsdBuilder, SparseMatrix

        H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        decoder = BpLsdBuilder(H, error_rate=0.01).build()

        result = decoder.decode([0, 0, 0])
        assert result is not None

    def test_per_column_error_channel(self) -> None:
        """A per-column channel must steer the decode, not just be accepted."""
        from pecos_rslib.decoders import BpLsdBuilder, SparseMatrix

        H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        favor_1_3 = BpLsdBuilder(H, error_rate=[0.001, 0.3, 0.001, 0.3]).build()
        assert list(favor_1_3.decode([1, 1, 1]).decoding) == [0, 1, 0, 1]
        favor_0_2 = BpLsdBuilder(H, error_rate=[0.3, 0.001, 0.3, 0.001]).build()
        assert list(favor_0_2.decode([1, 1, 1]).decoding) == [1, 0, 1, 0]


class TestUnionFindDecoder:
    """Tests for UnionFindDecoder via UnionFindBuilder."""

    def test_create_decoder(self) -> None:
        """Test construction via builder."""
        from pecos_rslib.decoders import SparseMatrix, UnionFindBuilder

        H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        decoder = UnionFindBuilder(H).method("inversion").build()

        assert decoder is not None

    def test_decode_trivial(self) -> None:
        """Test decoding trivial syndrome."""
        from pecos_rslib.decoders import SparseMatrix, UnionFindBuilder

        H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        decoder = UnionFindBuilder(H).build()

        result = decoder.decode_syndrome([0, 0, 0])
        assert result is not None

    def test_methods(self) -> None:
        """Test different UF methods."""
        from pecos_rslib.decoders import SparseMatrix, UnionFindBuilder

        H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])

        decoder_inv = UnionFindBuilder(H).method("inversion").build()
        result_inv = decoder_inv.decode_syndrome([0, 0, 0])
        assert result_inv is not None

        decoder_peel = UnionFindBuilder(H).method("peeling").build()
        result_peel = decoder_peel.decode_syndrome([0, 0, 0])
        assert result_peel is not None

    def test_wrong_syndrome_length_raises_value_error(self) -> None:
        """A syndrome whose length differs from the check count is a caller error."""
        from pecos_rslib.decoders import DemAwareDecoder, SparseMatrix, UnionFindBuilder

        H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
        decoder = UnionFindBuilder(H).build()
        with pytest.raises(ValueError, match="Syndrome length 2 does not match check count 3"):
            decoder.decode_syndrome([0, 1])

        dem = "error(0.1) D0 L0\nerror(0.1) D0 D1\nerror(0.1) D1"
        dem_decoder = DemAwareDecoder.from_dem(dem, decoder_type="union_find")
        with pytest.raises(ValueError, match="Syndrome length 3 does not match check count 2"):
            dem_decoder.decode_syndrome([0, 1, 1])


class TestMinSumBpDecoder:
    """Tests for MinSumBpDecoder via MinSumBpBuilder."""

    def test_create_decoder(self) -> None:
        """Test construction via builder."""
        from pecos_rslib.decoders import MinSumBpBuilder

        H = [[1, 1, 0], [0, 1, 1]]
        decoder = MinSumBpBuilder(H, error_priors=[0.003, 0.003, 0.003]).build()

        assert decoder is not None

    def test_decode_trivial(self) -> None:
        """Test decoding zero syndrome."""
        from pecos_rslib.decoders import MinSumBpBuilder

        H = [[1, 1, 0], [0, 1, 1]]
        decoder = MinSumBpBuilder(H, error_priors=[0.003, 0.003, 0.003]).build()

        result = decoder.decode([0, 0])
        assert result.converged
        assert result.decoding == [0, 0, 0]

    def test_decode_single_error(self) -> None:
        """Test decoding a single-error syndrome."""
        from pecos_rslib.decoders import MinSumBpBuilder

        H = [[1, 1, 0], [0, 1, 1]]
        decoder = MinSumBpBuilder(H, error_priors=[0.003, 0.003, 0.003]).build()

        result = decoder.decode([1, 0])
        assert result.converged
        assert result.decoding == [1, 0, 0]

    def test_builder_chaining(self) -> None:
        """Test builder method chaining."""
        from pecos_rslib.decoders import MinSumBpBuilder

        H = [[1, 1, 0], [0, 1, 1]]
        decoder = MinSumBpBuilder(H, error_priors=[0.003, 0.003, 0.003]).max_iter(100).alpha(0.8).gamma0(0.5).build()

        result = decoder.decode([0, 0])
        assert result.converged

    def test_properties(self) -> None:
        """Test check_count and bit_count properties."""
        from pecos_rslib.decoders import MinSumBpBuilder

        H = [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]
        decoder = MinSumBpBuilder(H, error_priors=[0.01, 0.01, 0.01, 0.01]).build()

        assert decoder.check_count == 3
        assert decoder.bit_count == 4


class TestRelayBpDecoder:
    """Tests for RelayBpDecoder via RelayBpBuilder."""

    def test_create_decoder(self) -> None:
        """Test construction via builder."""
        from pecos_rslib.decoders import RelayBpBuilder

        H = [[1, 1, 0], [0, 1, 1]]
        decoder = RelayBpBuilder(H, error_priors=[0.003, 0.003, 0.003]).build()

        assert decoder is not None

    def test_decode_trivial(self) -> None:
        """Test decoding zero syndrome."""
        from pecos_rslib.decoders import RelayBpBuilder

        H = [[1, 1, 0], [0, 1, 1]]
        decoder = RelayBpBuilder(H, error_priors=[0.003, 0.003, 0.003]).seed(42).build()

        result = decoder.decode([0, 0])
        assert result.converged
        assert result.decoding == [0, 0, 0]

    def test_decode_single_error(self) -> None:
        """Test decoding a single-error syndrome."""
        from pecos_rslib.decoders import RelayBpBuilder

        H = [[1, 1, 0], [0, 1, 1]]
        decoder = RelayBpBuilder(H, error_priors=[0.003, 0.003, 0.003]).seed(42).build()

        result = decoder.decode([1, 0])
        assert result.converged
        assert result.decoding == [1, 0, 0]

    def test_builder_chaining(self) -> None:
        """Test builder method chaining with relay-specific options."""
        from pecos_rslib.decoders import RelayBpBuilder

        H = [[1, 1, 0], [0, 1, 1]]
        decoder = (
            RelayBpBuilder(H, error_priors=[0.003, 0.003, 0.003])
            .gamma0(0.9)
            .pre_iter(40)
            .num_sets(20)
            .set_max_iter(30)
            .seed(42)
            .stopping("n_conv_1")
            .build()
        )

        result = decoder.decode([0, 0])
        assert result.converged

    def test_properties(self) -> None:
        """Test check_count and bit_count properties."""
        from pecos_rslib.decoders import RelayBpBuilder

        H = [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]
        decoder = RelayBpBuilder(H, error_priors=[0.01, 0.01, 0.01, 0.01]).build()

        assert decoder.check_count == 3
        assert decoder.bit_count == 4

    def test_seed_reproducibility(self) -> None:
        """Test that same seed gives same results."""
        from pecos_rslib.decoders import RelayBpBuilder

        H = [[1, 1, 0], [0, 1, 1]]
        priors = [0.003, 0.003, 0.003]

        d1 = RelayBpBuilder(H, error_priors=priors).seed(123).build()
        d2 = RelayBpBuilder(H, error_priors=priors).seed(123).build()

        r1 = d1.decode([1, 0])
        r2 = d2.decode([1, 0])

        assert r1.decoding == r2.decoding
        assert r1.converged == r2.converged


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


def test_experimental_factory_names_the_missing_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pecos_rslib_exp", None)
    import pecos.decoders as decoders

    with pytest.raises(
        ImportError,
        match=r"frontier requires the optional pecos-rslib-exp package.*not published to PyPI.*just build",
    ):
        _ = decoders.frontier


def test_relay_builder_removed_max_iter_and_zero_convergence_target() -> None:
    from pecos_rslib.decoders import RelayBpBuilder

    builder = RelayBpBuilder([[1, 1, 0], [0, 1, 1]], [0.1] * 3)
    with pytest.raises(AttributeError, match="max_iter"):
        builder.max_iter(100)
    with pytest.raises(ValueError, match="positive integer"):
        builder.stopping("n_conv_0").build()


@pytest.mark.parametrize("family", ["RelayBpBuilder", "MinSumBpBuilder"])
def test_relay_family_input_errors_are_value_errors(family: str) -> None:
    import pecos_rslib.decoders as decoders

    builder = getattr(decoders, family)
    with pytest.raises(ValueError, match=r"check_matrix.*2"):
        builder([[1, 2]], [0.1, 0.1]).build()
    for priors in [[-0.1, 0.1], [1.1, 0.1], [float("nan"), 0.1], [float("inf"), 0.1], [0.1]]:
        with pytest.raises(ValueError, match="probability"):
            builder([[1, 1]], priors).build()
    decoder = builder([[1, 1]], [0.1, 0.1]).build()
    with pytest.raises(ValueError, match=r"syndrome.*2"):
        decoder.decode([2])
    with pytest.raises(ValueError, match="syndrome length"):
        decoder.decode([0, 0])
    with pytest.raises(ValueError, match="alpha"):
        builder([[1, 1]], [0.1, 0.1]).alpha(-1.0).build()


def test_min_sum_zero_iterations_is_value_error() -> None:
    from pecos_rslib.decoders import MinSumBpBuilder

    with pytest.raises(ValueError, match=r"max_iter.*0"):
        MinSumBpBuilder([[1, 1]], [0.1, 0.1]).max_iter(0).build()


def test_relay_internal_decode_failure_remains_runtime_error() -> None:
    from pecos_rslib.decoders import MinSumBpBuilder

    decoder = MinSumBpBuilder([[1, 1]], [0.1, 0.1]).gamma0(1e308).build()
    with pytest.raises(RuntimeError, match="non-finite posterior"):
        decoder.decode([1])


@pytest.fixture
def native_bp_batch_data() -> tuple[list[list[int]], list[list[int]]]:
    """Sparse code with short cycles exercises relay legs on 50 weight-two shots."""
    import numpy as np

    rng = np.random.default_rng(712)
    matrix = np.zeros((30, 60), dtype=np.uint8)
    for column in range(60):
        matrix[rng.choice(30, 3, replace=False), column] = 1
    errors = np.zeros((50, 60), dtype=np.uint8)
    for shot in errors:
        shot[rng.choice(60, 2, replace=False)] = 1
    return matrix.tolist(), ((errors @ matrix.T) % 2).tolist()


def test_relay_explicit_gammas(native_bp_batch_data: tuple[list[list[int]], list[list[int]]]) -> None:
    from pecos_rslib.decoders import RelayBpBuilder, RelayBpDecoder

    matrix, shots = native_bp_batch_data
    rows = [[-0.8] * 60, [0.9] * 60]

    def builder() -> RelayBpBuilder:
        return RelayBpBuilder(matrix, [0.05] * 60).seed(42).pre_iter(0).num_sets(5).set_max_iter(3).stopping("all")

    def results(decoder: RelayBpDecoder) -> list[tuple[list[int], bool, int]]:
        decoded = [decoder.decode(shot) for shot in shots]
        return [(result.decoding, result.converged, result.iterations) for result in decoded]

    explicit = results(builder().explicit_gammas(rows).build())
    assert explicit == results(builder().explicit_gammas(rows).build())
    assert explicit == results(builder().explicit_gammas((rows * 3)[:5]).build())
    assert explicit != results(builder().build())
    assert results(builder().explicit_gammas(rows).explicit_gammas(None).build()) == results(builder().build())
    assert results(builder().gamma_dist_interval(0.3, 0.3).build()) == results(
        builder().explicit_gammas([[0.3] * 60]).build(),
    )


@pytest.mark.parametrize("interval", [(1.0, -1.0), (float("nan"), 1.0), (0.0, float("inf")), (-1e308, 1e308)])
def test_relay_gamma_interval_validation(interval: tuple[float, float]) -> None:
    from pecos_rslib.decoders import RelayBpBuilder

    with pytest.raises(ValueError, match=r"gamma_dist_interval.*got \("):
        RelayBpBuilder([[1, 1]], [0.1, 0.1]).gamma_dist_interval(*interval).build()


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([], r"explicit_gammas.*row count 0"),
        ([[0.1]], r"explicit_gammas.*column count 1"),
        ([[0.1, float("inf")]], r"explicit_gammas\[0\]\[1\].*inf"),
    ],
)
def test_relay_explicit_gamma_validation(rows: list[list[float]], message: str) -> None:
    from pecos_rslib.decoders import RelayBpBuilder

    with pytest.raises(ValueError, match=message):
        RelayBpBuilder([[1, 1]], [0.1, 0.1]).explicit_gammas(rows).build()


@pytest.mark.parametrize("family", ["RelayBpBuilder", "MinSumBpBuilder"])
def test_native_bp_decode_batch(
    family: str,
    native_bp_batch_data: tuple[list[list[int]], list[list[int]]],
) -> None:
    import numpy as np
    import pecos_rslib.decoders as decoders

    matrix, shots = native_bp_batch_data
    builder = getattr(decoders, family)(matrix, [0.05] * 60)
    if family == "RelayBpBuilder":
        builder.seed(42).pre_iter(2).num_sets(5).set_max_iter(10)
    decoder = builder.build()
    expected = bytes(bit for shot in shots for bit in decoder.decode(shot).decoding)
    flat = np.asarray(shots).astype(np.uint8).tobytes()
    actual = decoder.decode_batch(flat, num_shots=50)
    assert isinstance(actual, bytes)
    assert len(actual) == 50 * decoder.bit_count
    assert actual == expected
    split = 17 * decoder.check_count
    assert decoder.decode_batch(flat[:split], 17) + decoder.decode_batch(flat[split:], 33) == expected
    assert decoder.decode_batch(b"", 0) == b""
    for invalid in [flat[:-1], flat + b"\x00"]:
        with pytest.raises(ValueError, match=rf"length {len(invalid)}.*num_shots \(50\).*check_count \(30\)"):
            decoder.decode_batch(invalid, 50)
    with pytest.raises(ValueError, match=r"syndrome.*2"):
        decoder.decode_batch(flat[:-1] + b"\x02", 50)
    with pytest.raises(ValueError, match=r"num_shots.*check_count"):
        decoder.decode_batch(b"", sys.maxsize)
    assert decoder.decode_batch(flat, 50) == expected
