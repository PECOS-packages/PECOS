# Copyright 2026 The PECOS Developers
#
# Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
# in compliance with the License. You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under the License
# is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing permissions and limitations under
# the License.

"""Uniform observable-flip values across decoder predictions and sampled truth."""

from __future__ import annotations

import pytest
from pecos_rslib.decoders import (
    BpOsdBuilder,
    BpOsdDecoder,
    DemAwareDecoder,
    ObservableFlips,
    PyMatchingDecoder,
    SparseMatrix,
    TesseractDecoder,
)
from pecos_rslib.qec import (
    ObservableFlips as QecObservableFlips,
)
from pecos_rslib.qec import (
    SampleBatch,
)

_ONE_OBSERVABLE_DEM = """detector D0
detector D1
logical_observable L0
error(0.1) D0
error(0.1) D1 L0
"""


def _wide_dem(num_observables: int = 71) -> str:
    lines = ["error(0.1) D0 L0", "error(0.1) D1 L70", "detector D0", "detector D1"]
    lines += [f"logical_observable L{index}" for index in range(num_observables)]
    return "\n".join(lines)


def test_same_observable_flips_type_is_exported_from_both_namespaces() -> None:
    assert QecObservableFlips is ObservableFlips


def test_indexing_iteration_indices_mask_and_repr() -> None:
    flips = ObservableFlips.from_mask(0b101, 3)

    assert len(flips) == 3
    assert [flips[index] for index in range(len(flips))] == [True, False, True]
    assert flips[-1] is True
    assert flips[-2] is False
    assert flips[-3] is True
    assert list(flips) == [True, False, True]
    assert flips.indices() == [0, 2]
    assert flips.mask == 0b101
    assert repr(flips) == "ObservableFlips(num_observables=3, mask=5)"

    for index in (3, -4):
        with pytest.raises(IndexError) as error:
            _ = flips[index]
        assert str(index) in str(error.value)
        assert "num_observables=3" in str(error.value)


def test_equality_requires_same_type_bits_and_length() -> None:
    flips = ObservableFlips.from_mask(1, 2)

    assert flips == ObservableFlips.from_bits([True, False])
    assert flips != ObservableFlips.from_bits([False, True])
    assert flips != ObservableFlips.from_mask(1, 3)
    assert flips.__eq__([True, False]) is NotImplemented
    assert flips.__eq__(1) is NotImplemented
    assert (flips == [True, False]) is False
    assert (flips == 1) is False


def test_from_mask_rejects_high_bits_and_round_trips() -> None:
    with pytest.raises(ValueError, match=rf"mask={1 << 5}.*num_observables=2"):
        ObservableFlips.from_mask(1 << 5, 2)

    mask = (1 << 70) | (1 << 2)
    flips = ObservableFlips.from_mask(mask, 71)
    assert flips.mask == mask
    assert flips[70] is True
    assert flips[69] is False


def test_constructors_accept_integer_like_values() -> None:
    """The accessors this type bridges from hand back ints, and masks arrive as NumPy scalars.

    Both constructors go through ``__index__``, so ``int``, ``bool`` and NumPy
    integer scalars are all accepted on the same footing.
    """
    numpy = pytest.importorskip("numpy")

    # ``MwpmResult.correction`` and ``TesseractResult.observable_bits`` return list[int].
    assert ObservableFlips.from_bits([1, 0, 1]) == ObservableFlips.from_mask(0b101, 3)
    assert ObservableFlips.from_bits([True, 0, 1]) == ObservableFlips.from_mask(0b101, 3)
    assert ObservableFlips.from_bits(list(numpy.array([1, 0, 1]))) == ObservableFlips.from_mask(0b101, 3)
    assert ObservableFlips.from_bits([numpy.True_, numpy.False_]) == ObservableFlips.from_mask(0b01, 2)

    assert ObservableFlips.from_mask(numpy.uint64(5), 3) == ObservableFlips.from_mask(5, 3)
    assert ObservableFlips.from_mask(numpy.int64(5), 3) == ObservableFlips.from_mask(5, 3)


def test_constructors_reject_non_integers_and_non_bits() -> None:
    # Truthiness is never used: a non-bit integer is an error, not something to coerce.
    with pytest.raises(ValueError, match="bit at index 1 must be 0 or 1, got 2"):
        ObservableFlips.from_bits([1, 2, 0])

    # A missing __index__ is "not an integer", which Python reports as TypeError.
    with pytest.raises(TypeError, match="cannot be interpreted as an integer"):
        ObservableFlips.from_bits(["a"])
    with pytest.raises(TypeError, match="cannot be interpreted as an integer"):
        ObservableFlips.from_mask("x", 3)

    with pytest.raises(ValueError, match="mask=-1 is negative"):
        ObservableFlips.from_mask(-1, 3)


def test_from_bits_accepts_an_iterable_and_indices_agree_with_getitem() -> None:
    bits = [False, True, False, True]
    flips = ObservableFlips.from_bits(bit for bit in bits)

    assert list(flips) == bits
    assert flips.indices() == [index for index in range(len(flips)) if flips[index]]


def test_sample_batch_observable_metadata_and_shot_bounds() -> None:
    batch = SampleBatch([[0], [1]], [0, 3])

    assert batch.num_shots == 2
    assert batch.num_observables == 2
    assert batch.get_observable_flips(1) == ObservableFlips.from_mask(3, 2)
    with pytest.raises(IndexError, match=r"Shot index 2.*num_shots=2"):
        batch.get_observable_flips(2)


def test_uniform_loop_preserves_one_observable_error_counts() -> None:
    syndromes = [[0, 0], [1, 0], [0, 1], [1, 1]]
    batch = SampleBatch(syndromes, [0, 1, 0, 1])
    decoders = {
        "pymatching": PyMatchingDecoder.from_dem(_ONE_OBSERVABLE_DEM),
        "tesseract": TesseractDecoder.from_dem(_ONE_OBSERVABLE_DEM),
        "bp_osd": BpOsdDecoder.from_dem(_ONE_OBSERVABLE_DEM),
    }
    old_counts = dict.fromkeys(decoders, 0)
    new_counts = dict.fromkeys(decoders, 0)

    for shot in range(batch.num_shots):
        syndrome = batch.get_syndrome(shot)
        actual_mask = batch.get_observable_mask(shot) & 1
        actual_flips = batch.get_observable_flips(shot)
        results = {name: decoder.decode_syndrome(syndrome) for name, decoder in decoders.items()}

        old_counts["pymatching"] += results["pymatching"].correction[0] != actual_mask
        old_counts["tesseract"] += results["tesseract"].observables_mask & 1 != actual_mask
        old_counts["bp_osd"] += results["bp_osd"].observables_mask & 1 != actual_mask
        for name, result in results.items():
            new_counts[name] += result.observable_flips != actual_flips

    assert old_counts == new_counts == {"pymatching": 2, "tesseract": 2, "bp_osd": 2}


def test_any_observable_and_per_observable_counts_are_distinct() -> None:
    batch = SampleBatch([[], []], [0, 3])
    predictions = [ObservableFlips.from_mask(1, 2), ObservableFlips.from_mask(1, 2)]
    any_observable_errors = 0
    per_observable_errors = [0] * batch.num_observables

    for shot, predicted in enumerate(predictions):
        actual = batch.get_observable_flips(shot)
        any_observable_errors += predicted != actual
        for index in range(batch.num_observables):
            per_observable_errors[index] += predicted[index] != actual[index]

    assert any_observable_errors == 2
    assert per_observable_errors == [1, 1]


def test_wide_observables_are_not_truncated_end_to_end() -> None:
    dem = _wide_dem()
    syndrome = [0, 1]
    batch = SampleBatch([syndrome], [1 << 70])
    actual = batch.get_observable_flips(0)
    decoders = [
        PyMatchingDecoder.from_dem(dem),
        DemAwareDecoder.from_dem(dem, decoder_type="bp_osd"),
    ]

    assert batch.num_observables == 71
    assert actual.mask == 1 << 70
    assert actual[70] is True
    assert actual[6] is False
    for decoder in decoders:
        predicted = decoder.decode_syndrome(syndrome).observable_flips
        assert predicted == actual
        assert len(predicted) == 71
        assert predicted.mask == 1 << 70
        assert predicted[70] is True
        assert predicted[6] is False


def test_bp_result_does_not_fabricate_observable_flips() -> None:
    decoder = BpOsdBuilder(SparseMatrix([[1]]), error_rate=0.1).build()
    result = decoder.decode_syndrome([0])

    assert not hasattr(result, "observable_flips")
